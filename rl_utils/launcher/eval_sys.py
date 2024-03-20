import argparse
import os
import os.path as osp
import re
import shlex
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from omegaconf import OmegaConf

from rl_utils.common.core_utils import logger
from rl_utils.launcher.run_exp import get_random_id, sub_in_args, sub_in_vars
from rl_utils.plotting.wb_query import get_run_command

RUN_DIR = "data/log/runs/"


def eval_ckpt(
    run_id,
    model_ckpt_path,
    new_run_id,
    cfg,
    proj_dat,
    modify_run_cmd_fn,
    args: argparse.Namespace,
    rest,
):
    eval_sys_cfg = cfg.eval_sys
    # Find the run command.
    run_path = osp.join(RUN_DIR, run_id + ".sh")
    if osp.exists(run_path):
        logger.info("Substituting slurm run command.")
        # Get the actually executed command.
        with open(run_path, "r") as f:
            cmd = f.readlines()[-1]
        cmd_parts = split_cmd_txt(cmd)
    else:
        logger.info("Substituting in the run command.")
        if args.cmd is None:
            cmd = get_run_command(
                run_id, eval_sys_cfg.wandb_run_name_k, cfg.wb_entity, cfg.proj_name
            )
        else:
            cmd = eval_sys_cfg.eval_run_cmd[args.cmd]
        add_all = cfg.get("add_all", None)
        if add_all is not None and args.cmd is not None:
            # If `args.cmd` is None, then we used the old command which already
            # has the `add_all` content included.
            cmd = sub_in_args(cmd, add_all)
        cmd = sub_in_vars(cmd, cfg, 0, "eval")
        ident = get_random_id()
        cmd = cmd.replace("$SLURM_ID", ident)

        cmd_parts = split_cmd_txt(cmd)
        # Dummy line for srun
        cmd_parts.insert(0, "")

    cmd_parts = sub_in_eval_type(cmd_parts, args, eval_sys_cfg)

    cmd_parts = change_arg_vals(
        cmd_parts,
        {
            **{k: add_eval_suffix for k in eval_sys_cfg.add_eval_to_vals},
            **eval_sys_cfg.change_vals,
        },
    )

    cmd_parts.extend(
        [
            eval_sys_cfg.ckpt_load_k,
            model_ckpt_path,
        ]
    )
    add_env_vars = cfg.add_env_vars
    if proj_dat is not None:
        for k in proj_dat.split(","):
            cmd_parts.extend(split_cmd_txt(cfg.proj_data[k]))
            add_env_vars.append(cfg.get("proj_dat_add_env_vars", {}).get(k, ""))

    cmd_parts = cmd_parts[1:]
    cmd_parts = [*add_env_vars, *cmd_parts]

    if modify_run_cmd_fn is not None:
        cmd_parts = modify_run_cmd_fn(cmd_parts, new_run_id, args)

    python_file = -1
    for i, cmd_part in enumerate(cmd_parts):
        if ".py" in cmd_part:
            python_file = i
            break
    new_cmd = (" ".join(cmd_parts[: python_file + 1])) + " "
    for i in range(python_file + 1, len(cmd_parts) - 1, 2):
        k, v = cmd_parts[i], cmd_parts[i + 1]
        if k.startswith("--"):
            sep = " "
        else:
            sep = eval_sys_cfg.sep
        new_cmd += f" {k}{sep}{v}"
    if args.cd is not None:
        new_cmd = f"CUDA_VISIBLE_DEVICES={args.cd} {new_cmd}"
    new_cmd = sub_in_vars(new_cmd, cfg, 0, "eval")
    new_cmd += " " + shlex.join(rest)

    # Remove the accelerate launch command.
    new_cmd = re.sub(
        r"accelerate launch --main_process_port (\d+) --num_processes (\d+) --config_file ",
        "",
        new_cmd,
    )
    for orig, sub in eval_sys_cfg.get("replace_strs", {}).items():
        new_cmd = new_cmd.replace(orig, sub)

    logger.info(f"EVALUATING {new_cmd}")
    os.system(new_cmd)
    return True


def get_ckpt_path_search(cfg, eval_sys_cfg, args, run_id) -> str:
    full_path = osp.join(cfg.base_data_dir, eval_sys_cfg.ckpt_search_dir, run_id)

    if not args.force_search:
        if osp.exists(full_path):
            return full_path
        else:
            raise ValueError(
                f"Could not find {run_id} from {eval_sys_cfg.ckpt_search_dir}"
            )

    search_cmd = eval_sys_cfg.search_cmd.replace("$RUN_ID", run_id)

    logger.info(f"Searching with {search_cmd}")
    process = subprocess.Popen(shlex.split(search_cmd), stdout=subprocess.PIPE)
    (output, err) = process.communicate()
    process.wait()
    output = output.decode("UTF-8").rstrip()

    ckpt_idxs = [
        int(f.split(".")[1]) for f in output.split("\n") if ".pth" in f and "ckpt" in f
    ]

    if args.idx is None:
        last_idx = max(ckpt_idxs)
    else:
        last_idx = args.idx

    ckpt_name = f"ckpt.{last_idx}.pth"
    full_path = osp.join(full_path, ckpt_name)
    download_cmd = (
        eval_sys_cfg.download_cmd.replace("$RUN_ID", run_id)
        .replace("$CKPT", ckpt_name)
        .replace("$FULL_PATH", full_path)
    )
    logger.info(f"Downloading with {download_cmd}")
    os.system(download_cmd)

    return full_path


def watch_directory(path, interval=1):
    print(f"Watching directory: {path} every {interval} seconds")
    already_seen = set(os.listdir(path))
    while True:
        time.sleep(interval)
        current_files = set(os.listdir(path))
        new_files = current_files - already_seen
        if new_files:
            for file in new_files:
                yield os.path.join(path, file)
            already_seen = current_files


def get_ckpt_full_path(cfg, eval_sys_cfg, args, run_id) -> str:
    full_path = get_ckpt_path_search(cfg, eval_sys_cfg, args, run_id)
    ckpt_idxs = [
        int(f.split(".")[1])
        for f in os.listdir(full_path)
        if ".pth" in f and "ckpt" in f
    ]
    if args.all:
        return [osp.join(full_path, f"ckpt.{i}.pth") for i in ckpt_idxs]
    elif args.idx is None:
        last_idx = max(ckpt_idxs)
    else:
        last_idx = args.idx

    return [osp.join(full_path, f"ckpt.{last_idx}.pth")]


def add_eval_suffix(x):
    x = x.strip()
    rnd = get_random_id()[:3]
    if x == "":
        return f"{rnd}_eval"
    elif x[-1] == "/":
        return f"{x[:-1]}_eval{rnd}/"
    elif "." in x:
        parts = x.split(".")
        return f"{parts[0]}_eval{rnd}.{parts[1]}"
    else:
        return f"{x}_eval{rnd}"


def change_arg_vals(cmd_parts: List[str], new_arg_values: Dict[str, Any]) -> List[str]:
    """
    If the argument value does not exist, it will be added.
    :param new_arg_values: If the value is a function, it will take as input
        the current argument value and return the new argument value.
    """
    did_find = {k: False for k in new_arg_values.keys()}
    for i in range(len(cmd_parts) - 1):
        if cmd_parts[i] in new_arg_values:
            replace_val = new_arg_values[cmd_parts[i]]
            if isinstance(replace_val, Callable):
                cmd_parts[i + 1] = replace_val(cmd_parts[i + 1])
            else:
                cmd_parts[i + 1] = replace_val
            did_find[cmd_parts[i]] = True
    all_not_found_k = [k for k, did_find in did_find.items() if not did_find]
    for not_found_k in all_not_found_k:
        new_val = new_arg_values[not_found_k]
        if isinstance(new_val, Callable):
            new_val = new_val("")

        cmd_parts.extend([not_found_k, new_val])

    return cmd_parts


def split_cmd_txt(cmd: str) -> List[str]:
    cmd = cmd.replace("='", '="DELIM').replace("'", "'" + '"').replace("DELIM", "'")
    return [y for x in shlex.split(cmd, posix=True) for y in x.split("=")]


def sub_in_eval_type(cmd_parts: List[str], args, eval_sys_cfg) -> List[str]:
    if args.eval is None:
        return cmd_parts

    eval_type = None
    for i in range(len(cmd_parts) - 1):
        if cmd_parts[i] != eval_sys_cfg.eval_type_lookup_k:
            continue
        # Specified as the argument value.
        eval_type = cmd_parts[i + 1]

    if eval_type is None:
        raise ValueError(
            f"Could not find eval type lookup key {eval_sys_cfg.eval_type_lookup_k}"
        )
    eval_types = eval_sys_cfg.eval_types[args.eval]
    add_args = eval_types.get(eval_type, {})
    logger.info(f"Adding eval arguments for {eval_type}: {add_args}")
    return change_arg_vals(
        cmd_parts,
        add_args,
    )


def run(
    modify_run_cmd_fn: Optional[
        Callable[[List[str], str, argparse.Namespace], List[str]]
    ] = None,
    add_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None, type=str)
    parser.add_argument("--proj-dat", default=None, type=str)
    parser.add_argument(
        "--idx",
        default=None,
        type=int,
        help="If not specified, will evaluate the last checkpoint in the folder. If specified, this will evaluate the desired checkpoint index.",
    )
    parser.add_argument("--cmd", default=None, type=str)
    parser.add_argument(
        "--eval",
        default=None,
        type=str,
        help="The evaluation mode. Useful to have different evaluation setups for train and test splits. If specified, this will add additional arguments based on the value of `eval_cfg.eval_type_lookup_k` and the specified arguments in `eval_cfg.eval_types`.",
    )
    parser.add_argument("--cd", default=None, type=str)
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="If specified, will evaluate all checkpoints in folders.",
    )
    parser.add_argument(
        "--force-search",
        action="store_true",
        help="If specified, force search for the most recent checkpoint using the config `search_cmd`",
    )
    if add_args_fn is not None:
        add_args_fn(parser)
    args, rest = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    eval_sys_cfg = cfg.eval_sys
    runs = args.runs.split(",")
    for run_id in runs:
        rnd_id = str(uuid.uuid4())[:3]
        new_run_id = f"{run_id}_eval_{rnd_id}"

        for full_path in get_ckpt_full_path(cfg, eval_sys_cfg, args, run_id):
            eval_ckpt(
                run_id,
                full_path,
                new_run_id,
                cfg,
                args.proj_dat,
                modify_run_cmd_fn,
                args,
                rest,
            )
            if args.debug:
                break


if __name__ == "__main__":
    run()
