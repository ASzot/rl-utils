import argparse
import os
import os.path as osp
import uuid
from typing import Any, Callable, Dict, List, Optional

from omegaconf import OmegaConf

from rl_utils.launcher.run_exp import get_random_id, sub_in_args, sub_in_vars

RUN_DIR = "data/log/runs/"


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


def split_cmd_txt(cmd):
    return [y for x in cmd.split(" ") for y in x.split("=")]


def eval_ckpt(
    run_id,
    model_ckpt_path,
    new_run_id,
    cfg,
    proj_dat,
    modify_run_cmd_fn,
    args: argparse.Namespace,
):
    eval_sys_cfg = cfg.eval_sys
    # Find the run command.
    run_path = osp.join(RUN_DIR, run_id + ".sh")
    if osp.exists(run_path):
        print("Substituting slurm run command.")
        # Get the actually executed command.
        with open(run_path, "r") as f:
            cmd = f.readlines()[-1]
        cmd_parts = split_cmd_txt(cmd)
    else:
        print("Substituting in the run command.")
        if args.cmd is None:
            cmd = eval_sys_cfg.eval_run_cmd
        else:
            cmd = eval_sys_cfg.eval_run_cmd[args.cmd]
        add_all = cfg.get("add_all", None)
        if add_all is not None:
            cmd = sub_in_args(cmd, add_all)
        cmd = sub_in_vars(cmd, cfg, 0, "eval")
        ident = get_random_id()
        cmd = cmd.replace("$SLURM_ID", ident)

        cmd_parts = split_cmd_txt(cmd)
        # Dummy line for srun
        cmd_parts.insert(0, "")

    def add_eval_suffix(x):
        x = x.strip()
        if x == "":
            return get_random_id() + "_eval"
        elif "." in x:
            parts = x.split(".")
            return parts[0] + "_eval." + parts[1]
        elif x[-1] == "/":
            return x[:-1] + "_eval/"
        else:
            return x + "_eval"

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
    new_cmd = sub_in_vars(new_cmd, cfg, 0, "eval")

    print("EVALUATING ", new_cmd)
    os.system(new_cmd)
    return True


def run(
    modify_run_cmd_fn: Optional[
        Callable[[List[str], str, argparse.Namespace], List[str]]
    ] = None,
    add_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=None, type=str)
    parser.add_argument("--proj-dat", default=None, type=str)
    parser.add_argument("--idx", default=None, type=int)
    parser.add_argument("--cmd", default=None, type=str)
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--debug", action="store_true")
    if add_args_fn is not None:
        add_args_fn(parser)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    eval_sys_cfg = cfg.eval_sys
    runs = args.runs.split(",")
    for run_id in runs:
        full_path = osp.join(cfg.base_data_dir, eval_sys_cfg.ckpt_search_dir, run_id)

        ckpt_idxs = [
            int(f.split(".")[1])
            for f in os.listdir(full_path)
            if ".pth" in f and "ckpt" in f
        ]
        if args.idx is None:
            last_idx = max(ckpt_idxs)
        else:
            last_idx = args.idx

        full_path = osp.join(full_path, f"ckpt.{last_idx}.pth")

        rnd_id = str(uuid.uuid4())[:3]
        new_run_id = f"{run_id}_eval_{rnd_id}"

        eval_ckpt(
            run_id,
            full_path,
            new_run_id,
            cfg,
            args.proj_dat,
            modify_run_cmd_fn,
            args,
        )
        if args.debug:
            break


if __name__ == "__main__":
    run()
