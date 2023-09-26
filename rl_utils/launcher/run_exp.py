import argparse
import os
import os.path as osp
import random
import string
import uuid

import yaml

try:
    import libtmux
except ImportError:
    libtmux = None
import shlex

from omegaconf import OmegaConf

from rl_utils.plotting.wb_query import query_s


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sess-id",
        type=int,
        default=-1,
        help="tmux session id to connect to. If unspec will run in current window",
    )
    parser.add_argument(
        "--sess-name",
        default=None,
        type=str,
        help="tmux session name to connect to",
    )
    parser.add_argument("--proj-dat", type=str, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--conda-env", type=str, default=None)
    parser.add_argument(
        "--time-freq",
        type=int,
        default=None,
        help="Sampling frequency for pyspy. If set, this will enable PySpy logging.",
    )
    parser.add_argument("--runs-dir", type=str, default="data/log/runs")
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        help="If not assigned then a randomly assigned one is generated.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--run-single",
        action="store_true",
        help="""
            If true, will run all commands in a single pane sequentially. This
            will chain together multiple runs in a cmd file rather than run
            them sequentially.
    """,
    )
    parser.add_argument(
        "--cd",
        default="-1",
        type=str,
        help="""
            String of CUDA_VISIBLE_DEVICES. A value of "-1" will not set
            CUDA_VISIBLE_DEVICES at all.
            """,
    )

    parser.add_argument(
        "--skip-add-all",
        action="store_true",
    )

    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--base-data-dir", type=str, default=None)

    # MULTIPROC OPTIONS
    parser.add_argument("--pt-proc", type=int, default=-1)
    parser.add_argument("--rdzv-endpoint", type=str, default=None)

    # YAML LAUNCH OPTIONS
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument(
        "--setup-command",
        type=str,
        default=None,
        help="What to substitute for `setup_command`.",
    )
    parser.add_argument("--secrets", type=str, default="")

    # SLURM OPTIONS
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument(
        "--slurm",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--slurm-no-batch",
        action="store_true",
        help="""
            If specified, will run with srun instead of sbatch
        """,
    )
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="""
            If true, will not export any environment variables from config.yaml
            """,
    )
    parser.add_argument(
        "--speed",
        action="store_true",
        help="""
            SLURM optimized for maximum CPU usage.
            """,
    )
    parser.add_argument(
        "--partition", type=str, default=None, help="Slum parition type."
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="""
            Slurm time limit. "10:00" is 10 minutes.
            """,
    )
    parser.add_argument(
        "--c",
        type=str,
        default="7",
        help="""
            Number of cpus for SLURM job
            """,
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="""
            Number of nodes for the job. Currently only template runs support
            this option.
            """,
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cpu-mem",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--g",
        type=str,
        default="1",
        help="""
            Number of gpus for SLURM job
            """,
    )
    parser.add_argument(
        "--ntasks",
        type=str,
        default=None,
        help="""
            Number of processes for SLURM job
            """,
    )

    return parser


def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if " " in x else x for x in spec_args]
    return " ".join(spec_args)


def get_cmds(rest):
    cmd = rest[0]
    if len(rest) > 1:
        rest = " ".join(rest[1:])
    else:
        rest = ""

    if ".cmd" in cmd:
        with open(cmd) as f:
            cmds = f.readlines()
    else:
        cmds = [cmd]

    cmds = list(filter(lambda x: not (x.startswith("#") or x == "\n"), cmds))
    return [f"{cmd.rstrip()} {rest}" for cmd in cmds]


def get_tmux_window(sess_name, sess_id):
    if libtmux is None:
        raise ValueError("Must install libtmux to use auto tmux capability")
    server = libtmux.Server()

    if sess_name is None:
        sess = server.get_by_id("$%i" % sess_id)
    else:
        sess = server.find_where({"session_name": sess_name})
    if sess is None:
        raise ValueError("invalid session id")

    return sess.new_window(attach=False, window_name="auto_proc")


def as_list(x, max_num):
    if isinstance(x, int):
        return [x for _ in range(max_num)]
    x = x.split("|")
    if len(x) == 1:
        return [x[0] for _ in range(max_num)]
    return x


def get_random_id() -> str:
    rnd_id = str(uuid.uuid4())[:8]
    return random.choice(string.ascii_uppercase) + rnd_id


def get_cmd_run_str(cmd, args, cmd_idx, num_cmds, proj_cfg, ident=None):
    conda_env = proj_cfg.get("conda_env", None)

    if args.conda_env is None and conda_env is not None:
        python_path = osp.join(
            osp.expanduser("~"), "miniconda3", "envs", conda_env, "bin"
        )
        python_path = proj_cfg.get("conda_path", python_path)
    else:
        python_path = args.conda_env

    g = as_list(args.g, num_cmds)
    c = as_list(args.c, num_cmds)
    if args.ntasks is None:
        ntasks = g[:]
    else:
        ntasks = as_list(args.ntasks, num_cmds)
    if ident is None:
        ident = get_random_id()
        if args.name_prefix is not None:
            ident = args.name_prefix + "_" + ident
    log_file = osp.join(args.runs_dir, ident) + ".log"
    cmd = cmd.replace("$SLURM_ID", ident)

    if args.partition is None:
        env_vars = " ".join(proj_cfg["add_env_vars"])
        return f"{env_vars} {cmd}"
    else:
        if not args.slurm_no_batch:
            run_file, run_name = generate_slurm_batch_file(
                log_file,
                ident,
                python_path,
                cmd,
                args.partition,
                ntasks[cmd_idx],
                g[cmd_idx],
                c[cmd_idx],
                args,
                proj_cfg,
            )
            return f"sbatch {run_file}"
        else:
            srun_settings = (
                f"--gres=gpu:{args.g} "
                + f"-p {args.partition} "
                + f"-c {args.c} "
                + f"-J {ident} "
                + f"-o {log_file}"
            )

            # This assumes the command begins with "python ..."
            return f"srun {srun_settings} {python_path}/{cmd}"


def sub_wb_query(cmd, proj_cfg):
    parts = cmd.split("&")
    if len(parts) < 3:
        return [cmd]

    new_cmd = [parts[0]]
    parts = parts[1:]

    for i in range(len(parts)):
        if i % 2 == 0:
            wb_query = parts[i]
            result = query_s(wb_query, proj_cfg, verbose=False)
            if len(result) == 0:
                raise ValueError(f"Got no response from {wb_query}")
            sub_vals = []
            for match in result:
                del match["rank"]
                if len(match) > 1:
                    raise ValueError(f"Only single value query supported, got {match}")
                sub_val = list(match.values())[0]
                sub_vals.append(sub_val)

            new_cmd = [c + sub_val for c in new_cmd for sub_val in sub_vals]
        else:
            for j in range(len(new_cmd)):
                new_cmd[j] += parts[i]

    return new_cmd


def log(s, args):
    print(s)


def split_cmd(cmd):
    cmd_parts = cmd.split(" ")
    ret_cmds = [[]]
    for cmd_part in cmd_parts:
        prefix = ""
        if "=" in cmd_part:
            prefix, cmd_part = cmd_part.split("=")
            prefix += "="

        if "," in cmd_part:
            ret_cmds = [
                ret_cmd + [prefix + split_part]
                for ret_cmd in ret_cmds
                for split_part in cmd_part.split(",")
            ]
        else:
            ret_cmds = [ret_cmd + [prefix + cmd_part] for ret_cmd in ret_cmds]
    return [" ".join(ret_cmd) for ret_cmd in ret_cmds]


def sub_in_args(old_cmd: str, new_args: str):
    old_parts = shlex.split(old_cmd, posix=False)
    new_parts = shlex.split(new_args, posix=False)

    i = 0
    while i < len(new_parts):
        if new_parts[i] in old_parts:
            old_i = old_parts.index(new_parts[i])
            old_parts[old_i + 1] = new_parts[i + 1]
        else:
            old_parts.extend(new_parts[i : i + 2])
        i += 2
    return " ".join(old_parts)


def sub_in_vars(cmd, proj_cfg, rank_i, group_id, override_base_data_dir=None):
    return (
        cmd.replace("$GROUP_ID", group_id)
        .replace(
            "$DATA_DIR",
            proj_cfg["base_data_dir"]
            if override_base_data_dir is None
            else override_base_data_dir,
        )
        .replace("$CMD_RANK", str(rank_i))
        .replace("$PROJECT_NAME", proj_cfg.get("proj_name", ""))
        .replace("$WB_ENTITY", proj_cfg.get("wb_entity", ""))
    )


def yamlize_cmd(cmd, template_cfg, args, ident):
    for key, val in template_cfg["sub_paths"].items():
        cmd = cmd.replace(key, val)
    template_path = template_cfg.templates[args.template]
    with open(template_path, "r") as f:
        base_template = yaml.safe_load(f)
    path = osp.join(args.runs_dir, get_random_id() + ".yaml")
    secrets = args.secrets.split(",")
    for secret in secrets:
        if secret == "":
            continue
        k, v = secret.split("=")
        base_template["environment_variables"][k] = v
    base_template["name"] = ident
    if "prefix_cmd" in template_cfg:
        cmd = template_cfg.prefix_cmd + " " + cmd
    base_template["command"] = cmd
    num_gpus = int(args.g)
    gpu_ratio = template_cfg["max_num_gpus"] / num_gpus
    base_template["resources"]["num_gpus"] = num_gpus
    # Scale other resource use by the number of gpus.
    base_template["resources"]["num_cpus"] = int(
        base_template["resources"]["num_cpus"] // gpu_ratio
    )
    base_template["resources"]["memory_gb"] = int(
        base_template["resources"]["memory_gb"] // gpu_ratio
    )
    base_template["resources"]["num_nodes"] = args.num_nodes
    if args.setup_command is not None:
        base_template["setup_command"] = args.setup_command

    with open(path, "w") as f:
        yaml.dump(
            base_template,
            f,
            indent=4,
            sort_keys=False,
        )

    return path


def execute_command_file(run_cmd, args, proj_cfg):
    if args.c is None:
        args.c = "7"

    if not osp.exists(args.runs_dir):
        os.makedirs(args.runs_dir)

    cmds = get_cmds(run_cmd)

    # Sub in W&B args
    cmds = [c for cmd in cmds for c in sub_wb_query(cmd, proj_cfg)]

    n_cmds = len(cmds)

    add_all = proj_cfg.get("add_all", None)
    if add_all is not None and not args.skip_add_all:
        cmds = [sub_in_args(cmd, add_all) for cmd in cmds]

    if args.time_freq is None:
        pyspy_s = ""
    else:
        pyspy_s = f"py-spy record --idle --function --native --subprocesses --rate {args.time_freq} --output data/profile/scope.speedscope --format speedscope -- "

    cmds = [f"{pyspy_s}{x}" for x in cmds]

    # Add on the project data
    if args.proj_dat is not None:
        proj_data = proj_cfg.get("proj_data", {})

        def sub_in_proj_dat(cmd, k):
            if k not in proj_data:
                raise ValueError(
                    f"Could not find {k} in proj-data which contains {list(proj_data.keys())}"
                )
            return sub_in_args(cmd, proj_data[k])

        for k in args.proj_dat.split(","):
            cmds = [sub_in_proj_dat(cmd, k) for cmd in cmds]
            env_var_dat = proj_cfg.get("proj_dat_add_env_vars", {}).get(k, None)
            if env_var_dat is not None:
                cmds = [env_var_dat + " " + cmd for cmd in cmds]

    if args.group_id is None:
        group_ident = get_random_id()
    else:
        group_ident = args.group_id
    print(f"Assigning group ID {group_ident}")
    cmds = [
        sub_in_vars(cmd, proj_cfg, rank_i, group_ident, args.base_data_dir)
        for rank_i, cmd in enumerate(cmds)
    ]

    if args.pt_proc != -1:
        pt_dist_str = f"torchrun --nproc_per_node {args.pt_proc}"
        if args.rdzv_endpoint is not None:
            pt_dist_str += f" --rdzv-endpoint={args.rdzv_endpoint}"

        def make_dist_cmd(x):
            return x.replace("python", pt_dist_str)

        cmds[0] = make_dist_cmd(cmds[0])

    DELIM = " ; "

    cd = as_list(args.cd, n_cmds)

    def launch(cmd):
        if not args.check:
            os.system(cmd)

    if args.template is not None:
        template_cfg = proj_cfg["yaml_launch"]
        template_cmd = template_cfg["cmd"]
        for cmd_idx, cmd in enumerate(cmds):
            ident = get_random_id()
            if args.name_prefix is not None:
                ident = args.name_prefix + "_" + ident
            run_cmd = get_cmd_run_str(cmd, args, cmd_idx, n_cmds, proj_cfg, ident)
            run_cmd = template_cmd % yamlize_cmd(run_cmd, template_cfg, args, ident)
            log(f"Running {run_cmd}", args)
            launch(run_cmd)

    elif args.sess_id == -1 and args.sess_name is None:
        if args.partition is not None:
            for cmd_idx, cmd in enumerate(cmds):
                run_cmd = get_cmd_run_str(cmd, args, cmd_idx, n_cmds, proj_cfg)
                log(f"Running {run_cmd}", args)
                launch(run_cmd)
        elif args.run_single:
            cmds = [get_cmd_run_str(x, args, 0, 1, proj_cfg) for x in cmds]
            exec_cmd = DELIM.join(cmds)

            log(f"Running {exec_cmd}", args)
            launch(exec_cmd)

        elif n_cmds == 1:
            exec_cmd = get_cmd_run_str(cmds[0], args, 0, n_cmds, proj_cfg)
            if cd[0] != "-1":
                exec_cmd = "CUDA_VISIBLE_DEVICES=" + cd[0] + " " + exec_cmd
            log(f"Running {exec_cmd}", args)
            launch(exec_cmd)
        else:
            raise ValueError(
                f"Running multiple jobs. You must specify tmux session id. Tried to run {cmds}"
            )
    else:
        if args.run_single:
            cmds = DELIM.join(cmds)
            cmds = [cmds]

        for cmd_idx, cmd in enumerate(cmds):
            new_window = get_tmux_window(args.sess_name, args.sess_id)

            log("running full command %s\n" % cmd, args)

            run_cmd = get_cmd_run_str(cmd, args, cmd_idx, n_cmds, proj_cfg)

            # Send the keys to run the command
            if args.partition is None:
                last_pane = new_window.attached_pane
                last_pane.send_keys(run_cmd, enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=50)
                pane.send_keys("source deactivate")

                if "conda_env" in proj_cfg:
                    pane.send_keys("source activate " + proj_cfg["conda_env"])
                pane.enter()
                if cd[cmd_idx] != "-1":
                    pane.send_keys("export CUDA_VISIBLE_DEVICES=" + cd[cmd_idx])
                    pane.enter()
                else:
                    pane.send_keys(run_cmd)

                pane.enter()
            else:
                pane = new_window.split_window(attach=False)
                pane.set_height(height=10)
                pane.send_keys(run_cmd)

        log("everything should be running...", args)


def generate_slurm_batch_file(
    log_file, ident, python_path, cmd, partition, ntasks, g, c, args, proj_cfg
):
    ignore_nodes_s = ",".join(proj_cfg.get("slurm_ignore_nodes", []))
    if len(ignore_nodes_s) != 0:
        ignore_nodes_s = "#SBATCH -x " + ignore_nodes_s

    add_options = [ignore_nodes_s]
    if args.time is not None:
        add_options.append(f"#SBATCH --time={args.time}")
    if args.comment is not None:
        add_options.append(f'#SBATCH --comment="{args.comment}"')
    if args.constraint is not None:
        add_options.append(f"#SBATCH --constraint={args.constraint}")
    if args.cpu_mem is not None:
        add_options.append(f"#SBATCH --mem-per-cpu={args.cpu_mem}")
    add_options = "\n".join(add_options)

    python_parts = cmd.split("python")
    has_python = False
    if len(python_parts) > 1:
        cmd = "python" + python_parts[1]
        has_python = True

    if not args.skip_env:
        env_vars = proj_cfg.get("add_env_vars", [])
        env_vars = [f"export {x}" for x in env_vars]

        if args.proj_dat is not None:
            for k in args.proj_dat.split(","):
                env_var_dat = proj_cfg.get("proj_dat_add_env_vars", {}).get(k, None)
                if env_var_dat is not None:
                    proj_env_vars = env_var_dat.split(" ")
                    for proj_env_var in proj_env_vars:
                        env_vars.append(f"export {proj_env_var}")

        env_vars = "\n".join(env_vars)

    cpu_options = "#SBATCH --cpus-per-task %i" % int(c)
    if args.speed:
        cpu_options += "#SBATCH --overcommit\n"
        cpu_options += "#SBATCH --cpu-freq=performance\n"

    if has_python:
        run_cmd = python_path + "/" + cmd
        requeue_s = "#SBATCH --requeue"
    else:
        run_cmd = cmd
        requeue_s = ""

    fcontents = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --gres gpu:%i
%s
#SBATCH --nodes 1
#SBATCH --signal=USR1@600
#SBATCH --ntasks-per-node %i
%s
#SBATCH -p %s
%s

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR
%s

set -x
srun %s"""
    job_name = ident
    log_file_loc = "/".join(log_file.split("/")[:-1])
    fcontents = fcontents % (
        job_name,
        log_file,
        int(g),
        cpu_options,
        int(ntasks),
        requeue_s,
        partition,
        add_options,
        env_vars,
        run_cmd,
    )
    job_file = osp.join(log_file_loc, job_name + ".sh")
    with open(job_file, "w") as f:
        f.write(fcontents)
    return job_file, job_name


def full_execute_command_file():
    parser = get_arg_parser()
    args, rest = parser.parse_known_args()
    if args.cfg is None:
        proj_cfg = {}
    else:
        proj_cfg = OmegaConf.load(args.cfg)
    slurm_cfg = proj_cfg.get("slurm", {})
    if args.slurm is not None:
        def_slurm = slurm_cfg[args.slurm]
        if "c" in def_slurm:
            args.c = def_slurm["c"]

        if args.time is None and "time" in def_slurm:
            args.time = def_slurm["time"]

        if args.partition is None and "partition" in def_slurm:
            args.partition = def_slurm["partition"]

        if args.constraint is None and "constraint" in def_slurm:
            args.constraint = def_slurm["constraint"]

        if args.cpu_mem is None and "cpu_mem" in def_slurm:
            args.cpu_mem = def_slurm["cpu_mem"]

    execute_command_file(rest, args, proj_cfg)
