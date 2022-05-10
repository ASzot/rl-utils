import argparse
import datetime
import os
import os.path as osp
import random
import re
import string
import subprocess
import sys
import time
import uuid

import libtmux
import numpy as np
from rlf.args import str2bool
from rlf.exp_mgr.wb_data_mgr import get_run_params
from rlf.exp_mgr.wb_query import query_s

RUNS_DIR = "data/log/runs"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sess-id",
        type=int,
        default=-1,
        help="tmux session id to connect to. If unspec will run in current window",
    )
    parser.add_argument(
        "--sess-name", default=None, type=str, help="tmux session name to connect to"
    )
    parser.add_argument(
        "--cmd", type=str, required=True, help="list of commands to run"
    )
    parser.add_argument("--proj-dat", type=str, default=None)
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
    parser.add_argument("--cfg", type=str, default="./config.yaml")

    # MULTIPROC OPTIONS
    parser.add_argument("--mp-offset", type=int, default=0)
    parser.add_argument("--pt-proc", type=int, default=-1)

    # SLURM OPTIONS
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--inject-slurm-id", type=str2bool, default=True)
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
        "--st", type=str, default=None, help="Slum parition type [long, short]"
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
        default="1",
        help="""
            Number of processes for SLURM job
            """,
    )

    return parser


def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if " " in x else x for x in spec_args]
    return " ".join(spec_args)


def get_cmds(cmd_path, spec_args, args):
    if ".cmd" in cmd_path:
        open_cmd = osp.join(cmd_path + ".cmd")
        with open(open_cmd) as f:
            cmds = f.readlines()
    else:
        cmds = [cmd_path]

    cmds = list(filter(lambda x: not (x.startswith("#") or x == "\n"), cmds))
    cmds = [cmd.rstrip() + " " for cmd in cmds]

    # Check if any commands are references to other commands
    all_ref_cmds = []
    for i, cmd in enumerate(cmds):
        if cmd.startswith("R:"):
            cmd_parts = cmd.split(":")[1].split(" ")
            ref_cmd_loc = cmd_parts[0]
            full_ref_cmd_loc = osp.join(config_mgr.get_prop("cmds_loc"), ref_cmd_loc)
            ref_cmds = get_cmds(
                full_ref_cmd_loc.rstrip(), [*cmd_parts[1:], *spec_args], args
            )
            all_ref_cmds.extend(ref_cmds)
    cmds = list(filter(lambda x: not x.startswith("R:"), cmds))

    cmds = [cmd + add_on_args(spec_args) for cmd in cmds]

    cmds.extend(all_ref_cmds)
    return cmds


def get_tmux_window(sess_name, sess_id):
    server = libtmux.Server()

    if sess_name is None:
        tmp = server.list_sessions()
        sess = server.get_by_id("$%i" % sess_id)
    else:
        sess = server.find_where({"session_name": sess_name})
    if sess is None:
        raise ValueError("invalid session id")

    return sess.new_window(attach=False, window_name="auto_proc")


def transform_k(s, use_split, replace_s):
    prefix_parts = s.split(use_split)

    before_prefix = use_split.join(prefix_parts[:-1])
    prefix = prefix_parts[-1]
    parts = prefix.split(" ")
    prefix = parts[0]
    after_prefix = " ".join(parts[1:])

    return before_prefix + f"{use_split} {replace_s} " + after_prefix


def as_list(x, max_num):
    if isinstance(x, int):
        return [x for _ in range(max_num)]
    x = x.split("|")
    if len(x) == 1:
        x = [x[0] for _ in range(max_num)]
    return x


def get_cmd_run_str(cmd, args, cd, cmd_idx, num_cmds, slurm_add):
    env_vars = " ".join(config_mgr.get_prop("add_env_vars", []))
    if len(env_vars) != 0:
        env_vars += " "
    conda_env = config_mgr.get_prop("conda_env")
    python_path = osp.join(osp.expanduser("~"), "miniconda3", "envs", conda_env, "bin")
    python_path = config_mgr.get_prop("conda_path", def_val=python_path)

    ntasks = as_list(args.ntasks, num_cmds)
    g = as_list(args.g, num_cmds)
    c = as_list(args.c, num_cmds)

    if args.st is None:
        return env_vars + cmd
    else:
        # Make command into a SLURM command
        ident = str(uuid.uuid4())[:8]
        if slurm_add[cmd_idx] is not None:
            ident = slurm_add[cmd_idx] + ident
        log_file = osp.join(RUNS_DIR, ident) + ".log"

        if not args.slurm_no_batch:
            run_file, run_name = generate_slurm_batch_file(
                log_file,
                ident,
                python_path,
                cmd,
                args.st,
                ntasks[cmd_idx],
                g[cmd_idx],
                c[cmd_idx],
                args,
            )
            return f"sbatch {run_file}"
        else:
            srun_settings = (
                f"--gres=gpu:{args.g} "
                + f"-p {args.st} "
                + f"-c {args.c} "
                + f"-J {ident} "
                + f"-o {log_file}"
            )

            # This assumes the command begins with "python ..."
            return f"srun {srun_settings} {python_path}/{cmd}"


def sub_wb_query(cmd, args):
    parts = cmd.split("&")
    if len(parts) < 3:
        return [cmd]

    new_cmd = [parts[0]]
    parts = parts[1:]

    mod_seeds = None

    for i in range(len(parts)):
        if i % 2 == 0:
            wb_query = parts[i]
            result = query_s(wb_query, verbose=False)
            if len(result) > 0 and "last_model" in result[0]:
                mod_seeds = [
                    " --seed " + x["last_model"].split("/")[-2].split("-")[2]
                    for x in result
                ]
            if len(result) == 0:
                raise ValueError(f"Got no response from {wb_query}")
            sub_vals = []
            for match in result:
                if len(match) > 1:
                    raise ValueError(f"Only single value query supported, got {match}")
                sub_val = list(match.values())[0]
                sub_vals.append(sub_val)

            new_cmd = [c + sub_val for c in new_cmd for sub_val in sub_vals]
        else:
            for j in range(len(new_cmd)):
                new_cmd[j] += parts[i]
    if mod_seeds is not None:
        new_cmd = [c + seed for c, seed in zip(new_cmd, mod_seeds)]

    return new_cmd


def log(s, args):
    print(s)


def execute_command_file(cmd_path, add_args_str, cd, sess_name, sess_id, seed, args):
    if not osp.exists(RUNS_DIR):
        os.makedirs(RUNS_DIR)

    cmds = get_cmds(cmd_path, add_args_str, args)

    cmds = [c for cmd in cmds for c in sub_wb_query(cmd, args)]

    if args.proj_dat is not None:
        proj_data = config_mgr.get_prop("proj_data", {})
        lookups = args.proj_dat.split(",")
        for k in lookups:
            add_args = proj_data[k]
            cmds = [cmd + " " + add_args for cmd in cmds]

    if args.pt_proc != -1:
        pt_dist_str = f"MULTI_PROC_OFFSET={args.mp_offset} python -u -m torch.distributed.launch --use_env --nproc_per_node {args.pt_proc} "

        def make_dist_cmd(x):
            parts = x.split(" ")
            runf = None
            for i, part in enumerate(parts):
                if ".py" in part:
                    runf = i
                    break

            if runf is None:
                raise ValueError("Could not split command")

            rest = " ".join(parts[runf:])
            return pt_dist_str + rest

        cmds[0] = make_dist_cmd(cmds[0])

    slurm_add = [None] * len(cmds)

    DELIM = " ; "

    cd = as_list(cd, len(cmds))

    if sess_id == -1 and sess_name is None:
        if args.st is not None:
            for cmd_idx, cmd in enumerate(cmds):
                run_cmd = get_cmd_run_str(cmd, args, cd, cmd_idx, len(cmds), slurm_add)
                log(f"Running {run_cmd}", args)
                os.system(run_cmd)
        elif args.run_single:
            cmds = [get_cmd_run_str(x, args, cd, 0, 1, slurm_add) for x in cmds]
            exec_cmd = DELIM.join(cmds)

            log(f"Running {exec_cmd}", args)
            os.system(exec_cmd)

        elif len(cmds) == 1:
            exec_cmd = get_cmd_run_str(cmds[0], args, cd, 0, len(cmds), slurm_add)
            if cd[0] != "-1":
                exec_cmd = "CUDA_VISIBLE_DEVICES=" + cd[0] + " " + exec_cmd
            log(f"Running {exec_cmd}", args)
            os.system(exec_cmd)
        else:
            raise ValueError("Running multiple jobs. You must specify tmux session id")
    else:

        if args.run_single:
            cmds = DELIM.join(cmds)
            cmds = [cmds]

        for cmd_idx, cmd in enumerate(cmds):
            new_window = get_tmux_window(sess_name, sess_id)

            log("running full command %s\n" % cmd, args)

            run_cmd = get_cmd_run_str(cmd, args, cd, cmd_idx, len(cmds), slurm_add)

            # Send the keys to run the command
            conda_env = config_mgr.get_prop("conda_env")
            if args.st is None:
                last_pane = new_window.attached_pane
                last_pane.send_keys(run_cmd, enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=50)
                pane.send_keys("source deactivate")

                pane.send_keys("source activate " + conda_env)
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
    log_file, ident, python_path, cmd, st, ntasks, g, c, args
):
    ignore_nodes_s = ",".join(config_mgr.get_prop("slurm_ignore_nodes", []))
    if len(ignore_nodes_s) != 0:
        ignore_nodes_s = "#SBATCH -x " + ignore_nodes_s

    add_options = [ignore_nodes_s]
    if args.time is not None:
        add_options.append(f"#SBATCH --time={args.time}")
    if args.comment is not None:
        add_options.append(f'#SBATCH --comment="{args.comment}"')
    add_options = "\n".join(add_options)

    pre_python_txt = ""
    python_parts = cmd.split("python")
    has_python = False
    if len(python_parts) > 1:
        pre_python_txt = python_parts[0]
        cmd = "python" + python_parts[1]
        has_python = True

    cmd_line_exports = ""
    if not args.skip_env:
        env_vars = config_mgr.get_prop("add_env_vars", [])
        env_vars = [f"export {x}" for x in env_vars]
        env_vars = "\n".join(env_vars)

    cpu_options = "#SBATCH --cpus-per-task %i" % int(c)
    if args.speed:
        cpu_options = "#SBATCH --overcommit\n"
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

export MULTI_PROC_OFFSET=%i
%s

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

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
        st,
        add_options,
        args.mp_offset,
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
    config_mgr.init(args.cfg)

    if "rlf." in args.cmd:
        args.cmd = args.cmd[len("rlf.") :].replace(".", "/")
        rlf_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
        config_mgr.set_prop("cmds_loc", rlf_dir)

        cmd_path = osp.join(config_mgr.get_prop("cmds_loc"), args.cmd)
        print("Executing ", cmd_path)
    else:
        cmd_path = osp.join(config_mgr.get_prop("cmds_loc"), args.cmd)

    execute_command_file(
        cmd_path, rest, args.cd, args.sess_name, args.sess_id, args.seed, args
    )


def execute_from_string(arg_str, cmds_loc, executable_path):
    parser = get_arg_parser()
    args, rest = parser.parse_known_args(arg_str)

    config_mgr.init(args.cfg)

    if cmds_loc is not None:
        config_mgr.set_prop("cmds_loc", cmds_loc)
    if executable_path is not None:
        config_mgr.set_prop("file_path", executable_path)
    cmd_path = osp.join(config_mgr.get_prop("cmds_loc"), args.cmd)
    execute_command_file(
        cmd_path, rest, args.cd, args.sess_name, args.sess_id, args.seed, args
    )


if __name__ == "__main__":
    full_execute_command_file()
