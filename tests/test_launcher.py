import os.path as osp
import shlex

import pytest
from omegaconf import OmegaConf

from rl_utils.launcher.run_exp import execute_command_file, get_arg_parser


def _get_cmd_prefix():
    cfg_path = osp.dirname(osp.realpath(__file__))
    return f"--cfg {cfg_path}/test_cfg.yaml --check --name-prefix test --proj-dat test_proj_dat"


def _run(cmd: str):
    parser = get_arg_parser()
    args, rest = parser.parse_known_args(shlex.split(cmd))
    if args.cfg is None:
        proj_cfg = {}
    else:
        proj_cfg = OmegaConf.load(args.cfg)
    execute_command_file(rest, args, proj_cfg)


def test_reg_run():
    _run(f"{_get_cmd_prefix()} python test.py")


@pytest.mark.parametrize(
    "add_opts",
    (
        "--c 16",
        "--g 2",
    ),
)
def test_slurm_run(add_opts):
    _run(f"{_get_cmd_prefix()} {add_opts} --partition test_partition python test.py")
