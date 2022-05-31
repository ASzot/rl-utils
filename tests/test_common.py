import torch

from rl_utils.common import DictDataset, Evaluator, extract_next_tensor
from rl_utils.envs import create_vectorized_envs
from rl_utils.interfaces import RandomPolicy

TRAJ_SAVE_PATH = "data/trajs/traj.pt"


def test_dataset_load_and_manipulation():
    envs = create_vectorized_envs(
        "PointMass-v0",
        32,
    )
    evaluator = Evaluator(
        envs, 0, 0, "data/vids/", fps=10, save_traj_name=TRAJ_SAVE_PATH
    )
    policy = RandomPolicy(envs.action_space)
    evaluator.evaluate(policy, 10, 0)

    dataset = torch.load(TRAJ_SAVE_PATH)
    dataset = extract_next_tensor(dataset)

    dataset = DictDataset(
        dataset,
        ["obs", "actions", "rewards", "masks", "next_obs"],
    )
