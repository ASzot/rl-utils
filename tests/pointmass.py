import torch
from rl_helper.envs import create_vectorized_envs


def test_create():
    envs = create_vectorized_envs(
        "PointMass-v0",
        32,
    )
    envs.reset()
    for _ in range(100):
        rnd_ac = torch.tensor(envs.action_space.sample())
        rnd_ac = rnd_ac.view(1, -1).repeat(32, 1)
        envs.step(rnd_ac)
