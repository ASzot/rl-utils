import gym.spaces as spaces
import torch

import rl_utils.common as cutils

from .vec_env import VecEnvObservationWrapper, VecEnvWrapper


class VecEnvClipActions(VecEnvWrapper):
    def __init__(self, venv):
        obs_space = venv.action_space
        self._ac_low = torch.from_numpy(obs_space.low)
        self._ac_high = torch.from_numpy(obs_space.high)
        VecEnvWrapper.__init__(self, venv)

    def step(self, actions):
        actions = torch.clip(actions, self._ac_low, self._ac_high)
        return super().step(actions)


class VecEnvPermuteFrames(VecEnvObservationWrapper):
    def __init__(self, venv):
        obs_space = venv.observation_space
        if not isinstance(obs_space, spaces.Dict):
            obs_space = spaces.Dict({None: obs_space})

        permute_ks = {k: v for k, v in obs_space.spaces.items() if len(v.shape) == 3}
        for k, v in permute_ks.items():
            s = v.shape
            obs_space[k] = cutils.reshape_obs_space(v, (s[2], s[0], s[1]))

        VecEnvWrapper.__init__(self, venv, observation_space=obs_space)

    def process(self, obs):
        return obs.permute(0, 3, 1, 2)
