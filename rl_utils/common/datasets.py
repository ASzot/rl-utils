from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(
        self,
        load_data: Dict[str, torch.Tensor],
        load_keys: Optional[List[str]] = None,
        detach_all: bool = True,
    ):
        """
        :parameters load_keys: Subset of keys that are loaded from `load_data`.
        """
        if load_keys is None:
            load_keys = load_data.keys()

        self._load_data = {
            k: v.detach() if detach_all else v
            for k, v in load_data.items()
            if k in load_keys
        }
        tensor_sizes = [tensor.size(0) for tensor in self._load_data.values()]
        if len(set(tensor_sizes)) != 1:
            raise ValueError("Tensors to dataset are not of the same shape")
        self._dataset_len = tensor_sizes[0]

    @property
    def all_data(self):
        return self._load_data

    def get_data(self, k: str) -> torch.Tensor:
        return self._load_data[k]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self._load_data.items()}

    def __len__(self) -> int:
        return self._dataset_len


def extract_next_tensor(dataset: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    obs = dataset["observations"].detach()

    final_final_obs = dataset["infos"][-1]["final_obs"]

    next_obs = torch.cat([obs[1:], final_final_obs.unsqueeze(0)], 0)
    num_eps = 1
    for i in range(obs.shape[0] - 1):
        cur_info = dataset["infos"][i]
        if "final_obs" in cur_info:
            num_eps += 1
            next_obs[i] = cur_info["final_obs"].detach()
    masks = ~(dataset["terminals"].bool())

    num_terminals = masks.size(0) - masks.sum()
    if num_eps != num_terminals.sum():
        raise ValueError(
            f"Inconsistency in # of episodes {num_eps} vs {dataset['terminals'].sum()}"
        )
    dataset["next_obs"] = next_obs.detach()

    return dataset


def extract_final_obs(
    obs: torch.Tensor, masks: torch.Tensor, final_obs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param obs: Shape (N, ...)
    :param masks: Shape (N, ...)
    :param final_obs: Shape (N-1, ...)

    :returns: obs, next_obs, masks all of shape (N-1, ...)
    """
    cur_obs = obs[:-1]
    masks = masks[1:]
    next_obs = (masks * obs[1:]) + ((1 - masks) * final_obs)
    return cur_obs, next_obs, masks
