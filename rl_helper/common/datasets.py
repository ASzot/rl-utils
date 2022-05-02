from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(
        self, load_data: Dict[str, torch.Tensor], load_keys: Optional[List[str]] = None
    ):
        if load_keys is None:
            load_keys = load_data.keys()

        self._load_data = {k: v for k, v in load_data.items() if k in load_keys}
        tensor_sizes = [tensor.size(0) for tensor in self._load_data.values()]
        if len(set(tensor_sizes)) != 1:
            raise ValueError("Tensors to dataset are not of the same shape")
        self._dataset_len = tensor_sizes[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._load_data.items()}

    def __len__(self):
        return self._dataset_len
