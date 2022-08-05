import torch


def tensor_hash(X: torch.Tensor) -> int:
    """
    Returns a unique representation of a tensor. Warning, this will be slow for
    large tensors.
    """
    flat_X = X.detach().view(-1).cpu().tolist()
    return hash(tuple(flat_X))
