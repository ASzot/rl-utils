import torch


def cumsum_discount(x: torch.Tensor, discount: float) -> torch.Tensor:
    """
    :param x: Tensor of shape (batch_size, ..., 1)
    :returns: Tensor of shape same size as `x`.
    """

    r = torch.zeros_like(x)
    r[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        r[t] = x[t] + discount * r[t + 1]
    return r
