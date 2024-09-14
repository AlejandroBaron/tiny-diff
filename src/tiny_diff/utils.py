from typing import Callable

import torch


def loss_backward(loss: Callable, optimizer: torch.optim.Optimizer):
    """Backward loss computation."""
    optimizer.zero_grad()
    loss.backward()
    parameters = list(optimizer.param_groups[0]["params"])
    torch.nn.utils.clip_grad_norm_(parameters, 2)
    optimizer.step()


def match_shape(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Adds dims to x to match like."""

    num_dims_to_add = len(like.shape) - len(x.shape)
    for _ in range(num_dims_to_add):
        x = x.unsqueeze(-1)

    return x
