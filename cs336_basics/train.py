import math
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import os

from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, Callable, Iterable, Iterator, Optional



def wei_log_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max, max_i = torch.max(in_features, dim=dim, keepdim = True)
    exp = torch.exp(in_features - max)
    log_exp = in_features - max
    sum = torch.sum(exp, dim=dim, keepdim = True)
    # log_softmax = log(exp / sum) = log(exp) - log(sum)
    return log_exp - torch.log(sum)


def wei_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs_reshaped = rearrange(inputs, "b ... v -> (b ...) v")  # (batch_size, vocab_size)
    targets_reshaped = rearrange(targets, "b ... -> (b ...)")  # (batch_size,)
    log_softmax = wei_log_softmax(inputs_reshaped, dim=-1)
    pro = log_softmax.gather(1, targets_reshaped.unsqueeze(1)).squeeze(0)
    return torch.mean(-pro)



class WeiAdamWOptimizer(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
    if lr < 0:
     raise ValueError(f"Invalid learning rate: {lr}")
    if betas[0] < 0 or betas[0] >= 1:
     raise ValueError(f"Invalid learning parameter: {betas[0]}")
    if betas[1] < 0 or betas[1] >= 1:
     raise ValueError(f"Invalid learning parameter: {betas[1]}")
    defaults = {"alpha": lr, "beta1": betas[0], "beta2": betas[1], "theta": weight_decay, "eps": eps}
    super().__init__(params, defaults)

    for group in self.param_groups:
      for p in group["params"]:
        state = self.state[p]
        state["m"] = torch.zeros_like(p.data)
        state["v"] = torch.zeros_like(p.data)
        state["alpha_t"] = lr

  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      alpha = group["alpha"] # Get the learning rate.
      beta1 = group["beta1"]
      beta2 = group["beta2"]
      theta = group["theta"]
      eps = group["eps"]
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 1) # Get iteration number from the state, or initial value.
        state["m"] = state["m"] * beta1 + (1 - beta1) * p.grad.data
        state["v"] = state["v"] * beta2 + (1 - beta2) * p.grad.data**2
        state["alpha_t"] = alpha * (1 - beta2**t)**0.5 / (1 - beta1**t)
        
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= state["alpha_t"] * state["m"] / (state["v"]**0.5 + eps) # Update weight tensor in-place.
        p.data = p.data - alpha * theta * p.data# Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
    return loss


def copy_learning_rate_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * cos_percent)
        )
    else:
        return min_learning_rate


def copy_clip_grad(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0, eps: float = 1e-6):
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            total_norm += torch.sum(param.grad ** 2)
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def copy_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_len = dataset.shape[0]
    if dataset_len < context_length:
        raise ValueError(f"Dataset length {dataset_len} is less than context length {context_length}.")

    starts = np.random.randint(0, dataset_len - context_length, size=batch_size)
    inputs = np.stack([dataset[start:start + context_length] for start in starts], dtype=np.int64)
    targets = np.stack([dataset[start + 1:start + context_length + 1] for start in starts], dtype=np.int64)

    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device)
    )


def wei_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """

    checkpoint = {}
    checkpoint["iteration"] = iteration
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, out)


def wei_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
