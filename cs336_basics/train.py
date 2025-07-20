import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional


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
    log_softmax = wei_log_softmax(inputs, dim=-1)
    pro = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(0)
    return torch.mean(-pro)


class AdamWOptimizer(torch.optim.Optimizer):
  def __init__(self, params, alpha=1e-3, beta1=0, beta2=0, theta=0, eps=0.00000001):
    if alpha < 0:
     raise ValueError(f"Invalid learning rate: {alpha}")
    if beta1 < 0 or beta1 >= 1:
     raise ValueError(f"Invalid learning parameter: {beta1}")
    if beta2 < 0 or beta2 >= 1:
     raise ValueError(f"Invalid learning parameter: {beta2}")
    defaults = {"alpha": alpha, "beta1": beta1, "beta2": beta2, "theta": theta, "eps": eps}
    super().__init__(params, defaults)

    for group in self.param_groups:
      for p in group["params"]:
        state = self.state[p]
        state["m"] = torch.zeros_like(p.data)
        state["v"] = torch.zeros_like(p.data)
        state["alpha_t"] = alpha

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
        t = state.get("t", 0) # Get iteration number from the state, or initial value.
        state["m"] = state["m"] * beta1 + (1 - beta1) * p.grad.data
        state["v"] = state["v"] * beta2 + (1 - beta2) * p.grad.data**2
        if t == 0:
          state["alpha_t"] = alpha
        else:
          state["alpha_t"] = alpha * (1 - beta2**t)**0.5 / (1 - beta1**t)
        
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= state["alpha_t"] * state["m"] / (state["v"]**0.5 + eps) # Update weight tensor in-place.
        p.data = p.data - alpha * theta * p.data# Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
    return loss
