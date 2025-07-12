
import math
import torch
import torch.nn as nn
from einops import rearrange, einsum


class WeiLinear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    super(WeiLinear, self).__init__()
    w = torch.empty((out_features, in_features), device=device, dtype=dtype)
    s = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(w, std=s, a=-3*s, b = 3*s)
    self.w = nn.Parameter(w)

  def forward(self, x: torch.Tensor):
    return einsum(self.w, x, "out_features in_features, ... in_features -> ... out_features")


class WeiEmbedding(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    super(WeiEmbedding, self).__init__()
    embedding = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
    nn.init.trunc_normal_(embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)
    self.embedding = nn.Parameter(embedding)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    # dim=0 select rows
    reshaped = torch.reshape(token_ids, (-1,))
    re = torch.index_select(self.embedding, dim=0, index=reshaped)
    result_shape = token_ids.shape + (self.embedding.shape[-1],)
    return torch.reshape(re, result_shape)


class WeiRMSNorm(nn.Module):

  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    self.d_model = d_model
    self.eps = eps
    super(WeiRMSNorm, self).__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    avg = torch.sum(torch.square(x), dim=-1) / self.d_model
    result = x / torch.unsqueeze(torch.sqrt(avg + self.eps), -1)
    return result.to(in_dtype)
