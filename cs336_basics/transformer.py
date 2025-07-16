
import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor


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
    super(WeiRMSNorm, self).__init__()
    self.d_model = d_model
    self.eps = eps
    self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    avg = torch.sum(torch.square(x), dim=-1) / self.d_model
    result = x / torch.unsqueeze(torch.sqrt(avg + self.eps), -1) * self.g
    return result.to(in_dtype)


class WeiPositionwiseFfd(nn.Module):

  def __init__(self, d_model, d_ff, device=None, dtype=None):
    super(WeiPositionwiseFfd, self).__init__()
    self.w_1 = WeiLinear(d_model, d_ff, device=device, dtype=dtype)
    self.w_3 = WeiLinear(d_model, d_ff, device=device, dtype=dtype)
    self.w_2 = WeiLinear(d_ff, d_model, device=device, dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    s0 = self.w_1(x)
    s1 = (torch.sigmoid(s0) * s0) * self.w_3(x) 
    return self.w_2(s1)

class WeiRoPE(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super(WeiRoPE, self).__init__()
    self.theta = theta
    self.d_k = d_k
    self.max_seq_len = max_seq_len
    self.device = device

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    assert self.d_k % 2 == 0, "d_k must be even"
    assert self.max_seq_len >= x.shape[-2], "max sql len exceed"
    # reshape x to ... squ, d_k / 2, 2
    x_new_shape = x.shape[0:-1] + (self.d_k // 2, 2)
    x_reshaped = x.reshape(x_new_shape)

    # reshape other parameters to ... squ, d_k / 2, 1
    token_positions_reshaped = token_positions.unsqueeze(-1).unsqueeze(-1)
    token_positions_reshaped = token_positions_reshaped.repeat(tuple(1 for i in range(len(token_positions_reshaped.shape) - 2)) + (self.d_k // 2, 1))

    k = torch.tensor(range(0, self.d_k // 2), device=self.device)
    k = k.unsqueeze(-1)
    k_new_shape = x.shape[0:-1] + (self.d_k // 2, 1)
    k = k.expand(k_new_shape)
    
    result = torch.stack([
        # i = token_positions[..., 0]
        # k = 2*k[..., 0]
        # x = x_reshaped[..., 0] x_reshaped[..., 1]
        torch.cos(token_positions_reshaped[..., 0] / (self.theta ** (2*k[..., 0] / self.d_k))) * x_reshaped[..., 0] - torch.sin(token_positions_reshaped[..., 0] / (self.theta ** (2*k[..., 0] / self.d_k))) * x_reshaped[..., 1],
        torch.sin(token_positions_reshaped[..., 0] / (self.theta ** (2*k[..., 0] / self.d_k))) * x_reshaped[..., 0] + torch.cos(token_positions_reshaped[..., 0] / (self.theta ** (2*k[..., 0] / self.d_k))) * x_reshaped[..., 1]

    ], -1)
    return result.reshape(x.shape)

def wei_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
  max, max_i = torch.max(in_features, dim=dim, keepdim = True)
  exp = torch.exp(in_features - max)
  sum = torch.sum(exp, dim=dim, keepdim = True)
  return exp / sum


class WeiAttention(nn.Module):
  def __init__(self, device=None, dtype=None):
    super(WeiAttention, self).__init__()
    self.device = device
    self.dtype = dtype

  def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor  | None = None) -> torch.Tensor:
    att = einsum(q, k, "batch_size ... seq_len_q d_k , batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k")
    s_dk = math.sqrt(k.shape[-1])
    att = att / s_dk

    if mask is not None:
      int_mask = torch.where(mask, 0, -torch.inf)
      att = att + int_mask

    att = wei_softmax(att, dim=-1)
    return einsum(att, v, "batch_size ... seq_len_q seq_len_k , batch_size ... seq_len_k d_k -> batch_size ... seq_len_q d_k")

class WeiMultiHeadSelfAttention(nn.Module):
  def __init__(self, d_model: int, num_heads: int, d_att_in: int, d_q: int, d_k: int, d_v: int, device=None, dtype=None):
    super(WeiMultiHeadSelfAttention, self).__init__()
    self.w_q = WeiLinear(d_att_in, num_heads * d_q, device=device, dtype=dtype)
    self.w_k = WeiLinear(d_att_in, num_heads * d_k, device=device, dtype=dtype)
    self.w_v = WeiLinear(d_att_in, num_heads * d_v, device=device, dtype=dtype)
    self.attention = WeiAttention(device=device, dtype=dtype)
    self.w_o = WeiLinear(num_heads * d_v, d_model, device=device, dtype=dtype)
    self.device = device
    self.dtype = dtype

  def forward(self, in_features: torch.Tensor) -> torch.Tensor:
    q = self.w_q(in_features)
    k = self.w_k(in_features)
    v = self.w_v(in_features)
    sequence_length = in_features.shape[-2]
    mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=self.device)
    mask = torch.triu(mask).transpose(0,1)
    mask = mask.expand(in_features.shape[0: -2] + (sequence_length, sequence_length))
    att_output = self.attention(q, k, v, mask)
    return self.w_o(att_output)
