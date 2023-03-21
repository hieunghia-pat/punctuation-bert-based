import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import math

class SelfAttention(nn.Module):
  """
    Self-Attention mechanism.
  """
  def __init__(self, emb_dim, kqv_dim, num_heads=1):
    super(SelfAttention, self).__init__()
    self.emb_dim = emb_dim
    self.kqv_dim = kqv_dim
    self.num_heads = num_heads

    # self.w_k = nn.Linear(emb_dim, kqv_dim*num_heads, bias=False)
    # self.w_q = nn.Linear(emb_dim, kqv_dim*num_heads, bias=False)
    # self.w_v = nn.Linear(emb_dim, kqv_dim*num_heads, bias=False)
    # self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)
  
  def get_mask(self):
    pass

  # def forward(self, inputs, attention_mask=None):
  #   b, t, _ = inputs.shape
  #   e = self.kqv_dim
  #   h = self.num_heads

  #   if attention_mask is not None:
  #     attention_mask = attention_mask.unsqueeze(-1)
  #     inputs = inputs * attention_mask
    
  #   keys = self.w_k(inputs).view(b, t, h, e)
  #   values = self.w_v(inputs).view(b, t, h, e)
  #   queries = self.w_q(inputs).view(b, t, h ,e)

  #   keys = keys.transpose(1, 2).contiguous().view(b*h, t, e)
  #   queries = queries.transpose(1, 2).contiguous().view(b*h, t, e)
  #   values = values.transpose(1, 2).contiguous().view(b*h, t, e)

  #   dot = torch.bmm(queries, keys.transpose(1, 2))
  #   dot = dot / np.sqrt(e)

  #   if attention_mask is not None:
  #     dot = dot * attention_mask
  #   dot = F.softmax(dot, dim=2)
  #   out = torch.bmm(dot, values).view(b, h, t, e)
  #   out = out.transpose(1, 2).contiguous().view(b, t, h*e)
  #   if attention_mask is not None:
  #     out = out * attention_mask
  #   out = self.w_out(out)
    
  #   return out

  # def forward(self, inputs, attention_mask=None):
  #   b, t, _ = inputs.shape
  #   e = self.kqv_dim
  #   h = self.num_heads

  #   if attention_mask is not None:
  #     attention_mask = attention_mask.unsqueeze(-1)
  #     inputs = inputs * attention_mask
    
  #   keys = inputs
  #   values = inputs
  #   queries = inputs

  #   dot = torch.bmm(queries, keys.transpose(1, 2))

  #   if attention_mask is not None:
  #     dot = dot * attention_mask
  #   dot = F.softmax(dot, dim=2)
  #   out = torch.bmm(dot, values)
  #   if attention_mask is not None:
  #     out = out * attention_mask

  #   return out

  def forward(self, inputs, attention_mask=None):
    e = self.kqv_dim
    keys = inputs
    values = inputs
    queries = inputs

    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(e)

    if attention_mask is not None:
      attention_mask = attention_mask.unsqueeze(1)
      scores = scores.masked_fill(attention_mask==0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    out = torch.matmul(scores, values)

    return out

    
