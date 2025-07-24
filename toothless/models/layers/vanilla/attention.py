from typing import Tuple
import torch
from torch import nn, Tensor

from ..utils import RotaryPositionalEncoding


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, q_seq_len, d_model = query.shape
        kv_seq_len = key.shape[1]

        # Linear projections
        q = self.q_proj(query)  # [batch, seq_len, d_model]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, q_seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, kv_seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # Transpose for attention computation: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(attn_mask, float("-inf")).type(v.dtype)

        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float("-inf")).type(v.dtype)

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, q_seq_len, d_model)

        # Final linear projection
        output = self.out_proj(output)

        return output, attn_weights
