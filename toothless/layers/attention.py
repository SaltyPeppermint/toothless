from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .rope import RotaryPositionalEncoding


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        # Linear projections
        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
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


class PackedRoPEMHA(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        d_model: Size of embedding dim
        total_head_dim (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // n_heads
        n_heads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.bias = bias
        self.head_dim = head_dim
        self.total_head_dim = head_dim * n_heads

        self.packed_proj = nn.Linear(d_model, self.total_head_dim * 3, bias=bias, **factory_kwargs)
        self.rope = RotaryPositionalEncoding(self.head_dim)
        self.out_proj = nn.Linear(self.total_head_dim, d_model, bias=bias, **factory_kwargs)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Rotate query and key
            4. Prepare for SDPA
            5. Run SDPA
            6. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``batch``, ``q_seq_len``, ``d_model``)
            key (torch.Tensor): key of shape (``batch``, ``kv_seq_len``, ``d_model``)
            value (torch.Tensor): value of shape (``batch``, ``kv_seq_len``, ``d_model``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``batch``, ``q_seq_len``, ``kv_seq_len``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape [batch, q_seq_len, E_q)
        """
        # Step 1. Apply input projection
        if query is key and key is value:
            result = self.packed_proj(query)
            query, key, value = torch.chunk(result, 3, dim=-1)
        else:
            q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim=0)
            if self.bias:
                q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            query, key, value = (
                F.linear(query, q_weight, q_bias),
                F.linear(key, k_weight, k_bias),
                F.linear(value, v_weight, v_bias),
            )

        # Step 2. Split heads
        # reshape query, key, value to separate by head
        # [batch, q_seq_len, total_head_dim] -> [batch, q_seq_len, n_heads, head_dim]
        query = query.unflatten(-1, [self.n_heads, self.head_dim])
        # [batch, kv_seq_len, total_head_dim] -> [batch, kv_seq_len, n_heads, head_dim]
        key = key.unflatten(-1, [self.n_heads, self.head_dim])
        # [batch, kv_seq_len, total_head_dim] -> [batch, kv_seq_len, n_heads, head_dim]
        value = value.unflatten(-1, [self.n_heads, self.head_dim])

        # Step 3. Apply RoPE to Q and K
        query = self.rope(query)
        key = self.rope(key)

        # Step 4: Transpose for attention computation:
        # [batch, q_seq_len, n_heads, head_dim] -> [batch, n_heads, q_seq_len, head_dim]
        query = query.transpose(1, 2)
        # [batch, kv_seq_len, n_heads, head_dim] -> [batch, n_heads, kv_seq_len, head_dim]
        key = key.transpose(1, 2)
        # [batch, kv_seq_len, n_heads, head_dim] -> [batch, n_heads, kv_seq_len, head_dim]
        value = value.transpose(1, 2)

        if is_causal:
            L, S = query.shape[-2], key.shape[-2]
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=padding_mask.device).tril(diagonal=0)
            attn_mask = padding_mask | causal_mask
        else:
            attn_mask = padding_mask

        # Step 5. Run SDPA
        # [batch, n_heads, q_seq_len, head_dim]
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal, attn_mask=attn_mask
        )
        # [batch, n_heads, q_seq_len, head_dim] -> [batch, q_seq_len, n_heads, head_dim] -> [batch, q_seq_len, total_head_dim]
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 6. Apply output projection
        # [batch, q_seq_len, total_head_dim] -> [batch, q_seq_len, d_model]
        attn_output = self.out_proj(attn_output)

        return attn_output
