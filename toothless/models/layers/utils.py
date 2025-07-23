import copy
import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.pe = pe  # not sure if this works out correctly but it makes the error go away

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    RoPE encodes positional information by rotating query and key vectors.
    """

    def __init__(self, d_model, max_seq_len=8192, base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute rotation matrices for efficiency
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self.cached_seq_len = 0

    def _compute_rope_cache(self, seq_len, device, dtype):
        """Pre-compute cos and sin values for RoPE."""
        if seq_len <= self.cached_seq_len and self.cos_cached is not None:
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

        # Create position indices
        position = torch.arange(seq_len, device=device, dtype=dtype)

        # Create frequency indices (only for half the dimensions due to complex pairs)
        freq_seq = torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
        inv_freq = 1.0 / (self.base ** (freq_seq / self.d_model))

        # Compute outer product to get all position-frequency combinations
        freqs = torch.outer(position, inv_freq)  # [seq_len, d_model//2]

        # Duplicate frequencies to match full d_model
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_model]

        # Compute cos and sin
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        # Cache the results
        self.cos_cached = cos_vals
        self.sin_cached = sin_vals
        self.cached_seq_len = seq_len

        return cos_vals, sin_vals

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply RoPE to input tensor.
        x: [batch_size, seq_len, d_model]
        """

        seq_len = x.size(2)

        # Split into real and imaginary parts (treat pairs as complex numbers)
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply rotation
        cos = cos[:seq_len, 0::2]  # Match dimensions
        sin = sin[:seq_len, 0::2]

        # Rotate: (x1 + i*x2) * (cos + i*sin) = (x1*cos - x2*sin) + i*(x1*sin + x2*cos)
        rotated_x1 = x1 * cos.unsqueeze(0) - x2 * sin.unsqueeze(0)
        rotated_x2 = x1 * sin.unsqueeze(0) + x2 * cos.unsqueeze(0)

        # Recombine
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.flatten(-2)  # Merge last two dimensions

        return rotated

    def forward(self, t: Tensor, seq_len=None):
        """
        Apply RoPE to query and key tensors.

        Args:
            query: Query tensor [batch_size, seq_len, d_model] or [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor with same shape as query
            seq_len: Sequence length (inferred if None)

        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = t.shape[1]

        device = t.device
        dtype = t.dtype

        # Get cached cos/sin values
        cos, sin = self._compute_rope_cache(seq_len, device, dtype)

        return self._apply_rope(t, cos, sin)


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, dropout: float = 0.1, with_pos: bool = False):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if with_pos:
            self.pos_emb = RoPEPositionalEncoding(embedding_dim)
        else:
            self.pos_emb = None
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        words_embeddings = self.word_embeddings(x)
        if self.pos_emb is not None:
            words_embeddings = self.pos_emb(words_embeddings)

        embeddings = self.norm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])
