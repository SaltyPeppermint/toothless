import copy
import torch
from torch import nn, Tensor


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) with cached frequency values.

    Args:
        dim: Dimension of the embedding (should be even)
        max_seq_len: Maximum sequence length for precomputed cache (default: 256)
        base: Base for frequency computation (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 256, base: float = 10000.0):
        super().__init__()

        assert dim % 2 == 0, "Embedding dimension must be even"

        # Precompute frequency inverse values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cache = None
        self._sin_cache = None

        self.max_seq_len = max_seq_len
        self._cache_seq_len = 0

    def _compute_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos/sin values for given sequence length."""
        if seq_len <= self._cache_seq_len and self._cos_cache is not None:
            return

        # Generate position indices
        t = torch.arange(seq_len, device=device, dtype=dtype)

        # Compute frequencies for all positions
        freqs = torch.outer(t, self.inv_freq)  # type: ignore # [seq_len, dim//2]

        # Create cos and sin caches
        self._cos_cache = torch.cos(freqs)  # [seq_len, dim//2]
        self._sin_cache = torch.sin(freqs)  # [seq_len, dim//2]
        self._cache_seq_len = seq_len
        return

    def _apply_rotary_pos_emb(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary positional embedding to input tensor."""
        # Split x into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]  # [..., dim//2] each

        # Apply rotation
        rotated = torch.stack(
            [
                x1 * cos - x2 * sin,  # Imaginary part
                x1 * sin + x2 * cos,  # Real part
            ],
            dim=-1,
        )

        # Flatten the last two dimensions back to original shape
        return rotated.flatten(-2)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """
        Apply rotary positional encoding to query and key tensors.

        Args:
            x: Tensor of shape [..., seq_len, dim]
            start_pos: Starting position for the sequence (useful for generation)

        Returns:
            Rotated Tensor
        """
        seq_len = x.size(-2)

        # Ensure cache is large enough
        cache_len = max(self.max_seq_len, start_pos + seq_len)
        self._compute_cos_sin_cache(cache_len, x.device, x.dtype)

        # Get cos/sin values for the current sequence
        cos = self._cos_cache[start_pos : start_pos + seq_len]  # type: ignore # [seq_len, dim//2]
        sin = self._sin_cache[start_pos : start_pos + seq_len]  # type: ignore # [seq_len, dim//2]

        # Expand dimensions to match input tensors
        # cos/sin shape: [seq_len, dim//2] -> [..., seq_len, dim//2]
        cos = cos[None, :, :].expand_as(x[..., ::2])
        sin = sin[None, :, :].expand_as(x[..., ::2])

        # Apply rotary embedding
        return self._apply_rotary_pos_emb(x, cos, sin)


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])
