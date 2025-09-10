import torch
from torch import nn, Tensor

from .attention import PackedRoPEMHA
from .swiglu import SwiGLUFFN
from ..args import ModelArgs


class TransformerEncoderLayer(nn.Module):
    def __init__(self, conf: ModelArgs, device: torch.device | None = None, dtype: torch.dtype | None = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = PackedRoPEMHA(
            conf.d_model, conf.n_heads, conf.head_dim, dropout=conf.dropout, **factory_kwargs
        )
        self.attn_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

        self.feed_forward = SwiGLUFFN(conf.d_model, conf.dim_feed_forward, **factory_kwargs)
        self.ff_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

    def forward(self, src: Tensor, padding_mask: Tensor):
        # Self-attention with residual connection
        normed_src = self.attn_norm(src)
        self_attn_out = self.self_attn(normed_src, normed_src, normed_src, padding_mask, is_causal=False)
        src = src + self_attn_out
        # Feed forward with residual connection
        src = src + self.feed_forward(self.ff_norm(src))

        return src
