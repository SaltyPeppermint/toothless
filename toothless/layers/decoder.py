import torch
from torch import nn, Tensor

from .attention import PackedRoPEMHA
from .swiglu import SwiGLUFFN
from ..args import ModelArgs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, conf: ModelArgs, device: torch.device | None = None, dtype: torch.dtype | None = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = PackedRoPEMHA(
            conf.d_model, conf.n_heads, conf.head_dim, dropout=conf.dropout, is_causal=True, **factory_kwargs
        )
        self.self_attn_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

        self.l_cross_attn = PackedRoPEMHA(
            conf.d_model, conf.n_heads, conf.head_dim, dropout=conf.dropout, **factory_kwargs
        )
        self.l_cross_attn_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

        self.r_cross_attn = PackedRoPEMHA(
            conf.d_model, conf.n_heads, conf.head_dim, dropout=conf.dropout, **factory_kwargs
        )
        self.r_cross_attn_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

        self.feed_forward = SwiGLUFFN(conf.d_model, conf.dim_feed_forward, **factory_kwargs)
        self.ff_norm = nn.RMSNorm(conf.d_model, **factory_kwargs)

    def forward(
        self, tgt: Tensor, tgt_mask: Tensor, l_mem: Tensor, l_mask: Tensor, r_mem: Tensor, r_mask: Tensor
    ) -> Tensor:
        # Self attention
        normed_tgt = self.self_attn_norm(tgt)
        attn_out = self.self_attn(normed_tgt, normed_tgt, normed_tgt, tgt_mask)
        tgt = tgt + attn_out

        # Left Memory attention
        normed_memory = self.l_cross_attn_norm(l_mem)
        normed_tgt = self.l_cross_attn_norm(tgt)
        l_cross_attn_out = self.l_cross_attn(normed_tgt, normed_memory, normed_memory, l_mask)

        # Right Memory attention
        normed_memory = self.r_cross_attn_norm(r_mem)
        normed_tgt = self.r_cross_attn_norm(tgt)
        r_cross_attn_out = self.l_cross_attn(normed_tgt, normed_memory, normed_memory, r_mask)

        tgt = tgt + l_cross_attn_out + r_cross_attn_out

        # Feedforward
        tgt = tgt + self.feed_forward(self.ff_norm(tgt))
        return tgt
