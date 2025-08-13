from torch import nn, Tensor

from .attention import RoPEMultiheadAttention
from .swiglu import SwiGLUFFN
from ..args import ModelArguments


class TransformerEncoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.attn_norm = nn.RMSNorm(conf.d_model)

        self.feed_forward = SwiGLUFFN(conf.d_model, conf.dim_feed_forward)
        self.ff_norm = nn.RMSNorm(conf.d_model)

    def forward(self, src: Tensor, src_mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None):
        # Self-attention with residual connection
        normed_src = self.attn_norm(src)
        self_attn_out, _ = self.self_attn(normed_src, normed_src, normed_src, src_mask, src_key_padding_mask)
        src = src + self_attn_out
        # Feed forward with residual connection
        src = src + self.feed_forward(self.ff_norm(src))

        return src
