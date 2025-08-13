from torch import nn, Tensor

from .attention import RoPEMultiheadAttention
from .swiglu import SwiGLUFFN
from ..args import ModelArguments


class TransformerDecoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.self_attn_norm = nn.RMSNorm(conf.d_model)

        self.cross_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.cross_attn_norm = nn.RMSNorm(conf.d_model)

        self.feed_forward = SwiGLUFFN(conf.d_model, conf.dim_feed_forward)
        self.ff_norm = nn.RMSNorm(conf.d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # Self attention
        normed_tgt = self.self_attn_norm(tgt)
        attn_out, _ = self.self_attn(normed_tgt, normed_tgt, normed_tgt, tgt_mask, tgt_key_padding_mask)
        tgt = tgt + attn_out

        # Memory attention
        normed_memory = self.cross_attn_norm(memory)
        normed_tgt = self.cross_attn_norm(tgt)
        cross_attn_out, _ = self.cross_attn(
            normed_tgt, normed_memory, normed_memory, memory_mask, memory_key_padding_mask
        )
        tgt = tgt + cross_attn_out

        # Feedforward
        tgt = tgt + self.feed_forward(self.ff_norm(tgt))
        return tgt
