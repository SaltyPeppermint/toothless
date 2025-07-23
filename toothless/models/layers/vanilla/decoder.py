from torch import nn, Tensor
import torch.nn.functional as F

from .attention import RoPEMultiheadAttention
from ....args import ModelArguments


class TransformerDecoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.self_attn_norm = nn.LayerNorm(conf.d_model)

        self.cross_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.cross_attn_norm = nn.LayerNorm(conf.d_model)

        self.ff_up = nn.Linear(conf.d_model, conf.dim_feed_forward)
        self.ff_down = nn.Linear(conf.dim_feed_forward, conf.d_model)
        self.ff_norm = nn.LayerNorm(conf.d_model)

        self.dropout = nn.Dropout(conf.dropout)

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
        attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask)
        tgt = tgt + self.dropout(attn_out)
        tgt = self.self_attn_norm(tgt)

        # Memory attention
        cross_attn_out, _ = self.cross_attn(tgt, memory, memory, memory_mask, memory_key_padding_mask)
        tgt = tgt + self.dropout(cross_attn_out)
        tgt = self.cross_attn_norm(tgt)

        # Feedforward
        ff_out = self.ff_down(self.dropout(F.gelu(self.ff_up(tgt))))
        tgt = tgt + self.dropout(ff_out)
        tgt = self.ff_norm(tgt)
        return tgt
