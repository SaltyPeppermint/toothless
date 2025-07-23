from torch import nn, Tensor
import torch.nn.functional as F

from .attention import RoPEMultiheadAttention
from ....args import ModelArguments


class TransformerEncoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(conf.d_model, conf.n_heads, conf.dropout)
        self.attn_norm = nn.LayerNorm(conf.d_model)

        self.ff_up = nn.Linear(conf.d_model, conf.dim_feed_forward)
        self.ff_down = nn.Linear(conf.dim_feed_forward, conf.d_model)
        self.ff_norm = nn.LayerNorm(conf.d_model)

        self.dropout = nn.Dropout(conf.dropout)

    def forward(self, src: Tensor, src_mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(src, src, src, src_mask, src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.attn_norm(src)

        # Feed forward with residual connection
        ff_output = self.ff_down(self.dropout(F.gelu(self.ff_up(src))))
        src = src + self.dropout(ff_output)
        src = self.ff_norm(src)
        return src
