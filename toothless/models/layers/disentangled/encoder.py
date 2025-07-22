import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ....args import ModelArguments
from .mha import MHTreeAttention
from .rel_pos import RelCoder
from ..utils import FeedForward, stack_layers


class ASTEncoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments, activation=F.gelu):
        super(ASTEncoderLayer, self).__init__()

        self.self_attn = MHTreeAttention(
            conf.d_model, conf.n_heads, conf.with_dis_attn, dropout=conf.dropout, cross_attn=False
        )
        self.self_norm = nn.LayerNorm(conf.d_model)
        self.self_dropout = nn.Dropout(conf.dropout)

        self.feed_forward = FeedForward(
            conf.d_model, conf.dim_feed_forward, dropout=conf.dropout, activation=activation
        )
        self.ff_norm = nn.LayerNorm(conf.d_model)
        self.ff_dropout = nn.Dropout(conf.dropout)

    def forward(
        self, src: Tensor, pos_indices: Tensor, mask: Tensor, rel_q: Tensor | None, rel_k: Tensor | None
    ) -> Tensor:
        src = src + self.self_dropout(self.self_attn(self.self_norm(src), pos_indices, mask, rel_q=rel_q, rel_k=rel_k))
        src = src + self.ff_dropout(self.feed_forward(self.ff_norm(src)))

        return src


class DisASTEncoder(RelCoder):
    def __init__(self, conf: ModelArguments, k: int):
        super(DisASTEncoder, self).__init__(conf, k)
        encoder_layer = ASTEncoderLayer(conf, activation=F.gelu)

        self.layers = stack_layers(encoder_layer, conf.num_layers)
        self.norm = nn.LayerNorm(conf.d_model)

    def forward(self, src: Tensor, src_anc: Tensor, src_sib: Tensor, mask: Tensor) -> Tensor:
        """
        seq_len -1 for rightshift of train samples for autoregressive training

        :param src:     [batch_size, seq_len -1, d_model]
        :param mask:    [batch_size, 1, 1, seq_len]
        :return         [batch_size, seq_len -1, d_model]
        """
        rel_q, rel_k = self.rel_pos_emb()

        pos_indices = self.concat_pos(src_anc, src_sib)

        output = src  # batch_size, max_len, d_model
        for layer in self.layers:
            output = layer(output, pos_indices, mask, rel_q, rel_k)

        return self.norm(output)
