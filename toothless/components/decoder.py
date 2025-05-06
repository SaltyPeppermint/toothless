import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from toothless.args import ModelArguments
from toothless.components.mha import MHTreeAttention
from toothless.components.rel_pos import RelCoder
from toothless.components.utils import FeedForward, stack_layers


class ASTDoubleDecoderLayer(nn.Module):
    def __init__(self, conf: ModelArguments, activation=F.gelu):
        super(ASTDoubleDecoderLayer, self).__init__()

        self.self_norm = nn.LayerNorm(conf.d_model)
        self.self_attn = MHTreeAttention(conf.d_model, conf.n_heads, conf.with_dis_attn, dropout=conf.dropout)
        self.self_dropout = nn.Dropout(conf.dropout)

        self.l_norm = nn.LayerNorm(conf.d_model)
        self.l_cross_attn = MHTreeAttention(
            conf.d_model, conf.n_heads, conf.with_dis_attn, dropout=conf.dropout, cross_attn=True
        )
        self.l_dropout = nn.Dropout(conf.dropout)

        self.r_norm = nn.LayerNorm(conf.d_model)
        self.r_cross_attn = MHTreeAttention(
            conf.d_model, conf.n_heads, conf.with_dis_attn, dropout=conf.dropout, cross_attn=True
        )
        self.r_dropout = nn.Dropout(conf.dropout)

        self.ff_norm = nn.LayerNorm(conf.d_model)
        self.feed_forward = FeedForward(
            conf.d_model, conf.dim_feed_forward, dropout=conf.dropout, activation=activation
        )
        self.ff_dropout = nn.Dropout(conf.dropout)

    def forward(
        self,
        tgt: Tensor,
        tgt_pos_indices: Tensor,
        tgt_mask: Tensor,
        l_mem: Tensor,
        l_pos_indices: Tensor,
        l_mask: Tensor,
        r_mem: Tensor,
        r_pos_indices: Tensor,
        r_mask: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
    ) -> Tensor:
        tgt = tgt + self.self_dropout(
            self.self_attn(self.self_norm(tgt), tgt_pos_indices, tgt_mask, rel_q=rel_q, rel_k=rel_k)
        )
        tgt = tgt + self.l_dropout(
            self.l_cross_attn(
                self.l_norm(tgt),
                tgt_pos_indices,
                l_mask,
                cross_state=l_mem,
                cross_pos_indices=l_pos_indices,
                rel_q=rel_q,
                rel_k=rel_k,
            )
        )
        tgt = tgt + self.r_dropout(
            self.r_cross_attn(
                self.r_norm(tgt),
                tgt_pos_indices,
                r_mask,
                cross_state=r_mem,
                cross_pos_indices=r_pos_indices,
                rel_q=rel_q,
                rel_k=rel_k,
            )
        )
        return tgt + self.ff_dropout(self.feed_forward(self.ff_norm(tgt)))


class ASTDoubleDecoder(RelCoder):
    def __init__(self, conf: ModelArguments, k: int):
        super(ASTDoubleDecoder, self).__init__(conf, k)
        decoder_layer = ASTDoubleDecoderLayer(conf, activation=F.gelu)

        self.layers = stack_layers(decoder_layer, conf.num_layers)
        self.norm = nn.LayerNorm(conf.d_model)

    def forward(
        self,
        tgt: Tensor,
        tgt_anc: Tensor,
        tgt_sib: Tensor,
        tgt_mask: Tensor,
        l_mem: Tensor,
        l_mem_anc: Tensor,
        l_mem_sib: Tensor,
        l_mask: Tensor,
        r_mem: Tensor,
        r_mem_anc: Tensor,
        r_mem_sib: Tensor,
        r_mask: Tensor,
    ) -> Tensor:
        """
        seq_len -1 for rightshift of train samples for autoregressive training

        :param tgt:         [batch_size, seq_len - 1, d_model]
        :param tgt_mask:    [batch_size, 1, seq_len - 1, seq_len - 1]
        :param l_mem:       [batch_size, seq_len, d_model]
        :param l_mask:      [batch_size, 1, 1, seq_len]
        :param r_mem:       [batch_size, seq_len, d_model]
        :param r_mask:      [batch_size, 1, 1, seq_len]
        :return             [batch_size, seq_len - 1, d_model]
        """
        rel_q, rel_k = self.rel_pos_emb()

        tgt_pos_indices = self.concat_pos(tgt_anc, tgt_sib)
        l_pos_indices = self.concat_pos(l_mem_anc, l_mem_sib)
        r_pos_indices = self.concat_pos(r_mem_anc, r_mem_sib)

        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                tgt_pos_indices,
                tgt_mask,
                l_mem,
                l_pos_indices,
                l_mask,
                r_mem,
                r_pos_indices,
                r_mask,
                rel_q,
                rel_k,
            )

        return self.norm(output)
