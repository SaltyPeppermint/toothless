import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from toothless.tree_model.components.mha import FastMHA
from toothless.tree_model.components.rel_pos import RelCoder
from toothless.tree_model.components.utils import (
    FeedForward,
    SublayerConnection,
    stack_layers,
)


class ASTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feed_forward: int,
        dropout: float = 0.2,
        activation=F.gelu,
    ):
        super(ASTEncoderLayer, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.self_attn = FastMHA(d_model, num_heads, dropout=dropout, cross_attn=False)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)
        self.sublayers = stack_layers(SublayerConnection(d_model, dropout), 2)

    def forward(
        self,
        src: Tensor,
        pos_indices: Tensor,
        mask: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
    ) -> Tensor:
        """ """
        src = self.sublayers[0](src, lambda x: self.self_attn(x, pos_indices, mask, rel_q=rel_q, rel_k=rel_k))
        src = self.sublayers[1](src, self.feed_forward)
        return src


class ASTEncoder(RelCoder):
    def __init__(
        self,
        d_model: int,
        dim_feed_forward: int,
        num_layers: int,
        n_anc_heads: int,
        n_sib_heads: int,
        pos_type: list[str],
        max_rel_pos: int,
        dropout: float = 0.2,
    ):
        super(ASTEncoder, self).__init__(n_anc_heads, n_sib_heads, pos_type, max_rel_pos, d_model, dropout)
        encoder_layer = ASTEncoderLayer(
            d_model, n_anc_heads + n_sib_heads, dim_feed_forward, dropout, activation=F.gelu
        )

        self.layers = stack_layers(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

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
