import torch
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


class ASTDoubleDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feed_forward: int,
        dropout: float = 0.2,
        activation=F.gelu,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(ASTDoubleDecoderLayer, self).__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.num_heads = num_heads
        self.d_model = d_model

        self.self_attn = FastMHA(d_model, num_heads, dropout=dropout, cross_attn=False)
        self.l_cross_attn = FastMHA(d_model, num_heads, dropout=dropout, cross_attn=True)
        self.r_cross_attn = FastMHA(d_model, num_heads, dropout=dropout, cross_attn=True)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)

        self.dropout = nn.Dropout(dropout)
        self.sublayers = stack_layers(SublayerConnection(d_model, dropout), 4)

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
        tgt = self.sublayers[0](tgt, lambda x: self.self_attn(x, tgt_pos_indices, tgt_mask, rel_q=rel_q, rel_k=rel_k))

        tgt = self.sublayers[1](
            tgt,
            lambda x: self.l_cross_attn(
                x, tgt_pos_indices, l_mask, cross_state=l_mem, cross_pos_indices=l_pos_indices, rel_q=rel_q, rel_k=rel_k
            ),
        )

        tgt = self.sublayers[2](
            tgt,
            lambda x: self.r_cross_attn(
                x, tgt_pos_indices, r_mask, cross_state=r_mem, cross_pos_indices=r_pos_indices, rel_q=rel_q, rel_k=rel_k
            ),
        )

        tgt = self.sublayers[3](tgt, self.feed_forward)
        return tgt


class ASTDoubleDecoder(RelCoder):
    def __init__(
        self,
        decoder_layer: ASTDoubleDecoderLayer,
        num_layers: int,
        n_anc_heads: int,
        n_sib_heads: int,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(ASTDoubleDecoder, self).__init__(
            n_anc_heads, n_sib_heads, pos_type, max_rel_pos, d_model, dropout, device, dtype
        )
        self.layers = stack_layers(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self.tgt_pos_pad = None
        self.l_mem_pos_pad = None
        self.r_mem_pos_pad = None

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
