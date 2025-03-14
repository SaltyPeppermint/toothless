import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from toothless.tree_model.components.mha import MHSelfAttn
from toothless.tree_model.components.utils import (
    FeedForward,
    SublayerConnection,
    concat_vec,
    stack_modules,
)
from toothless.tree_model.embeddings import FastRelEmbeddings


class ASTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feed_forward: int, dropout: float = 0.2, activation=F.gelu):
        super(ASTEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.self_attn = MHSelfAttn(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)

        self.sublayers = stack_modules(SublayerConnection(d_model, dropout), 2)

    def forward(self, src, pos_enc, pos_enc_padding, rel_q, rel_k, rel_v) -> Tensor:
        src = self.sublayers[0](src, lambda x: self.self_attn(x, x, x, pos_enc, pos_enc_padding, rel_q, rel_k, rel_v))
        src = self.sublayers[1](src, self.feed_forward)
        return src


class ASTEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: ASTEncoderLayer,
        num_layers: int,
        n_anc_heads: int,
        n_sib_heads: int,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
    ):
        super(ASTEncoder, self).__init__()
        self.layers = stack_modules(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.n_anc_heads = n_anc_heads
        self.n_sib_heads = n_sib_heads
        d_k = d_model // (n_anc_heads + n_sib_heads)

        if n_anc_heads > 0:
            self.anc_rel_emb = FastRelEmbeddings(d_k, n_anc_heads, max_rel_pos, pos_type, dropout=dropout)
        if n_sib_heads > 0:
            self.sib_rel_emb = FastRelEmbeddings(d_k, n_sib_heads, max_rel_pos, pos_type, dropout=dropout)

        self.pos_enc_padding = None

    def forward(self, src_data) -> Tensor:
        batch_size, max_rel_pos, max_ast_len = src_data.anc_edges.size()
        rel_q, rel_k, rel_v = self.rel_pos_emb()

        # Formerly "Start Nodes"
        pos_enc = self.concat_pos(src_data.anc_edges, src_data.sib_edges)

        need_pos_enc_padding = True
        if self.pos_enc_padding is not None and batch_size == self.pos_enc_padding.size(0):
            need_pos_enc_padding = False

        # Formerly "End Nodes"
        if need_pos_enc_padding:
            pos_enc_padding = torch.arange(max_ast_len, device=pos_enc.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.pos_enc_padding = pos_enc_padding.repeat(
                batch_size, self.n_anc_heads + self.n_sib_heads, max_rel_pos, 1
            )

        output = src_data.emb
        for layer in self.layers:
            output = layer(output, pos_enc, self.pos_enc_padding, rel_q, rel_k, rel_v)

        return self.norm(output)

    def rel_pos_emb(self):
        rel_anc_q, rel_anc_k, rel_anc_v = None, None, None
        rel_sib_q, rel_sib_k, rel_sib_v = None, None, None
        if self.n_anc_heads > 0:
            rel_anc_q, rel_anc_k, rel_anc_v = self.anc_rel_emb()
        if self.n_sib_heads > 0:
            rel_sib_q, rel_sib_k, rel_sib_v = self.sib_rel_emb()

        rel_q = concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = concat_vec(rel_anc_k, rel_sib_k, dim=1)
        rel_v = concat_vec(rel_anc_v, rel_sib_v, dim=1)
        return rel_q, rel_k, rel_v

    def concat_pos(self, rel_anc_pos, rel_sib_pos) -> Tensor:
        if self.anc_heads == 0:
            return rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        if self.sib_heads == 0:
            return rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)

        rel_anc_pos = rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)
        rel_sib_pos = rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        rel_pos = torch.cat([rel_anc_pos, rel_sib_pos], dim=1)

        return rel_pos
