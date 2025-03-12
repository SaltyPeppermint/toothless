import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from toothless.tree_model.components.mha import FastMultiHeadedAttention
from toothless.tree_model.components.utils import (
    FeedForward,
    MHAConfig,
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

        self.self_attn = FastMultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)

        self.sublayers = stack_modules(SublayerConnection(d_model, dropout), 2)

    def forward(self, src, start_nodes, end_nodes, rel_q, rel_k, rel_v) -> Tensor:
        src = self.sublayers[0](src, lambda x: self.self_attn(x, x, x, start_nodes, end_nodes, rel_q, rel_k, rel_v))
        src = self.sublayers[1](src, self.feed_forward)
        return src


class ASTEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: ASTEncoderLayer,
        num_layers: int,
        mha_config: MHAConfig,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
    ):
        super(ASTEncoder, self).__init__()
        self.layers = stack_modules(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mha_config = mha_config
        d_k = d_model // mha_config.total_heads()

        if mha_config.anc_heads > 0:
            self.anc_rel_emb = FastRelEmbeddings(d_k, mha_config.anc_heads, max_rel_pos, pos_type, dropout=dropout)
        if mha_config.sib_heads > 0:
            self.sib_rel_emb = FastRelEmbeddings(d_k, mha_config.sib_heads, max_rel_pos, pos_type, dropout=dropout)

        self.end_nodes = None

    def forward(self, src_data) -> Tensor:
        output = src_data.emb
        rel_anc_pos = src_data.anc_edges
        rel_sib_pos = src_data.sib_edges

        batch_size, max_rel_pos, max_ast_len = rel_anc_pos.size()
        rel_anc_q, rel_anc_k, rel_anc_v = None, None, None
        rel_sib_q, rel_sib_k, rel_sib_v = None, None, None
        if self.mha_config.anc_heads > 0:
            rel_anc_q, rel_anc_k, rel_anc_v = self.anc_rel_emb()
        if self.mha_config.sib_heads > 0:
            rel_sib_q, rel_sib_k, rel_sib_v = self.sib_rel_emb()

        rel_q = concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = concat_vec(rel_anc_k, rel_sib_k, dim=1)
        rel_v = concat_vec(rel_anc_v, rel_sib_v, dim=1)

        start_nodes = self.concat_pos(rel_anc_pos, rel_sib_pos)

        need_end_nodes = True
        if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
            need_end_nodes = False

        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.end_nodes = end_nodes.repeat(batch_size, self.mha_config.total_heads(), max_rel_pos, 1)

        for layer in self.layers:
            output = layer(output, start_nodes, self.end_nodes, rel_q, rel_k, rel_v)

        return self.norm(output)

    def concat_pos(self, rel_anc_pos, rel_sib_pos) -> Tensor:
        if self.anc_heads == 0:
            return rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        if self.sib_heads == 0:
            return rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)

        rel_anc_pos = rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)
        rel_sib_pos = rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        rel_pos = torch.cat([rel_anc_pos, rel_sib_pos], dim=1)

        return rel_pos
