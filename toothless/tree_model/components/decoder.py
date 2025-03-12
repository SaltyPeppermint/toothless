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


class ASTDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feed_forward: int, dropout: float = 0.2, activation=F.gelu):
        super(ASTDecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.self_attn = FastMultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.l_multihead_attn = FastMultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.r_multihead_attn = FastMultiHeadedAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.sublayers = stack_modules(SublayerConnection(d_model, dropout), 4)

    # def forward(self, src, start_nodes, end_nodes, rel_q, rel_k, rel_v):
    #     src, attn_weights = self.sublayers[0](
    #         src, lambda x: self.self_attn(x, x, x, start_nodes, end_nodes, rel_q, rel_k, rel_v)
    #     )
    #     src, _ = self.sublayers[1](src, self.feed_forward)
    #     return src

    def forward(
        self,
        tgt: Tensor,
        l_mem: Tensor,
        r_mem: Tensor,
        rel_q: Tensor,
        rel_k: Tensor,
        rel_v: Tensor,
        tgt_mask: Tensor | None = None,
        l_mem_mask: Tensor | None = None,
        r_mem_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        l_mem_key_padding_mask: Tensor | None = None,
        r_mem_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        tgt = self.sublayers[0](
            tgt, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        )

        tgt = self.sublayers[1](
            tgt,
            lambda x: self.l_multihead_attn(
                x, l_mem, l_mem, attn_mask=l_mem_mask, key_padding_mask=l_mem_key_padding_mask
            ),
        )

        tgt = self.sublayers[2](
            tgt,
            lambda x: self.r_multihead_attn(
                x, r_mem, r_mem, attn_mask=r_mem_mask, key_padding_mask=r_mem_key_padding_mask
            ),
        )

        tgt = self.sublayers[3](tgt, self.feed_forward)
        return tgt


class ASTDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: ASTDecoderLayer,
        num_layers: int,
        mha_config: MHAConfig,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
    ):
        super(ASTDecoder, self).__init__()
        self.layers = stack_modules(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mha_config = mha_config
        d_k = d_model // mha_config.total_heads()

        if mha_config.anc_heads > 0:
            self.anc_rel_emb = FastRelEmbeddings(d_k, mha_config.anc_heads, max_rel_pos, pos_type, dropout=dropout)
        if mha_config.sib_heads > 0:
            self.sib_rel_emb = FastRelEmbeddings(d_k, mha_config.sib_heads, max_rel_pos, pos_type, dropout=dropout)

        self.end_nodes = None

    def forward(
        self,
        tgt_data,
        l_mem_data,
        r_mem_data,
        tgt_mask,
        l_mem_mask=None,
        r_mem_mask=None,
        tgt_key_padding_mask=None,
        l_mem_key_padding_mask=None,
        r_mem_key_padding_mask=None,
    ) -> tuple[Tensor | None]:
        batch_size, max_rel_pos, max_ast_len = tgt_data.anc_edges.size()
        rel_anc_q, rel_anc_k, rel_anc_v = None, None, None
        rel_sib_q, rel_sib_k, rel_sib_v = None, None, None
        if self.mha_config.anc_heads > 0:
            rel_anc_q, rel_anc_k, rel_anc_v = self.anc_rel_emb()
        if self.mha_config.sib_heads > 0:
            rel_sib_q, rel_sib_k, rel_sib_v = self.sib_rel_emb()

        rel_q = concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = concat_vec(rel_anc_k, rel_sib_k, dim=1)
        rel_v = concat_vec(rel_anc_v, rel_sib_v, dim=1)

        tgt_start_nodes = self.concat_pos(tgt_data.anc_edges, tgt_data.sib_edges)
        l_mem_start_nodes = self.concat_pos(l_mem_data.anc_edges, l_mem_data.sib_edges)
        r_mem_start_nodes = self.concat_pos(r_mem_data.anc_edges, r_mem_data.sib_edges)

        need_end_nodes = True
        if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
            need_end_nodes = False

        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=tgt_start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.end_nodes = end_nodes.repeat(batch_size, self.mha_config.total_heads(), max_rel_pos, 1)

        output = None
        for layer in self.layers:
            # output, attn_weights = layer(tgt_data.src_emb, tgt_start_nodes, self.end_nodes, rel_q, rel_k, rel_v)
            output = layer(
                tgt_data.emb,
                l_mem_data.emb,
                r_mem_data.emb,
                rel_q,
                rel_k,
                rel_v,
                tgt_mask=tgt_mask,
                l_memory_mask=l_mem_mask,
                r_memory_mask=r_mem_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                l_memory_key_padding_mask=l_mem_key_padding_mask,
                r_memory_key_padding_mask=r_mem_key_padding_mask,
            )

        # if self.norm is not None:
        #     output = self.norm(output)

        return self.norm(output)

    def concat_pos(self, rel_anc_pos, rel_sib_pos):
        if self.anc_heads == 0:
            return rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        if self.sib_heads == 0:
            return rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)

        rel_anc_pos = rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)
        rel_sib_pos = rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        rel_pos = torch.cat([rel_anc_pos, rel_sib_pos], dim=1)

        return rel_pos


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, dim_feed_forward: int = 2048, dropout: float = 0.2, activation=F.gelu
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout)
        self.sublayers = stack_modules(SublayerConnection(d_model, dropout), 3)

        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        tgt = self.sublayers[0](
            tgt, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        )

        tgt = self.sublayers[1](
            tgt,
            lambda x: self.multihead_attn(
                x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
            ),
        )

        tgt = self.sublayers[2](tgt, self.feed_forward)
        return tgt


class BaseDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.LayerNorm | None = None):
        super(BaseDecoder, self).__init__()
        self.layers = stack_modules(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
