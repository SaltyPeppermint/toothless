import copy
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from embeddings import FastRelEmbeddings
from toothless.tree_model.model import MHAConfig


class FastASTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feed_forward: int, dropout: float = 0.2, activation=F.gelu):
        super(FastASTEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feed_forward, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.sublayers = _stack_modules(SublayerConnection(d_model, dropout), 2)

    def forward(self, src, start_nodes, end_nodes, rel_q, rel_k, rel_v):
        src, attn_weights = self.sublayers[0](
            src, lambda x: self.self_attn(x, x, x, start_nodes, end_nodes, rel_q, rel_k, rel_v)
        )
        src, _ = self.sublayers[1](src, self.feed_forward)
        return src


class FastASTEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: FastASTEncoderLayer,
        num_layers: int,
        mha_config: MHAConfig,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
    ):
        super(FastASTEncoder, self).__init__()
        self.layers = _stack_modules(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mha_config = mha_config
        d_k = d_model // mha_config.total_heads()
        if mha_config.ancestor_heads > 0:
            self.anc_rel_emb = FastRelEmbeddings(d_k, mha_config.ancestor_heads, max_rel_pos, pos_type, dropout=dropout)
        if mha_config.sibling_heads > 0:
            self.sib_rel_emb = FastRelEmbeddings(d_k, mha_config.sibling_heads, max_rel_pos, pos_type, dropout=dropout)

        self.end_nodes = None

    def forward(self, data):
        output = data.src_emb
        rel_anc_pos = data.anc_edges
        rel_sib_pos = data.sib_edges

        batch_size, max_rel_pos, max_ast_len = rel_anc_pos.size()
        rel_anc_q, rel_anc_k, rel_anc_v = None, None, None
        rel_sib_q, rel_sib_k, rel_sib_v = None, None, None
        if self.mha_config.ancestor_heads > 0:
            rel_anc_q, rel_anc_k, rel_anc_v = self.anc_rel_emb()
        if self.mha_config.sibling_heads > 0:
            rel_sib_q, rel_sib_k, rel_sib_v = self.sib_rel_emb()
        rel_q = _concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = _concat_vec(rel_anc_k, rel_sib_k, dim=1)
        rel_v = _concat_vec(rel_anc_v, rel_sib_v, dim=1)

        start_nodes = self.concat_pos(rel_anc_pos, rel_sib_pos)

        need_end_nodes = True
        if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
            need_end_nodes = False

        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.end_nodes = end_nodes.repeat(batch_size, self.mha_config.total_heads(), max_rel_pos, 1)

        for i, layer in enumerate(self.layers):
            output = layer(output, start_nodes, self.end_nodes, rel_q, rel_k, rel_v)

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
        self.sublayers = _stack_modules(SublayerConnection(d_model, dropout), 3)

        # self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ):
        tgt, attn_weights = self.sublayers[0](
            tgt, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        )

        tgt, attn_weights = self.sublayers[1](
            tgt,
            lambda x: self.multihead_attn(
                x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
            ),
        )

        tgt, _ = self.sublayers[2](tgt, self.feed_forward)
        return tgt, attn_weights


class BaseDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.LayerNorm | None = None):
        super(BaseDecoder, self).__init__()
        self.layers = _stack_modules(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        attn_weights = None

        for mod in self.layers:
            output, attn_weights = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feed_forward: int, dropout: float = 0.1, activation=F.gelu):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: Tensor):
        return self.linear2(self.dropout(self.activation(self.linear1(x)))), None


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module):
        output, attn_weights = sublayer(self.norm(x))
        return x + self.dropout(output), attn_weights


def _concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def _stack_modules(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def c2p_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, relative_pos: Tensor):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


def p2c_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, key_layer: Tensor):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


def pos_dynamic_expand(pos_index: Tensor, p2c_att: Tensor, key_layer: Tensor):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


def transpose_for_scores(x: Tensor, num_heads: int):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)
