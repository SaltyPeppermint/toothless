import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feed_forward: int,
        dropout: float = 0.1,
        activation=F.gelu,
    ):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        output = sublayer(self.norm(x))
        return x + self.dropout(output)


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, N: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def c2p_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, relative_pos: Tensor):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            relative_pos.size(-1),
        ]
    )


def p2c_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, key_layer: Tensor):
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            key_layer.size(-2),
            key_layer.size(-2),
        ]
    )


def pos_dynamic_expand(pos_index: Tensor, p2c_att: Tensor, key_layer: Tensor):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))
