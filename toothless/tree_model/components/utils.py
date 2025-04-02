import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Buffer

from toothless.tree_model.args import ModelArguments


class Generator(nn.Module):
    def __init__(self, conf: ModelArguments, tgt_vocab_size: int):
        super(Generator, self).__init__()

        self.token_linear = nn.Linear(conf.d_model, tgt_vocab_size)
        self.token_dropout = nn.Dropout(conf.dropout)

    def forward(self, outputs: Tensor) -> Tensor:
        out = self.token_linear(outputs)
        return F.log_softmax(self.token_dropout(out), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feed_forward: int, dropout: float = 0.1, activation=F.gelu):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.pe = pe  # not sure if this works out correctly but it makes the error go away

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class RotaryPositionalEncoding(nn.Module):
    """
    Initialize the Rotary Positional Encoding.

    :param dim: Dimension of the embeddings (must be even)
    :param max_seq_len: Maximum sequence length to cache positional encodings for
    :param base: Base value for frequency computation (default: 5000.0)
    """

    def __init__(self, emb_size: int, max_seq_len: int = 256, base: float = 5000.0):
        super().__init__()

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, emb_size, 2) / emb_size))
        # Position indices
        t = torch.arange(max_seq_len)

        # Compute frequencies
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.inv_freq = Buffer(inv_freq, persistent=False)
        self.cos_cached = Buffer(emb.cos()[None, None, :, :], persistent=False)
        self.sin_cached = Buffer(emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x):
        """
        Rotate the second half of the feature dimension.

        :param x: Input tensor to rotate
        :return: Rotated tensor
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Apply rotary positional encoding to input tensor.

        :param x: Input tensor of shape [batch_size, seq_len, n_heads, d_k]
        :return: Tensor with rotary positional encoding applied
        """

        seq_len = x.size(1)

        # Apply rotary embedding
        cos = self.cos_cached[:, :, :seq_len, ...].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(x.device)

        return (x * cos) + (self._rotate_half(x) * sin)


class Embeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        with_pos: bool = False,
    ):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if with_pos:
            self.pos_emb = RotaryPositionalEncoding(embedding_dim)
        else:
            self.pos_emb = None
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        words_embeddings = self.word_embeddings(x)
        if self.pos_emb is not None:
            words_embeddings = self.pos_emb(words_embeddings)

        embeddings = self.norm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, n_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


# def c2p_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, relative_pos: Tensor) -> Tensor:
#     return c2p_pos.expand(
#         [
#             query_layer.size(0),
#             query_layer.size(1),
#             query_layer.size(2),
#             relative_pos.size(-1),
#         ]
#     )


# def p2c_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, key_layer: Tensor) -> Tensor:
#     return c2p_pos.expand(
#         [
#             query_layer.size(0),
#             query_layer.size(1),
#             key_layer.size(-2),
#             key_layer.size(-2),
#         ]
#     )


# def pos_dynamic_expand(pos_index: Tensor, p2c_att: Tensor, key_layer: Tensor) -> Tensor:
#     return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))
