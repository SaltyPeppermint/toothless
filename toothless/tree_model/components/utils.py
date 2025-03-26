import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feed_forward: int, dropout: float = 0.1, activation=F.gelu):
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


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
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
            self.pos_emb = PositionalEncoding(embedding_dim)
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


class Generator(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super(Generator, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, outputs):
        out = self.linear(outputs)
        return F.log_softmax(self.dropout(out), dim=-1)


# class GreedyGenerator(nn.Module):
#     def __init__(
#         self, model: FastASTTrans, max_tgt_len: int, bos_token: int, unk_token
#     ):  # smth about multi gpu and model.module?
#         super(GreedyGenerator, self).__init__()

#         self.model = model
#         self.max_tgt_len = max_tgt_len
#         self.start_pos = bos_token
#         self.unk_pos = unk_token

#     def forward(self, data):
#         data.tgt_seq = None
#         self.model.process_data(data)

#         l_encoder_outputs = self.model.l_encode(data)
#         r_encoder_outputs = self.model.r_encode(data)

#         batch_size = r_encoder_outputs.size(0)
#         ys = torch.ones(batch_size, 1, requires_grad=False).fill_(self.start_pos).long().to(r_encoder_outputs.device)
#         for i in range(self.max_tgt_len - 1):
#             # data.tgt_mask = make_std_mask(ys, 0)
#             data.tgt_emb = self.model.tgt_embedding(ys)
#             decoder_outputs, decoder_attn = self.model.decode(data, l_encoder_outputs, r_encoder_outputs)

#             out = self.model.generator(decoder_outputs)
#             out = out[:, -1, :]
#             _, next_word = torch.max(out, dim=1)
#             ys = torch.cat([ys, next_word.unsqueeze(1).long().to(r_encoder_outputs.device)], dim=1)

#         return ys[:, 1:]


def concat_vec(vec1, vec2, dim):
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1
    return torch.cat([vec1, vec2], dim=dim)


def stack_layers(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def c2p_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, relative_pos: Tensor) -> Tensor:
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            relative_pos.size(-1),
        ]
    )


def p2c_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, key_layer: Tensor) -> Tensor:
    return c2p_pos.expand(
        [
            query_layer.size(0),
            query_layer.size(1),
            key_layer.size(-2),
            key_layer.size(-2),
        ]
    )


def pos_dynamic_expand(pos_index: Tensor, p2c_att: Tensor, key_layer: Tensor) -> Tensor:
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))
