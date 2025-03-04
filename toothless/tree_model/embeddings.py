import math

import torch
from torch import nn
from torch import Tensor


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
    def __init__(self, embedding_dim: int, vocab_size: int, dropout: float = 0.1, with_pos: bool = False):
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


class RelEmbeddings(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int, pos_type: list[str], dropout: float = 0.0):
        super(RelEmbeddings, self).__init__()

        self.d_model = d_model
        self.k = 2 * k + 2
        self.pos_type = pos_type
        self.num_heads = num_heads
        if "p2q" in pos_type:
            self.rel_emb_q = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)  # pad id=k+1 -> zero
        if "p2k" in pos_type:
            self.rel_emb_k = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)
        if "p2v" in pos_type:
            self.rel_emb_v = nn.Embedding(self.k, d_model, padding_idx=self.k // 2)
        self.dropout = nn.Dropout(dropout)

    def get_rel_weights(self, rel_params: Tensor) -> Tensor:
        rel_params = rel_params * math.sqrt(self.d_model)
        rel_params = self.dropout(rel_params)

        rel_params = rel_params.unsqueeze(0).unsqueeze(0)
        rel_params = rel_params.repeat(1, self.num_heads, 1, 1)

        return rel_params

    def get_p2v_emb(self, inputs: Tensor) -> Tensor | None:
        if "p2v" in self.pos_type:
            rel_v = self.rel_emb_v(inputs) * math.sqrt(self.d_model)
            rel_v = self.dropout(rel_v)
            rel_v = rel_v.repeat(1, 1, 1, self.num_heads)
            return rel_v
        else:
            return None


class DebertaRelEmbeddings(RelEmbeddings):
    def __init__(self, d_model: int, num_heads: int, k: int, pos_type, dropout: float = 0.0):
        super(DebertaRelEmbeddings, self).__init__(d_model, num_heads, k, pos_type, dropout)

    def forward(self, inputs: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        rel_q, rel_k, rel_v = None, None, None
        if "p2q" in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if "p2k" in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        if "p2v" in self.pos_type:
            rel_v = self.get_p2v_emb(inputs)

        return rel_q, rel_k, rel_v


class FastRelEmbeddings(RelEmbeddings):
    def __init__(self, d_model: int, num_heads: int, k, pos_type: list[str], dropout: float = 0.0):
        super(FastRelEmbeddings, self).__init__(d_model, num_heads, k, pos_type, dropout)

    def forward(self) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        rel_q, rel_k, rel_v = None, None, None
        if "p2q" in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if "p2k" in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        if "p2v" in self.pos_type:
            rel_v = self.get_rel_weights(self.rel_emb_v.weight)

        return rel_q, rel_k, rel_v


def build_relative_position(
    query_size: int, key_size: int, max_relative_positions: int, device: int, need_traverse=False
) -> Tensor:
    """
    :return: obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    if need_traverse:
        rel_pos_ids = -rel_pos_ids
    rel_pos_ids += max_relative_positions + 1
    rel_pos_ids = torch.clamp(rel_pos_ids, 1, 2 * max_relative_positions + 1)
    return rel_pos_ids
