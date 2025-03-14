import math

import torch
import torch.nn as nn
from torch import Tensor

from toothless.tree_model.components.utils import stack_modules


class FastMultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, cross_attn: bool = False):
        super(FastMultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.cross_attn = cross_attn
        self.num_heads = num_heads
        self.linear_layers = stack_modules(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def finalize_output(self, output: Tensor) -> Tensor:
        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.linear_layers[-1](output)

        return output


class MHSelfAttn(FastMultiHeadedAttention):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MHSelfAttn, self).__init__(d_model, num_heads, dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_enc: Tensor,
        pos_pad: Tensor,
        rel_q: Tensor | None = None,
        rel_k: Tensor | None = None,
        rel_v: Tensor | None = None,
    ) -> Tensor:
        """relative q shape [1, 2k+1, dim]"""
        """relative v shape [1, 2k+1, dim]"""
        """"""
        query, key, value = [
            transpose_for_scores(layer(x), self.num_heads) for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        output = self.rel_attn(query, key, value, pos_enc, pos_pad, rel_q, rel_k, rel_v)

        return self.finalize_output(output)

    @staticmethod
    def rel_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos_enc: Tensor,
        pos_pad: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
        rel_v: Tensor | None,
    ):
        """
        :param q: [batch_size, num_heads, seq_len, d_k]
        :param k: [batch_size, num_heads, seq_len, d_k]
        :param v: [batch_size, num_heads, seq_len, d_k]
        :param pos_enc: [batch_size, num_heads, k+1, seq_len]
        :param pos_pad: [batch_size, num_heads, k+1, seq_len]
        :param rel_q: [1, num_heads, 2k+2, d_k]
        :param rel_k: [1, num_heads, 2k+2, d_k]
        :param rel_v: [1, num_heads, 2k+2, d_k]
        :return:
        """
        batch_size, num_heads, seq_len, d_k = q.size()
        # max_rel_pos = pos_enc.size(2)
        scale_factor = 1

        q_weights = q.contiguous().view(-1, d_k)
        k_weights = k.contiguous().view(-1, d_k)
        v_weights = v.contiguous().view(-1, d_k)

        pos_enc_mask = pos_enc.eq(-1)

        pos_enc += pos_enc_mask
        mask_matrix = torch.cat([pos_enc_mask, pos_enc_mask], dim=-2)
        mask_matrix = mask_matrix.view(batch_size, num_heads, -1)

        mask_matrix[:, :, :seq_len] = True

        map_pos = torch.arange(batch_size * num_heads, dtype=torch.long, device=pos_enc.device)
        map_pos = map_pos.unsqueeze(-1) * seq_len

        query_indexes = torch.cat([pos_pad, pos_enc], dim=-2)
        key_indexes = torch.cat([pos_enc, pos_pad], dim=-2)

        query_indexes = query_indexes.view(batch_size * num_heads, -1)
        key_indexes = key_indexes.view(batch_size * num_heads, -1)

        query_indexes += map_pos
        key_indexes += map_pos

        # query and key vec of context. shape [batch_size * num_heads * (k+1) * 2, seq_len, d_k]
        q_context = torch.embedding(q_weights, query_indexes)
        k_context = torch.embedding(k_weights, key_indexes)
        v_context = torch.embedding(v_weights, key_indexes)

        q_context = q_context.view(batch_size, num_heads, -1, seq_len, d_k)
        k_context = k_context.view(batch_size, num_heads, -1, seq_len, d_k)
        v_context = v_context.view(batch_size, num_heads, -1, seq_len, d_k)

        # Attention Calculation
        # context -> context
        c2c = torch.mul(q_context, k_context).sum(dim=-1)
        scores = c2c

        # context -> position
        if rel_k is not None:
            k_pos = rel_k.unsqueeze(-2)
            c2p = torch.mul(q_context, k_pos).sum(dim=-1)
            scores += c2p
            scale_factor += 1

        # position -> context
        if rel_q is not None:
            q_pos = rel_q.unsqueeze(-2)
            p2c = torch.mul(q_pos, k_context).sum(dim=-1)
            scores += p2c
            scale_factor += 1

        # scores = (c2c + c2p + p2c) / (3 * sqrt(d_k))
        scores = scores / (scale_factor * math.sqrt(d_k))
        scores = scores.view(batch_size, num_heads, -1)

        # score shape [batch_size, num_heads, 2(k+1) * seq_len]
        # index shape [batch_size, num_heads, 2(k+1) * seq_len]
        query_indexes = query_indexes - map_pos
        query_indexes = query_indexes.view(batch_size, num_heads, -1)

        # Mask out attention scores according to mask
        scores = scores.masked_fill(mask_matrix, -1e9)

        scores = torch.exp(scores)

        # Softmax Calculation
        # Above the line of the softmax
        score_sum = torch.zeros(scores.size(), device=scores.device)
        score_sum = score_sum.scatter_add(dim=-1, index=query_indexes, src=scores)

        score_sum += score_sum.eq(0) * -1e9
        score_sum = torch.gather(score_sum, index=query_indexes, dim=-1)

        # [batch_size, num_heads, 2(k+1), seq_len, 1]
        # Lower half of softmax
        scores = (scores / score_sum).view(batch_size, num_heads, -1, seq_len, 1)

        # shape [batch_size, num_heads, 2(k+1), seq_len, d_k]
        if rel_v is not None:
            v_context += rel_v.unsqueeze(-2)
        attn_v = torch.mul(scores, v_context)

        output = torch.zeros(attn_v.size(), device=attn_v.device)
        query_indexes = query_indexes.view(batch_size, num_heads, -1, seq_len).unsqueeze(-1)
        query_indexes = query_indexes.repeat_interleave(repeats=d_k, dim=-1)
        output = output.scatter_add(dim=-2, index=query_indexes, src=attn_v)

        output = output.sum(dim=-3)

        return output


class MHCrossAttn(FastMultiHeadedAttention):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MHCrossAttn, self).__init__(d_model, num_heads, dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_enc: Tensor,
        pos_pad: Tensor,
        cross_pos_enc: Tensor,
        cross_pos_pad: Tensor,
        rel_q: Tensor | None = None,
        rel_k: Tensor | None = None,
        rel_v: Tensor | None = None,
    ) -> Tensor:
        """relative q shape [1, 2k+1, dim]"""
        """relative v shape [1, 2k+1, dim]"""
        """"""
        query, key, value = [
            transpose_for_scores(layer(x), self.num_heads) for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        output = self.rel_cross_attn(
            query, key, value, pos_enc, pos_pad, cross_pos_enc, cross_pos_pad, rel_q, rel_k, rel_v
        )

        return self.finalize_output(output)

    @staticmethod
    def rel_cross_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_pos_enc: Tensor,
        kv_pos_pad: Tensor,
        q_pos_enc: Tensor,
        q_pos_pad: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
        rel_v: Tensor | None,
    ):
        """
        :param q: [batch_size, num_heads, seq_len, d_k]
        :param k: [batch_size, num_heads, seq_len, d_k]
        :param v: [batch_size, num_heads, seq_len, d_k]
        :param kv_pos_enc: [batch_size, num_heads, k+1, seq_len]
        :param kv_pos_pad: [batch_size, num_heads, k+1, seq_len]
        :param q_pos_enc: [batch_size, num_heads, k+1, seq_len]
        :param q_pos_pad: [batch_size, num_heads, k+1, seq_len]
        :param rel_q: [1, num_heads, 2k+2, d_k]
        :param rel_k: [1, num_heads, 2k+2, d_k]
        :param rel_v: [1, num_heads, 2k+2, d_k]
        :return:
        """
        # k and q dio not always have the same size
        batch_size, num_heads, kv_seq_len, d_k = k.size()
        _, _, q_seq_len, _ = q.size()
        scale_factor = 1

        q_weights = q.contiguous().view(-1, d_k)
        k_weights = k.contiguous().view(-1, d_k)
        v_weights = v.contiguous().view(-1, d_k)

        # Mask out the padding for kv
        kv_pos_enc_mask = kv_pos_enc.eq(-1)
        kv_pos_enc += kv_pos_enc_mask

        # We are allowed to attend to everything in the encoder but not future tokens
        # in the decoder
        # Mask matrix for future tokens is determined by the decoder side
        # (that always gives q)

        # Get boolean matrix to indicate where there is a -1 (inf) position encoding
        q_pos_enc_mask = q_pos_enc.eq(-1)
        q_pos_enc += q_pos_enc_mask
        # into shape [1, num_heads, 2k+2, d_k]
        mask_matrix = torch.cat([q_pos_enc_mask, q_pos_enc_mask], dim=-2)
        mask_matrix = mask_matrix.view(batch_size, num_heads, -1)

        # Mask out all tokens beyond the sequence length
        # There is nothing of value there
        mask_matrix[:, :, :q_seq_len] = True

        map_pos = torch.arange(batch_size * num_heads, dtype=torch.long, device=kv_pos_enc.device)
        kv_map_pos = map_pos.unsqueeze(-1) * kv_seq_len
        q_map_pos = map_pos.unsqueeze(-1) * q_seq_len

        q_indices = torch.cat([q_pos_pad, q_pos_enc], dim=-2)
        kv_indices = torch.cat([kv_pos_enc, kv_pos_pad], dim=-2)

        q_indices = q_indices.view(batch_size * num_heads, -1)
        kv_indices = kv_indices.view(batch_size * num_heads, -1)

        q_indices += q_map_pos
        kv_indices += kv_map_pos

        # query and key vec of context.
        # shape [batch_size * num_heads * (k+1) * 2, q_seq_len, d_k]
        q_context = torch.embedding(q_weights, q_indices)
        # shape [batch_size * num_heads * (k+1) * 2, kv_seq_len, d_k]
        k_context = torch.embedding(k_weights, kv_indices)
        v_context = torch.embedding(v_weights, kv_indices)

        q_context = q_context.view(batch_size, num_heads, -1, q_seq_len, d_k)
        k_context = k_context.view(batch_size, num_heads, -1, kv_seq_len, d_k)
        v_context = v_context.view(batch_size, num_heads, -1, kv_seq_len, d_k)

        # Attention calculation
        # context -> context
        c2c = torch.mul(q_context, k_context).sum(dim=-1)
        scores = c2c

        # context -> position
        if rel_k is not None:
            k_pos = rel_k.unsqueeze(-2)
            c2p = torch.mul(q_context, k_pos).sum(dim=-1)
            scores += c2p
            scale_factor += 1

        # position -> context
        if rel_q is not None:
            q_pos = rel_q.unsqueeze(-2)
            p2c = torch.mul(q_pos, k_context).sum(dim=-1)
            scores += p2c
            scale_factor += 1

        # Normalize attention
        # scores = (c2c + c2p + p2c) / (3 * sqrt(d_k))
        scores = scores / (scale_factor * math.sqrt(d_k))
        scores = scores.view(batch_size, num_heads, -1)

        # score shape [batch_size, num_heads, 2(k+1) * seq_len]
        # index shape [batch_size, num_heads, 2(k+1) * seq_len]
        q_indices = q_indices - q_map_pos
        q_indices = q_indices.view(batch_size, num_heads, -1)

        # Mask out tokens
        scores = scores.masked_fill(mask_matrix, -1e9)

        # Softmax attention
        scores = torch.exp(scores)

        score_sum = torch.zeros(scores.size(), device=scores.device)
        score_sum = score_sum.scatter_add(dim=-1, index=q_indices, src=scores)
        # score_sum.scatter_add_(dim=-1, index=q_indices, src=scores)

        score_sum += score_sum.eq(0) * -1e9
        score_sum = torch.gather(score_sum, index=q_indices, dim=-1)

        # [batch_size, num_heads, 2(k+1), q_seq_len, 1]
        scores = (scores / score_sum).view(batch_size, num_heads, -1, kv_seq_len, 1)

        # shape [batch_size, num_heads, 2(k+1), kv_seq_len, d_k]
        if rel_v is not None:
            # Scale v by position
            v_context += rel_v.unsqueeze(-2)

        # Multiply attention by value vector
        attn_v = torch.mul(scores, v_context)

        output = torch.zeros(attn_v.size(), device=attn_v.device)
        q_indices = q_indices.view(batch_size, num_heads, -1, q_seq_len).unsqueeze(-1)
        q_indices = q_indices.repeat_interleave(repeats=d_k, dim=-1)
        output = output.scatter_add(dim=-2, index=q_indices, src=attn_v)

        output = output.sum(dim=-3)

        return output


def transpose_for_scores(x: Tensor, num_heads: int):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)
