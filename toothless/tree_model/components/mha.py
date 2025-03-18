import math

import torch
import torch.nn as nn
from torch import Tensor


class FastMultiHeadedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        cross_attn: bool = False,
        device=None,
        dtype=None,
    ):
        super(FastMultiHeadedAttention, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_k = d_model // num_heads
        self.cross_attn = cross_attn
        self.num_heads = num_heads

        self.dropout = dropout

        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False, *factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, *factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, *factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, *factory_kwargs)

    def finalize_output(self, output: Tensor) -> Tensor:
        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.out_proj(output)

        return output


class MHSelfAttn(FastMultiHeadedAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super(MHSelfAttn, self).__init__(d_model, num_heads, dropout, False, device, dtype)

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

        query = transpose_for_scores(self.q_proj(query), self.num_heads)
        key = transpose_for_scores(self.k_proj(key), self.num_heads)
        value = transpose_for_scores(self.v_proj(value), self.num_heads)

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
        :param rel_q: [1, num_heads, 2(k+1), d_k]
        :param rel_k: [1, num_heads, 2(k+1), d_k]
        :param rel_v: [1, num_heads, 2(k+1), d_k]
        :return:
        """
        batch_size, num_heads, seq_len, d_k = q.size()
        # max_rel_pos = pos_enc.size(2)
        scale_factor = 1

        # Flatten along model dimension
        # Shape: [batch_size * num_heads * seq_len, d_k]
        q_weights = q.contiguous().view(-1, d_k)
        k_weights = k.contiguous().view(-1, d_k)
        v_weights = v.contiguous().view(-1, d_k)

        # Get positions where pos_enc is -1 aka inf
        pos_enc_mask = pos_enc.eq(-1)

        # Since we will mask out those positions later, it does not
        # matter what their value is. For possible computationally efficiency reasons
        # (sparse) we set those to 0 but remember the mask
        # Shape: [batch_size, num_heads, k+1, seq_len]
        pos_enc += pos_enc_mask
        # Concat along k+1 dim => 2(k+1).
        # Shape: [batch_size, num_heads, 2(k+1), seq_len]
        mask_matrix = torch.cat([pos_enc_mask, pos_enc_mask], dim=-2)
        # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        mask_matrix = mask_matrix.view(batch_size, num_heads, -1)

        # Additionally mask out everything that is beyond the sequence
        mask_matrix[:, :, :seq_len] = True

        # Generate the offsets into the q_weigths
        # Shape: [batch_size * num_heads]
        map_pos_offsets = torch.arange(batch_size * num_heads, dtype=torch.long, device=pos_enc.device)
        # Unsequeeze along seq_len -> [[0],[1],[2],..[batch_size * num_heads]],
        # then scale to seq len -> [[0],[1 * seq_len],[2 * seq_len],..[batch_size * num_heads * seq_len]]
        # Shape: [batch_size * num_heads, 1]
        map_pos_offsets = map_pos_offsets.unsqueeze(-1) * seq_len

        # Concat along k+1 dimension
        #
        # Shape: [batch_size, num_heads, 2(k+1), seq_len]
        query_indexes = torch.cat([pos_pad, pos_enc], dim=-2)
        kv_indexes = torch.cat([pos_enc, pos_pad], dim=-2)

        # Flatten along the batch_size * num_heads
        # Shape: [batch_size * num_heads, 2(k+1) * seq_len]
        query_indexes = query_indexes.view(batch_size * num_heads, -1)
        kv_indexes = kv_indexes.view(batch_size * num_heads, -1)

        # Add the offsets to the indices
        # Since map_pos_offsets is of shape [batch_size * num_heads, 1],
        # the value gets added to every 2(k+1) * seq_len
        query_indexes += map_pos_offsets
        kv_indexes += map_pos_offsets

        # query and key vec of context.
        # Shape: [batch_size * num_heads * 2(k+1), seq_len, d_k]
        q_context = torch.embedding(q_weights, query_indexes)
        k_context = torch.embedding(k_weights, kv_indexes)
        v_context = torch.embedding(v_weights, kv_indexes)

        # Shape: [batch_size, num_heads, 2(k+1), seq_len, d_k]
        q_context = q_context.view(batch_size, num_heads, -1, seq_len, d_k)
        k_context = k_context.view(batch_size, num_heads, -1, seq_len, d_k)
        v_context = v_context.view(batch_size, num_heads, -1, seq_len, d_k)

        # Attention Calculation.
        # Always sum over dimension of heads d_k
        # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        # context -> context
        c2c = torch.mul(q_context, k_context).sum(dim=-1)
        scores = c2c

        # context -> position
        if rel_k is not None:
            # Shape: [1, num_heads, 2(k+1), 1, d_k]
            k_pos = rel_k.unsqueeze(-2)
            c2p = torch.mul(q_context, k_pos).sum(dim=-1)
            scores += c2p
            scale_factor += 1

        # position -> context
        if rel_q is not None:
            # Shape: [1, num_heads, 2(k+1) 1, d_k]
            q_pos = rel_q.unsqueeze(-2)
            p2c = torch.mul(q_pos, k_context).sum(dim=-1)
            scores += p2c
            scale_factor += 1

        # scores = (c2c + c2p + p2c) / (3 * sqrt(d_k))
        scores = scores / (scale_factor * math.sqrt(d_k))
        # Flatten seq_len and 2(k+1) away
        # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        scores = scores.view(batch_size, num_heads, -1)

        # Remove the offsets in map_pos and reshape
        # Shape: [batch_size * num_heads, 2(k+1) * seq_len]
        query_indexes = query_indexes - map_pos_offsets
        # Re-Introduce the batch_size, num_heads dimensions to align with scores
        # Shape index: [batch_size, num_heads, 2(k+1) * seq_len]
        query_indexes = query_indexes.view(batch_size, num_heads, -1)

        # Mask out attention scores according to mask with very small values
        # Shape index: [batch_size, num_heads, 2(k+1) * seq_len]
        scores = scores.masked_fill(mask_matrix, -1e9)

        # # Exp Part of Softmax Calculation
        # scores = torch.exp(scores)

        # # Sum Part of Softmax Calculation
        # # First scatter_add all to all along the 2(k+1) * seq_len dimension
        # # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        # score_sum = torch.zeros(scores.size(), device=scores.device)
        # score_sum.scatter_add_(dim=-1, index=query_indexes, src=scores)
        # # 0 attention become very small attention values
        # score_sum += score_sum.eq(0) * -1e9
        # # Gather along 2(k+1) + seq_len
        # # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        # score_sum = torch.gather(score_sum, index=query_indexes, dim=-1)

        # # Div Part of Softmax Calculation
        # # Shape: [batch_size, num_heads, 2(k+1), seq_len, 1]
        # scores = (scores / score_sum).view(batch_size, num_heads, -1, seq_len, 1)

        # Softmax and add broadcast shape at 1
        # Shape: [batch_size, num_heads, 2(k+1), seq_len, 1]
        scores = scores.softmax(dim=-1).view(batch_size, num_heads, -1, seq_len, 1)

        # Add v_context to v before multiplication with attention
        if rel_v is not None:
            # Shape: [1, num_heads, 2(k+1), 1, d_k]
            v_context += rel_v.unsqueeze(-2)
        # Shape: [batch_size, num_heads, 2(k+1), seq_len, d_k]
        attn_v = torch.mul(scores, v_context)

        # Shape: [batch_size, num_heads, 2(k+1) * seq_len, 1]
        query_indexes = query_indexes.view(batch_size, num_heads, -1, seq_len).unsqueeze(-1)
        # Shape: [batch_size, num_heads, 2(k+1) * seq_len, d_k]
        query_indexes = query_indexes.repeat_interleave(repeats=d_k, dim=-1)

        # Shape: [batch_size, num_heads, 2(k+1), seq_len, d_k]
        output = torch.zeros(attn_v.size(), device=attn_v.device)
        output.scatter_add_(dim=-2, index=query_indexes, src=attn_v)

        # Sum over all (and all types of) values
        # Shape: [batch_size, num_heads, seq_len, d_k]
        output = output.sum(dim=-3)

        return output


class MHCrossAttn(FastMultiHeadedAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super(MHCrossAttn, self).__init__(d_model, num_heads, dropout, True, device, dtype)

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
        """relative q shape [1, 2(k+1), dim]"""
        """relative v shape [1, 2(k+1), dim]"""
        """"""
        query = transpose_for_scores(self.q_proj(query), self.num_heads)
        key = transpose_for_scores(self.k_proj(key), self.num_heads)
        value = transpose_for_scores(self.v_proj(value), self.num_heads)

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
        :param rel_q: [1, num_heads, 2(k+1), d_k]
        :param rel_k: [1, num_heads, 2(k+1), d_k]
        :param rel_v: [1, num_heads, 2(k+1), d_k]
        :return:
        """
        # k and q dio not always have the same size
        batch_size, num_heads, kv_seq_len, d_k = k.size()
        _, _, q_seq_len, _ = q.size()
        # TODO REST

        return Tensor()


def transpose_for_scores(x: Tensor, num_heads: int):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)
