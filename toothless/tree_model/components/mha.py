import math

import torch
import torch.nn as nn
from torch import Tensor


class FastMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        cross_attn: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(FastMHA, self).__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.d_k = d_model // num_heads
        self.cross_attn = cross_attn
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)

        self.out_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)

    def forward(
        self,
        hidden_state: Tensor,
        pos_indices: Tensor,
        mask: Tensor,
        query_states: Tensor | None = None,
        query_pos_indices: Tensor | None = None,
        rel_q: Tensor | None = None,
        rel_k: Tensor | None = None,
        rel_v: Tensor | None = None,
    ) -> Tensor:
        """relative q shape [1, 2k+1, dim]"""
        """relative v shape [1, 2k+1, dim]"""
        """"""

        if not query_states:
            query_states = hidden_state
        if not query_pos_indices:
            query_pos_indices = pos_indices

        query = transpose_for_scores(self.q_proj(query_states), self.num_heads)
        key = transpose_for_scores(self.k_proj(hidden_state), self.num_heads)
        value = transpose_for_scores(self.v_proj(hidden_state), self.num_heads)

        mask = self.make_mask(mask, pos_indices)

        output, _ = self.rel_attn(query, key, value, query_pos_indices, pos_indices, rel_q, rel_k, rel_v, mask)

        return self.finalize_output(output)

    def make_mask(self, mask: Tensor, pos_indices: Tensor) -> Tensor:
        # Get positions where pos_enc is 0 aka inf
        mask_matrix = pos_indices.eq(0)

        batch_size = pos_indices.size(0)
        seq_len = pos_indices.size(-1)

        # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        mask_matrix = mask_matrix.view(batch_size, self.num_heads, -1)

        # Additionally mask out everything that is beyond the sequence
        mask_matrix[:, :, :seq_len] = True

        # Apply triangular matrix
        if self.cross_attn:
            mask_matrix[:,] = triangle_matrix(seq_len, pos_indices.device)

        return mask_matrix

    def rel_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_pos_indices: Tensor,
        kv_pos_indices: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
        rel_v: Tensor | None,
        mask: Tensor,
    ):
        """
        :param q: [batch_size, num_heads, seq_len, d_k]
        :param k: [batch_size, num_heads, seq_len, d_k]
        :param v: [batch_size, num_heads, seq_len, d_k]
        :param q_c2p_pos: [batch_size, num_heads, k+1, seq_len]
        :param kv_c2p_pos: [batch_size, num_heads, k+1, seq_len]
        :param rel_q: [1, num_heads, k+1, d_k]
        :param rel_k: [1, num_heads, k+1, d_k]
        :param rel_v: [1, num_heads, k+1, d_k]
        :return:
        """
        seq_len = query.size(-2)
        print(query.size())
        print(seq_len)
        print(self.d_k)
        # query = query.view(-1, seq_len, self.d_k)

        # max_rel_pos = pos_enc.size(2)
        scale_factor = 1
        if rel_k is not None:
            scale_factor += 1
        if rel_q is not None:
            scale_factor += 1

        # Attention Calculation.
        # Always sum over dimension of heads d_k
        # Shape: [batch_size, num_heads, 2(k+1) * seq_len]
        # context -> context
        scale = 1 / math.sqrt(query.size(-1) * scale_factor)
        c2c = torch.matmul(query, key.transpose(-1, -2) * scale)
        attn_scores = c2c

        # All relative attention mechanisms follow one scheme:
        # 1) Multiply out all possible n to m combinations of
        #    context and relative position
        # 2) Then pick the ones we are actually interested in as indicated
        #    by the indices
        # Actually efficient cause there arent that many relative positional
        # encodings (32 or so) and we can reuse the vectors
        # (presumably a lot will have the same distance)

        # position -> context
        if rel_q is not None:
            scale = 1 / math.sqrt(rel_q.size(-1) * scale_factor)
            pos_q = rel_q.unsqueeze(-2)
            p2c_att = torch.matmul(pos_q * scale, key.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-2, index=q_pos_indices)
            attn_scores += p2c_att

        # context -> position
        if rel_k is not None:
            scale = 1 / math.sqrt(rel_k.size(-1) * scale_factor)
            pos_k = rel_k.unsqueeze(-2)
            c2p_att = torch.matmul(query, pos_k.transpose(-1, -2) * scale)
            c2p_att = torch.gather(c2p_att, dim=-1, index=kv_pos_indices)
            attn_scores += c2p_att

        # attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values.detach()
        # attn_scores = attn_scores.view(-1, self.num_heads, attn_scores.size(-2), attn_scores.size(-1))

        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_scores = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_scores)
        # context = torch.matmul(attn_probs.view(-1, attn_probs.size(-2), self.d_k), value)
        context = torch.matmul(attn_probs, value)

        if rel_v is not None:
            pos_v = rel_v.unsqueeze(-2)
            # positional_context = torch.bmm(attn_probs.view(-1, attn_probs.size(-2), self.d_k), pos_v)
            positional_context = torch.matmul(attn_probs, pos_v)
            positional_context = torch.gather(positional_context, dim=-1, index=kv_pos_indices)
            context += positional_context

        return context, attn_probs

    def finalize_output(self, output: Tensor) -> Tensor:
        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.out_proj(output)

        return output


def triangle_matrix(
    sz: int,
    device: torch.device | None = None,
) -> Tensor:
    m = torch.full((sz, sz), True, device=device)
    return torch.triu(m, diagonal=False)


def transpose_for_scores(x: Tensor, num_heads: int):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # .contiguous().view(-1, x.size(1), x.size(-1))
