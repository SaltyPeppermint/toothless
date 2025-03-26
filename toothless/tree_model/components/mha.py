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
    ):
        super(FastMHA, self).__init__()

        self.d_k = d_model // num_heads
        self.cross_attn = cross_attn
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        hidden_state: Tensor,
        pos_indices: Tensor,
        mask: Tensor,
        cross_state: Tensor | None = None,
        cross_pos_indices: Tensor | None = None,
        rel_q: Tensor | None = None,
        rel_k: Tensor | None = None,
    ) -> Tensor:
        """
        :param hidden_state:        [batch_size, seq_len, d_model]
        :param pos_indices:         [batch_size, num_heads, seq_len, seq_len]
        :param mask:                [batch_size, 1, 1 (or seql_len if cross_attention), seq_len]
        :param cross_state:         [batch_size, seq_len, d_model]  | None
        :param cross_pos_indices:   [batch_size, 2k+2, seq_len]     | None
        :param rel_v:               [k+1, d_k]                      | None
        :param rel_v:               [k+1, d_k]                      | None
        :return                     [batch_size, seq_len, d_model]
        """
        if cross_state is None:
            cross_state = hidden_state
        if cross_pos_indices is None:
            cross_pos_indices = pos_indices

        batch_size = hidden_state.size(0)
        # batch_size, num_heads, seq_len, d_proj
        query = self.q_proj(hidden_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k_proj(cross_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v_proj(cross_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output = self.rel_attn(query, key, value, cross_pos_indices, pos_indices, rel_q, rel_k, mask)
        return self.finalize_output(output)  # batch_size, max_seq_len, d_model

    def rel_attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_pos_indices: Tensor,
        kv_pos_indices: Tensor,
        rel_q: Tensor | None,
        rel_k: Tensor | None,
        mask: Tensor,
    ) -> Tensor:
        """
        :param query:           [batch_size, num_heads, seq_len, d_k]
        :param key:             [batch_size, num_heads, seq_len, d_k]
        :param value:           [batch_size, num_heads, seq_len, d_k]
        :param q_pos_indices:   [batch_size, num_heads, seq_len, seq_len]
        :param kv_pos_indices:  [batch_size, num_heads, seq_len, seq_len]
        :param rel_q:           [1, num_heads, 2k+2, d_k] | None
        :param rel_k:           [1, num_heads, 2k+2, d_k] | None
        :param mask:            [batch_size, 1, 1 (or seql_len if cross_attention), seq_len]
        :return:                [batch_size, num_heads, seq_len, d_k]
        """
        scale_factor = 1
        if rel_k is not None:
            scale_factor += 1
        if rel_q is not None:
            scale_factor += 1

        # Attention Calculation.
        # Always sum over dimension of heads d_k
        # context -> context
        scale = 1 / math.sqrt(self.d_k * scale_factor)
        c2c = torch.matmul(query, key.transpose(-1, -2) * scale)
        attn_scores = c2c  # [batch_size, num_heads, seq_len, seq_len]

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
            scale = 1 / math.sqrt(self.d_k * scale_factor)
            p2c_attn = torch.matmul(rel_q * scale, key.transpose(-1, -2))
            p2c_attn = torch.gather(p2c_attn, dim=-2, index=q_pos_indices)
            attn_scores += p2c_attn

        # context -> position
        if rel_k is not None:
            scale = 1 / math.sqrt(self.d_k * scale_factor)
            c2p_attn = torch.matmul(query, rel_k.transpose(-1, -2) * scale)
            # (
            #     torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            #     .squeeze(0)
            #     .expand([query.size(0), query.size(1), query.size(-1)])
            # )
            c2p_attn = torch.gather(c2p_attn, dim=-1, index=kv_pos_indices.transpose(-2, -1))
            attn_scores += c2p_attn

        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_scores = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_scores)

        context = torch.matmul(attn_probs, value)

        return context  # , attn_probs

    def finalize_output(self, output: Tensor) -> Tensor:
        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.out_proj(output)

        return output


def transpose_for_scores(x: Tensor, num_heads: int):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # .contiguous().view(-1, x.size(1), x.size(-1))
