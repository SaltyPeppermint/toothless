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

        self.d_proj = d_model // num_heads
        self.cross_attn = cross_attn
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)
        self.k_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)
        self.v_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)

        self.out_proj = nn.Linear(d_model, d_model, bias=True, **self.factory_kwargs)

    def forward(
        self,
        hidden_state: Tensor,
        pos_indices: Tensor,
        mask: Tensor,
        cross_state: Tensor | None = None,
        cross_pos_indices: Tensor | None = None,
        rel_q: Tensor | None = None,
        rel_k: Tensor | None = None,
        rel_v: Tensor | None = None,
    ) -> Tensor:
        """
        :param hidden_state:    [batch_size, seq_len, d_model]
        :param pos_indices:     [batch_size, k+1, seq_len]
        :param mask:            [batch_size, 1, 1 (or seql_len if cross_attention), seq_len]
        :param cross_state:
        :param cross_pos_indices:
        :param rel_v:           [k+1, d_proj] | None
        :param rel_v:           [k+1, d_proj] | None
        :param rel_v:           [k+1, d_proj] | None
        :return                 [batch_size, seq_len, d_model]
        """
        if cross_state is None:
            cross_state = hidden_state
        if cross_pos_indices is None:
            cross_pos_indices = pos_indices

        batch_size = hidden_state.size(0)
        # batch_size, num_heads, seq_len, d_proj
        query = self.q_proj(hidden_state).view(batch_size, -1, self.num_heads, self.d_proj).transpose(1, 2)
        key = self.k_proj(cross_state).view(batch_size, -1, self.num_heads, self.d_proj).transpose(1, 2)
        value = self.v_proj(cross_state).view(batch_size, -1, self.num_heads, self.d_proj).transpose(1, 2)

        # mask = self.make_mask(mask, pos_indices)

        output = self.rel_attn(query, key, value, cross_pos_indices, pos_indices, rel_q, rel_k, rel_v, mask)
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
        rel_v: Tensor | None,
        mask: Tensor,
    ) -> Tensor:
        """
        :param q:           [batch_size, num_heads, seq_len, d_proj]
        :param k:           [batch_size, num_heads, seq_len, d_proj]
        :param v:           [batch_size, num_heads, seq_len, d_proj]
        :param q_c2p_pos:   [batch_size, num_heads, k+1, seq_len]
        :param kv_c2p_pos:  [batch_size, num_heads, k+1, seq_len]
        :param rel_q:       [1, num_heads, k+1, d_proj]
        :param rel_k:       [1, num_heads, k+1, d_proj]
        :param rel_v:       [1, num_heads, k+1, d_proj]
        :param mask:        [batch_size, 1, 1 (or seql_len if cross_attention), seq_len]
        :return:            [batch_size, num_heads, seq_len, d_proj]
        """
        # max_rel_pos = pos_enc.size(2)
        scale_factor = 1
        if rel_k is not None:
            scale_factor += 1
        if rel_q is not None:
            scale_factor += 1

        # Attention Calculation.
        # Always sum over dimension of heads d_k
        # context -> context
        scale = 1 / math.sqrt(query.size(-1) * scale_factor)
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
            scale = 1 / math.sqrt(rel_q.size(-1) * scale_factor)
            print(f"rel_q shape: {rel_q.size()}")
            pos_q = rel_q.unsqueeze(-2)
            print(f"pos_q shape: {pos_q.size()}")
            p2c_attn = torch.matmul(pos_q * scale, key.transpose(-1, -2))
            print(f"p2c_att shape: {p2c_attn.size()}")
            p2c_attn = torch.gather(p2c_attn, dim=-2, index=q_pos_indices)
            attn_scores += p2c_attn

        # context -> position
        if rel_k is not None:
            scale = 1 / math.sqrt(rel_k.size(-1) * scale_factor)
            print(f"pos_k shape: {rel_k.size()}")
            pos_k = rel_k.unsqueeze(-2)
            print(f"pos_k shape: {pos_k.size()}")
            c2p_attn = torch.matmul(query, pos_k.transpose(-1, -2) * scale)
            # (
            #     torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            #     .squeeze(0)
            #     .expand([query.size(0), query.size(1), query.size(-1)])
            # )
            print(f"c2p_att shape: {c2p_attn.size()}")
            c2p_attn = torch.gather(c2p_attn, dim=-1, index=kv_pos_indices.transpose(-2, -1))
            attn_scores += c2p_attn

        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_scores = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_scores)

        context = torch.matmul(attn_probs, value)

        if rel_v is not None:
            pos_v = rel_v.unsqueeze(-2)
            positional_context = torch.matmul(attn_probs, pos_v)
            positional_context = torch.gather(positional_context, dim=-1, index=kv_pos_indices)
            context += positional_context

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
