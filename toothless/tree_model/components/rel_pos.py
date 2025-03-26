import torch
import torch.nn as nn
from torch import Tensor

from toothless.tree_model.components.utils import concat_vec


class RelCoder(nn.Module):
    def __init__(
        self,
        anc_heads: int,
        sib_heads: int,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(RelCoder, self).__init__()
        self.anc_heads = anc_heads
        self.sib_heads = sib_heads
        d_k = d_model // (anc_heads + sib_heads)

        if anc_heads > 0:
            self.anc_rel_emb = RelEmbeddings(
                d_k, anc_heads, max_rel_pos, pos_type, dropout=dropout, device=device, dtype=dtype
            )
        if sib_heads > 0:
            self.sib_rel_emb = RelEmbeddings(
                d_k, sib_heads, max_rel_pos, pos_type, dropout=dropout, device=device, dtype=dtype
            )

    def rel_pos_emb(self) -> tuple[Tensor | None, Tensor | None]:
        rel_anc_q, rel_anc_k = None, None
        rel_sib_q, rel_sib_k = None, None
        if self.anc_heads > 0:
            rel_anc_q, rel_anc_k = self.anc_rel_emb()
        if self.sib_heads > 0:
            rel_sib_q, rel_sib_k = self.sib_rel_emb()

        rel_q = concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = concat_vec(rel_anc_k, rel_sib_k, dim=1)

        return rel_q, rel_k  # , rel_v

    def concat_pos(self, rel_anc_pos: Tensor, rel_sib_pos: Tensor) -> Tensor:
        if self.anc_heads == 0:
            return rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        if self.sib_heads == 0:
            return rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)

        rel_anc_pos = rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)
        rel_sib_pos = rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        rel_pos = torch.cat([rel_anc_pos, rel_sib_pos], dim=1)
        return rel_pos


class RelEmbeddings(nn.Module):
    def __init__(
        self,
        d_k: int,
        num_heads: int,
        k: int,
        pos_type: list[str],
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(RelEmbeddings, self).__init__()

        self.d_k = d_k
        self.k = 2 * k + 2
        self.pos_type = pos_type
        self.num_heads = num_heads
        if "p2q" in pos_type:
            self.rel_emb_q = nn.Embedding(self.k, d_k, padding_idx=0, device=device, dtype=dtype)  # pad inf -> zero
        if "p2k" in pos_type:
            self.rel_emb_k = nn.Embedding(self.k, d_k, padding_idx=0, device=device, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

    def get_rel_weights(self, rel_params: Tensor) -> Tensor:
        # rel_params = rel_params * math.sqrt(self.d_model)
        rel_params = self.dropout(rel_params)
        rel_params = rel_params.unsqueeze(0).unsqueeze(0)
        rel_params = rel_params.repeat(1, self.num_heads, 1, 1)

        return rel_params

    def forward(
        self,
    ) -> tuple[Tensor | None, Tensor | None]:
        rel_q, rel_k = None, None
        if "p2q" in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if "p2k" in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        # if "p2v" in self.pos_type:
        #     rel_v = self.get_rel_weights(self.rel_emb_v.weight)

        return rel_q, rel_k

    # def get_p2v_emb(self, inputs: Tensor) -> Tensor | None:
    #     if "p2v" in self.pos_type:
    #         rel_v = self.rel_emb_v(inputs) * math.sqrt(self.d_k)
    #         rel_v = self.dropout(rel_v)
    #         rel_v = rel_v.repeat(1, 1, 1, self.num_heads)
    #         return rel_v
    #     else:
    #         return None


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
