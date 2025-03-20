import torch
import torch.nn as nn
from torch import Tensor

from toothless.tree_model.components.utils import concat_vec
from toothless.tree_model.embeddings import FastRelEmbeddings


class RelCoder(nn.Module):
    def __init__(
        self,
        n_anc_heads: int,
        n_sib_heads: int,
        pos_type: list[str],
        max_rel_pos: int,
        d_model: int,
        dropout: float = 0.2,
    ):
        super(RelCoder, self).__init__()

        self.n_anc_heads = n_anc_heads
        self.n_sib_heads = n_sib_heads
        d_k = d_model // (n_anc_heads + n_sib_heads)

        if n_anc_heads > 0:
            self.anc_rel_emb = FastRelEmbeddings(d_k, n_anc_heads, max_rel_pos, pos_type, dropout=dropout)
        if n_sib_heads > 0:
            self.sib_rel_emb = FastRelEmbeddings(d_k, n_sib_heads, max_rel_pos, pos_type, dropout=dropout)

        self.pos_enc_padding = None

    def rel_pos_emb(self):
        rel_anc_q, rel_anc_k, rel_anc_v = None, None, None
        rel_sib_q, rel_sib_k, rel_sib_v = None, None, None
        if self.n_anc_heads > 0:
            rel_anc_q, rel_anc_k, rel_anc_v = self.anc_rel_emb()
        if self.n_sib_heads > 0:
            rel_sib_q, rel_sib_k, rel_sib_v = self.sib_rel_emb()

        rel_q = concat_vec(rel_anc_q, rel_sib_q, dim=1)
        rel_k = concat_vec(rel_anc_k, rel_sib_k, dim=1)
        rel_v = concat_vec(rel_anc_v, rel_sib_v, dim=1)
        return rel_q, rel_k, rel_v

    def concat_pos(self, rel_anc_pos, rel_sib_pos) -> Tensor:
        if self.anc_heads == 0:
            return rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        if self.sib_heads == 0:
            return rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)

        rel_anc_pos = rel_anc_pos.unsqueeze(1).repeat_interleave(repeats=self.anc_heads, dim=1)
        rel_sib_pos = rel_sib_pos.unsqueeze(1).repeat_interleave(repeats=self.sib_heads, dim=1)
        rel_pos = torch.cat([rel_anc_pos, rel_sib_pos], dim=1)

        return rel_pos
