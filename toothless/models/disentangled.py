from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

from eggshell import rise  # type: ignore


from .layers.utils import Embedding
from .layers.disentangled.decoder import DisASTDoubleDecoder
from .layers.disentangled.encoder import DisASTEncoder
from .utils import make_tgt_mask
from ..args import ModelArguments
from ..data import partial_to_matrices, split_off_special
from ..vocab import SimpleVocab


class DisentangledDualTreeTransformer(nn.Module):
    def __init__(self, conf: ModelArguments, src_vocab_size: int, tgt_vocab_size: int, k: int, state_dict=None):
        super(DisentangledDualTreeTransformer, self).__init__()

        assert conf.disentangled
        assert conf.n_heads == conf.anc_heads + conf.sib_heads

        self.l_embedding = Embedding(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)
        self.r_embedding = Embedding(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)
        self.tgt_embedding = Embedding(conf.d_model, tgt_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)

        self.l_encoder = DisASTEncoder(conf, k)
        self.r_encoder = DisASTEncoder(conf, k)
        self.decoder = DisASTDoubleDecoder(conf, k)

        self.output_proj = nn.Linear(conf.d_model, tgt_vocab_size)

        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    def forward(self, data: dict[str, Tensor]):
        l_mem = self.l_encode(data)
        r_mem = self.r_encode(data)

        decoder_outputs = self.decode(data, l_mem, r_mem)
        return self.output_proj(decoder_outputs)

    def l_encode(self, data: dict[str, Tensor]) -> Tensor:
        l_emb = self.l_embedding(data["l_ids"])
        return self.l_encoder(l_emb, data["l_anc"], data["l_sib"], data["l_mask"])

    def r_encode(self, data: dict[str, Tensor]) -> Tensor:
        r_emb = self.r_embedding(data["r_ids"])
        return self.r_encoder(r_emb, data["r_anc"], data["r_sib"], data["r_mask"])

    def decode(self, data: dict[str, Tensor], l_mem: Tensor, r_mem: Tensor) -> Tensor:
        tgt = self.tgt_embedding(data["tgt_ids"])
        return self.decoder(
            tgt,
            data["tgt_anc"],
            data["tgt_sib"],
            data["tgt_mask"],
            l_mem,
            data["l_anc"],
            data["l_sib"],
            data["l_mask"],
            r_mem,
            data["r_anc"],
            data["r_sib"],
            data["r_mask"],
        )


class DisentangledGreedyGenerator(nn.Module):
    # smth about multi gpu and model.module?
    def __init__(self, model: DisentangledDualTreeTransformer, max_len: int, vocab: SimpleVocab, k: int):
        super(DisentangledGreedyGenerator, self).__init__()

        self.model = model
        self.max_len = max_len
        self.vocab = vocab
        self.k = k

    def model_device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]):
        l_mem = self.model.l_encode(batch)
        r_mem = self.model.r_encode(batch)

        batch_size = r_mem.size(0)
        device = self.model_device()

        assert self.vocab.pad_token_id == 0
        batch["tgt_ids"] = torch.zeros(batch_size, self.max_len, device=device, dtype=torch.long)
        batch["tgt_ids"][:, 0] = self.vocab.bos_token_id
        batch["tgt_probs"] = torch.zeros(batch_size, self.max_len, device=device, dtype=torch.float)
        finished_flags = torch.full((batch_size,), False).to(device)

        for i in range(self.max_len - 1):
            batch["tgt_mask"] = make_tgt_mask(batch["tgt_ids"], 0)
            batch["tgt_anc"], batch["tgt_sib"] = self.pos_matrices(batch["tgt_ids"])

            batch = {k: v.to(device) for k, v in batch.items()}
            decoder_outputs = self.model.decode(batch, l_mem, r_mem)
            out = self.model.output_proj(decoder_outputs)
            fresh_out = out[:, i, :].squeeze(1)

            probs, next_token = torch.max(F.log_softmax(fresh_out, dim=-1))

            batch["tgt_ids"][:, i + 1] = next_token
            batch["tgt_probs"][:, i + 1] = torch.exp(probs)

            finished_flags = finished_flags | next_token == self.vocab.eos_token_id
            if torch.all(finished_flags):
                break

        return batch["tgt_ids"], batch["tgt_probs"]

    def pos_matrices(self, tgt_ids: Tensor) -> tuple[Tensor, Tensor]:
        batch_tgt_anc, batch_tgt_sib = [], []
        for partial_ids in tgt_ids.tolist():
            padded_tgt_anc = torch.zeros((self.max_len, self.max_len), device=self.model_device(), dtype=torch.long)
            padded_tgt_sib = torch.zeros((self.max_len, self.max_len), device=self.model_device(), dtype=torch.long)

            partial_tok = split_off_special([self.vocab.id2token(i) for i in partial_ids], self.vocab)

            # If nothing generated we cant pad anything
            if 0 < len(partial_tok) <= rise.GeneratedRecExpr.count_expected_tokens(partial_tok):
                tgt_anc, tgt_sib = partial_to_matrices(partial_tok, self.k)
                # Initialize distance matrix with all zeroes meaning no adjacency
                # Then, for those tokens already generated, add the adjavencies to the matrix
                # This leaves a matrix of the shape where the 4th and 5th column are zero since they're unknown
                # 0 1 2 0 0
                # 0 0 0 0 0
                # 0 3 4 0 0
                # 0 0 0 0 0
                # 0 0 0 0 0
                padded_tgt_anc[: tgt_anc.size(0), : tgt_anc.size(1)] = tgt_anc
                padded_tgt_sib[: tgt_sib.size(0), : tgt_sib.size(1)] = tgt_sib
                # Extra padding since tgt will be shifted
            batch_tgt_anc.append(padded_tgt_anc)
            batch_tgt_sib.append(padded_tgt_sib)

        return torch.stack(batch_tgt_anc), torch.stack(batch_tgt_sib)
