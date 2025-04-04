from torch import Tensor
from torch import nn
import torch

from toothless.tree_model.args import ModelArguments
from toothless.tree_model.components.decoder import ASTDoubleDecoder
from toothless.tree_model.components.encoder import ASTEncoder
from toothless.tree_model.components.utils import Embeddings, Generator
from toothless.tree_model.data import make_std_mask, partial_to_matrices
from toothless.tree_model.vocab import SimpleVocab


class ASTTransformer(nn.Module):
    def __init__(self, conf: ModelArguments, src_vocab_size: int, tgt_vocab_size: int, k: int, state_dict=None):
        super(ASTTransformer, self).__init__()

        self.with_anc_pos = conf.anc_heads > 0
        self.with_sib_pos = conf.sib_heads > 0

        self.l_embedding = Embeddings(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)
        self.r_embedding = Embeddings(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)
        self.tgt_embedding = Embeddings(conf.d_model, tgt_vocab_size, dropout=conf.dropout, with_pos=conf.with_pos)

        self.l_encoder = ASTEncoder(conf, k)
        self.r_encoder = ASTEncoder(conf, k)
        self.decoder = ASTDoubleDecoder(conf, k)

        self.generator = Generator(conf, tgt_vocab_size)

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
        output = self.generator(decoder_outputs)
        return output

    def l_encode(self, data: dict[str, Tensor]) -> Tensor:
        l_emb = self.l_embedding(data["l_ids"])
        l_mem = self.l_encoder(l_emb, data["l_anc"], data["l_sib"], data["l_mask"])
        return l_mem

    def r_encode(self, data: dict[str, Tensor]) -> Tensor:
        r_emb = self.r_embedding(data["r_ids"])
        r_mem = self.r_encoder(r_emb, data["r_anc"], data["r_sib"], data["r_mask"])
        return r_mem

    def decode(self, data: dict[str, Tensor], l_mem: Tensor, r_mem: Tensor) -> Tensor:
        tgt = self.tgt_embedding(data["tgt_ids"])

        outputs = self.decoder(
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
        outputs = outputs.permute(1, 0, 2)
        return outputs


class GreedyGenerator(nn.Module):
    # smth about multi gpu and model.module?
    def __init__(self, model: ASTTransformer, max_len: int, vocab: SimpleVocab, k: int):
        super(GreedyGenerator, self).__init__()

        self.model = model
        self.max_len = max_len
        self.vocab = vocab
        self.k = k

    def forward(self, data: dict[str, Tensor]):
        l_mem = self.model.l_encode(data)
        r_mem = self.model.r_encode(data)

        batch_size = r_mem.size(0)

        assert self.vocab.pad_token_id == 0
        tgt_ids = torch.zeros(batch_size, 1, requires_grad=False, dtype=torch.long)
        tgt_ids[:, 0] = self.vocab.bos_token_id

        data["tgt_ids"] = tgt_ids.to(l_mem.device)

        for i in range(self.max_len - 1):
            data["tgt_mask"] = make_std_mask(data["tgt_ids"], 0)
            data["tgt_anc"], data["tgt_sib"] = self.pos_matrices(data["tgt_ids"])

            decoder_outputs, _ = self.model.decode(data, l_mem, r_mem)
            out = self.model.generator(decoder_outputs)
            out = out[:, i, :].squeeze(1)

            _prob, next_token = torch.max(out, dim=1)
            data["tgt_ids"][:, i] = next_token
            if next_token == self.vocab.eos_token_id:
                break

        return data["tgt_ids"]

    def pos_matrices(self, tgt_ids: Tensor) -> tuple[Tensor, Tensor]:
        batch_tgt_anc, batch_tgt_sib = [], []
        for partial_ids in tgt_ids.tolist():
            partial_tok = [self.vocab.id2token(i) for i in partial_ids]
            tgt_anc, tgt_sib = partial_to_matrices(partial_tok, self.k)
            batch_tgt_anc.append(tgt_anc)
            batch_tgt_sib.append(tgt_sib)

        return torch.stack(batch_tgt_anc), torch.stack(batch_tgt_sib)
