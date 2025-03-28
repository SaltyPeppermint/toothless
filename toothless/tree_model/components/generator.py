from torch import Tensor
import torch.nn.functional as F
from torch import nn

from toothless.tree_model.args import ModelArguments


class Generator(nn.Module):
    def __init__(self, conf: ModelArguments, tgt_vocab_size: int, seq_len: int, k: int):
        super(Generator, self).__init__()

        self.seq_len = seq_len
        self.k = k

        self.token_linear = nn.Linear(conf.d_model, tgt_vocab_size)
        self.token_dropout = nn.Dropout(conf.dropout)

        self.anc_linear = nn.Linear(conf.d_model, seq_len * (2 * k + 2))
        self.anc_dropout = nn.Dropout(conf.dropout)

        self.sib_linear = nn.Linear(conf.d_model, seq_len * (2 * k + 2))
        self.sib_dropout = nn.Dropout(conf.dropout)

    def forward(self, outputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        token_out = self.token_linear(outputs)
        token_logits = F.log_softmax(self.token_dropout(token_out), dim=-1)
        anc_out = self.anc_linear(outputs)
        anc_logits = F.log_softmax(self.anc_dropout(anc_out), dim=-1).view(
            self.seq_len, -1, self.seq_len, (2 * self.k + 2)
        )
        sib_out = self.sib_linear(outputs).view(self.seq_len, -1)
        sib_logits = F.log_softmax(self.sib_dropout(sib_out), dim=-1).view(
            self.seq_len, -1, self.seq_len, (2 * self.k + 2)
        )
        return token_logits, anc_logits, sib_logits


# class GreedyGenerator(nn.Module):
#     def __init__(self, model, max_tgt_len, unk_id, bos_id, multi_gpu=False):
#         super(GreedyGenerator, self).__init__()

#         self.model = model
#         self.max_tgt_len = max_tgt_len
#         self.start_pos = bos_id
#         self.unk_id = unk_id

#     def forward(self, data):
#         data.tgt_seq = None
#         self.model.process_data(data)

#         encoder_outputs = self.model.encode(data)

#         batch_size = encoder_outputs.size(0)
#         ys = torch.ones(batch_size, 1).fill_(self.start_pos).long().to(encoder_outputs.device)
#         for i in range(self.max_tgt_len - 1):
#             data.tgt_mask = make_std_mask(ys, 0)
#             data.tgt_emb = self.model.tgt_embedding(Variable(ys))
#             decoder_outputs, decoder_attn = self.model.decode(data, encoder_outputs)
#             out = self.model.generator(decoder_outputs)
#             out = out[:, -1, :]
#             _, next_word = torch.max(out, dim=1)
#             ys = torch.cat([ys, next_word.unsqueeze(1).long().to(encoder_outputs.device)], dim=1)

#         return ys[:, 1:]
