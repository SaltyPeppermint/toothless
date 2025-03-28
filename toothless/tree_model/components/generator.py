from torch import Tensor
import torch.nn.functional as F
from torch import nn

from toothless.tree_model.args import ModelArguments


class Generator(nn.Module):
    def __init__(self, conf: ModelArguments, tgt_vocab_size: int):
        super(Generator, self).__init__()

        self.token_linear = nn.Linear(conf.d_model, tgt_vocab_size)
        self.token_dropout = nn.Dropout(conf.dropout)

    def forward(self, outputs: Tensor) -> Tensor:
        out = self.token_linear(outputs)
        return F.log_softmax(self.token_dropout(out), dim=-1)


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
