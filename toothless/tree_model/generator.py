import torch.nn as nn
import torch

from dataset import make_std_mask
from model import FastASTTrans
from utils import UNK, BOS


class Generator(nn.Module):
    def __init__(self, tgt_vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super(Generator, self).__init__()
        self.soft_max = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, outputs):
        out = self.linear(outputs)
        gen_prob = self.soft_max(self.dropout(out))
        return torch.log(gen_prob)


class GreedyGenerator(nn.Module):
    def __init__(self, model: FastASTTrans, max_tgt_len: int):  # smth about multi gpu and model.module?
        super(GreedyGenerator, self).__init__()

        self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = BOS
        self.unk_pos = UNK

    def forward(self, data):
        data.tgt_seq = None
        self.model.process_data(data)

        encoder_outputs = self.model.encode(data)

        batch_size = encoder_outputs.size(0)
        ys = torch.ones(batch_size, 1, requires_grad=False).fill_(self.start_pos).long().to(encoder_outputs.device)
        for i in range(self.max_tgt_len - 1):
            data.tgt_mask = make_std_mask(ys, 0)
            data.tgt_emb = self.model.tgt_embedding(ys)
            decoder_outputs, decoder_attn = self.model.decode(data, encoder_outputs)
            out = self.model.generator(decoder_outputs)
            out = out[:, -1, :]
            _, next_word = torch.max(out, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1).long().to(encoder_outputs.device)], dim=1)

        return ys[:, 1:]
