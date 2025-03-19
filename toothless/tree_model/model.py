import torch
import torch.nn.functional as F
from torch import nn

from toothless.tree_model.components.decoder import ASTDecoder, ASTDecoderLayer
from toothless.tree_model.components.encoder import ASTEncoder, ASTEncoderLayer
from toothless.tree_model.embeddings import Embeddings


class FastASTTrans(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_anc_heads: int,
        n_sib_heads: int,
        max_rel_pos: int,
        pos_type: str,
        num_layers: int,
        dim_feed_forward: int,
        dropout: float,
        state_dict=None,
    ):
        super(FastASTTrans, self).__init__()
        self.num_heads = n_anc_heads + n_sib_heads

        self.pos_type = pos_type.split("_")

        self.src_embedding = Embeddings(d_model, src_vocab_size, dropout=dropout, with_pos=False)
        self.tgt_embedding = Embeddings(d_model, tgt_vocab_size, dropout=dropout, with_pos=True)

        encoder_layer = ASTEncoderLayer(
            d_model, self.num_heads, dim_feed_forward, dropout, activation=F.gelu
        )
        self.l_encoder = ASTEncoder(
            encoder_layer,
            num_layers,
            n_anc_heads,
            n_sib_heads,
            self.pos_type,
            max_rel_pos,
            d_model,
            dropout=dropout,
        )
        self.r_encoder = ASTEncoder(
            encoder_layer,
            num_layers,
            n_anc_heads,
            n_sib_heads,
            self.pos_type,
            max_rel_pos,
            d_model,
            dropout=dropout,
        )

        decoder_layer = ASTDecoderLayer(
            d_model, self.num_heads, dim_feed_forward, dropout=dropout, activation=F.gelu
        )
        self.decoder = ASTDecoder(
            decoder_layer,
            num_layers,
            n_anc_heads,
            n_sib_heads,
            self.pos_type,
            max_rel_pos,
            d_model,
            dropout=dropout,
        )

        self.generator = Generator(tgt_vocab_size, d_model, dropout)

        print("Init or load model.")
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    def base_process(self, data, pad_token):
        self.process_data(data)

        src_seq = data.src_seq
        data.src_mask = src_seq.eq(pad_token)
        data.src_emb = self.src_embedding(src_seq)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq
            # data.tgt_mask = make_std_mask(tgt_seq, pad_token)
            data.tgt_emb = self.tgt_embedding(tgt_seq)

    @staticmethod
    def process_data(data):
        batch_size = data.num_graphs
        for key in data.keys:
            new_value_shape = (batch_size, -1) + data[key].size()[1:]
            data[key] = data[key].view(*new_value_shape)

    # def process_data(self, data):
    #     self.base_process(data)

    def forward(self, data):
        self.process_data(data)

        l_encoder_outputs = self.l_encode(data)
        r_encoder_outputs = self.r_encode(data)
        decoder_outputs, attn_weights = self.decode(data, l_encoder_outputs, r_encoder_outputs)
        out = self.generator(decoder_outputs)
        return out

    def l_encode(self, data):
        return self.l_encoder(data)

    def r_encode(self, data):
        return self.r_encoder(data)

    def decode(self, data, l_encoder_outputs, r_encoder_outputs):
        tgt_emb = data.tgt_emb
        # tgt_mask = data.tgt_mask
        src_mask = data.src_mask

        tgt_emb = tgt_emb.permute(1, 0, 2)
        # tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)

        l_encoder_outputs = l_encoder_outputs.permute(1, 0, 2)
        r_encoder_outputs = r_encoder_outputs.permute(1, 0, 2)

        # TODO: adapt decoder to accept two encoder inputs
        outputs, attn_weights = self.decoder(
            tgt=tgt_emb, memory=l_encoder_outputs, memory_key_padding_mask=src_mask
        )
        outputs = outputs.permute(1, 0, 2)
        return outputs, attn_weights


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
    def __init__(
        self, model: FastASTTrans, max_tgt_len: int, bos_token: int, unk_token
    ):  # smth about multi gpu and model.module?
        super(GreedyGenerator, self).__init__()

        self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = bos_token
        self.unk_pos = unk_token

    def forward(self, data):
        data.tgt_seq = None
        self.model.process_data(data)

        l_encoder_outputs = self.model.l_encode(data)
        r_encoder_outputs = self.model.r_encode(data)

        batch_size = r_encoder_outputs.size(0)
        ys = (
            torch.ones(batch_size, 1, requires_grad=False)
            .fill_(self.start_pos)
            .long()
            .to(r_encoder_outputs.device)
        )
        for i in range(self.max_tgt_len - 1):
            # data.tgt_mask = make_std_mask(ys, 0)
            data.tgt_emb = self.model.tgt_embedding(ys)
            decoder_outputs, decoder_attn = self.model.decode(
                data, l_encoder_outputs, r_encoder_outputs
            )

            out = self.model.generator(decoder_outputs)
            out = out[:, -1, :]
            _, next_word = torch.max(out, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1).long().to(r_encoder_outputs.device)], dim=1)

        return ys[:, 1:]
