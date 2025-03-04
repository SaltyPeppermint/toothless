from dataclasses import dataclass

from torch import nn
import torch.nn.functional as F

from toothless.tree_model.components import FastASTEncoder, FastASTEncoderLayer
from toothless.tree_model.generator import Generator
from toothless.tree_model.dataset import make_std_mask
from toothless.tree_model.utils import PAD
from toothless.tree_model.embeddings import Embeddings


@dataclass
class MHAConfig(nn.Module):
    simple_heads: int = 0
    ancestor_heads: int = 0
    sibling_heads: int = 0
    depth_heads: int = 0

    def total_heads(self) -> int:
        return self.simple_heads + self.ancestor_heads + self.sibling_heads + self.depth_heads


class FastASTTrans(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        mha_config: MHAConfig,
        max_rel_pos: int,
        pos_type: str,
        num_layers: int,
        dim_feed_forward: int,
        dropout: float,
        state_dict=None,
    ):
        super(FastASTTrans, self).__init__()
        self.num_heads = mha_config.total_heads()

        self.pos_type = pos_type.split("_")

        self.src_embedding = Embeddings(d_model, src_vocab_size, dropout=dropout, with_pos=False)
        self.tgt_embedding = Embeddings(d_model, tgt_vocab_size, dropout=dropout, with_pos=True)

        encoder_layer = FastASTEncoderLayer(d_model, self.num_heads, dim_feed_forward, dropout)
        self.encoder = FastASTEncoder(
            encoder_layer, num_layers, mha_config, self.pos_type, max_rel_pos, d_model, dropout=dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, self.num_heads, dim_feedforward=dim_feed_forward, dropout=dropout, activation=F.gelu
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(d_model))

        self.generator = Generator(tgt_vocab_size, d_model, dropout)

        print("Init or load model.")
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    def base_process(self, data):
        self.process_data(data)

        src_seq = data.src_seq
        data.src_mask = src_seq.eq(PAD)
        data.src_emb = self.src_embedding(src_seq)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq
            data.tgt_mask = make_std_mask(tgt_seq, PAD)
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

        encoder_outputs = self.encode(data)
        decoder_outputs, attn_weights = self.decode(data, encoder_outputs)
        out = self.generator(decoder_outputs)
        return out

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data, encoder_outputs):
        tgt_emb = data.tgt_emb
        tgt_mask = data.tgt_mask
        src_mask = data.src_mask

        tgt_emb = tgt_emb.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
        outputs, attn_weights = self.decoder(
            tgt=tgt_emb, memory=encoder_outputs, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask
        )
        outputs = outputs.permute(1, 0, 2)
        return outputs, attn_weights
