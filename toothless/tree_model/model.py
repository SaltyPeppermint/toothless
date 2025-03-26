from torch import Tensor
from torch import nn

from toothless.tree_model.components.decoder import ASTDoubleDecoder
from toothless.tree_model.components.encoder import ASTEncoder
from toothless.tree_model.components.utils import Embeddings, Generator


class FastASTTrans(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_layers: int,
        dim_feed_forward: int,
        dropout: float,
        pos_type: str,
        n_anc_heads: int,
        n_sib_heads: int,
        max_rel_pos: int,
        state_dict=None,
    ):
        super(FastASTTrans, self).__init__()
        self.num_heads = n_anc_heads + n_sib_heads

        self.pos_type = pos_type.split("_")

        self.l_embedding = Embeddings(d_model, src_vocab_size, dropout=dropout, with_pos=False)
        self.r_embedding = Embeddings(d_model, src_vocab_size, dropout=dropout, with_pos=False)
        self.tgt_embedding = Embeddings(d_model, tgt_vocab_size, dropout=dropout, with_pos=True)

        self.l_encoder = ASTEncoder(
            d_model, dim_feed_forward, num_layers, n_anc_heads, n_sib_heads, self.pos_type, max_rel_pos, dropout=dropout
        )
        self.r_encoder = ASTEncoder(
            d_model, dim_feed_forward, num_layers, n_anc_heads, n_sib_heads, self.pos_type, max_rel_pos, dropout=dropout
        )
        self.decoder = ASTDoubleDecoder(
            d_model, dim_feed_forward, num_layers, n_anc_heads, n_sib_heads, self.pos_type, max_rel_pos, dropout=dropout
        )

        self.generator = Generator(tgt_vocab_size, d_model, dropout)

        print("Init or load model.")
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    # @staticmethod
    # def process_data(data):
    #     batch_size = data.num_graphs
    #     for key in data.keys:
    #         new_value_shape = (batch_size, -1) + data[key].size()[1:]
    #         data[key] = data[key].view(*new_value_shape)

    def forward(self, data: dict[str, Tensor]):
        l_mem = self.encode_l(data)
        r_mem = self.encode_r(data)

        decoder_outputs = self.decode(data, l_mem, r_mem)
        output = self.generator(decoder_outputs)
        return output

    def encode_l(self, data: dict[str, Tensor]) -> Tensor:
        l_emb = self.l_embedding(data["l_ids"])
        l_mem = self.l_encoder(l_emb, data["l_anc"], data["l_sib"], data["l_mask"])
        return l_mem

    def encode_r(self, data: dict[str, Tensor]) -> Tensor:
        r_emb = self.r_embedding(data["r_ids"])
        r_mem = self.r_encoder(r_emb, data["r_anc"], data["r_sib"], data["r_mask"])
        return r_mem

    def decode(self, data: dict[str, Tensor], l_mem: Tensor, r_mem: Tensor):
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
