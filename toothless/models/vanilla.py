from torch import Tensor
from torch import nn
import torch

from toothless.models.utils import create_causal_mask, create_padding_mask  # type: ignore # noqa: F401

from .layers import Embeddings
from ..args import ModelArguments
from ..vocab import SimpleVocab


class VanillaDualTreeTransformer(nn.Module):
    def __init__(self, conf: ModelArguments, src_vocab_size: int, tgt_vocab_size: int, k: int, state_dict=None):
        super(VanillaDualTreeTransformer, self).__init__()

        assert conf.n_heads == conf.anc_heads + conf.sib_heads

        self.conf = conf

        self.l_embedding = Embeddings(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=True)
        self.r_embedding = Embeddings(conf.d_model, src_vocab_size, dropout=conf.dropout, with_pos=True)
        self.tgt_embedding = Embeddings(conf.d_model, tgt_vocab_size, dropout=conf.dropout, with_pos=True)

        # Left Encoder
        l_encoder_layer = nn.TransformerEncoderLayer(
            d_model=conf.d_model,
            dim_feedforward=conf.dim_feed_forward,
            dropout=conf.dropout,
            batch_first=True,
            nhead=conf.n_heads,
            activation="gelu",
        )
        self.l_encoder = nn.TransformerEncoder(l_encoder_layer, num_layers=conf.num_layers)

        # Right Encoder
        r_encoder_layer = nn.TransformerEncoderLayer(
            d_model=conf.d_model,
            dim_feedforward=conf.dim_feed_forward,
            dropout=conf.dropout,
            batch_first=True,
            nhead=conf.n_heads,
            activation="gelu",
        )
        self.r_encoder = nn.TransformerEncoder(r_encoder_layer, num_layers=conf.num_layers)

        # Memory fusion layer to combine outputs from both encoders
        self.memory_fusion = nn.Linear(conf.d_model * 2, conf.d_model)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=conf.d_model,
            dim_feedforward=conf.dim_feed_forward,
            nhead=conf.n_heads,
            dropout=conf.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, conf.num_layers)

        # Output projection
        self.output_proj = nn.Linear(conf.d_model, tgt_vocab_size)

        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)

    def l_encode(self, l_ids: Tensor, l_mask: Tensor | None) -> Tensor:
        l_emb = self.l_embedding(l_ids)
        return self.l_encoder(l_emb, src_key_padding_mask=l_mask)

    def r_encode(self, r_ids: Tensor, r_mask: Tensor | None) -> Tensor:
        r_emb = self.l_embedding(r_ids)
        return self.l_encoder(r_emb, src_key_padding_mask=r_mask)

    def fuse_memories(self, l_mem: Tensor, r_mem: Tensor):
        """Combine outputs from both encoders."""
        # Simple concatenation + linear projection
        # Alternative approaches: attention-based fusion, gating, etc.
        batch_size, seq_len1, d_model = l_mem.shape
        seq_len2 = r_mem.shape[1]

        # Pad shorter sequence to match longer one
        if seq_len1 > seq_len2:
            padding = torch.zeros(batch_size, seq_len1 - seq_len2, d_model, device=r_mem.device)
            r_mem = torch.cat([r_mem, padding], dim=1)
        elif seq_len2 > seq_len1:
            padding = torch.zeros(batch_size, seq_len2 - seq_len1, d_model, device=l_mem.device)
            l_mem = torch.cat([l_mem, padding], dim=1)

        # Concatenate and project
        fused_memory = torch.cat([l_mem, r_mem], dim=-1)  # [batch, seq_len, 2*d_model]

        return self.memory_fusion(fused_memory)  # [batch, seq_len, d_model]

    @staticmethod
    def fuse_mask(l_mask: Tensor | None, r_mask: Tensor | None) -> Tensor | None:
        # Create memory padding mask
        memory_padding_mask = None
        if l_mask is not None and r_mask is not None:
            # Use the longer sequence's padding mask
            if l_mask.shape[1] >= r_mask.shape[1]:
                memory_padding_mask = l_mask
            else:
                memory_padding_mask = r_mask

        return memory_padding_mask

    def decode(self, tgt: Tensor, fused_memory: Tensor, tgt_padding_mask=None, memory_padding_mask=None):
        """Decode target sequence using fused encoder memories."""
        seq_len = tgt.shape[1]
        device = tgt.device

        # Create causal mask for decoder self-attention
        tgt_causal_mask = create_causal_mask(seq_len, device)

        # Target embeddings
        tgt_emb = self.tgt_embedding(tgt)

        # Apply decoder
        output = self.decoder(
            tgt_emb,
            fused_memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        return output

    def forward(
        self,
        tgt: Tensor,
        l_ids: Tensor,
        r_ids: Tensor,
        l_mask: Tensor | None = None,
        r_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ):
        # Encode both source sequences
        l_mem = self.l_encode(l_ids, l_mask)
        r_mem = self.r_encode(r_ids, r_mask)

        # Fuse outputs together
        fused_memory = self.fuse_memories(l_mem, r_mem)

        # Fuse masks
        fused_mask = self.fuse_mask(l_mask, r_mask)

        # Decode
        decoder_output = self.decode(tgt, fused_memory, tgt_padding_mask=tgt_mask, memory_padding_mask=fused_mask)

        # Project to vocabulary
        return self.output_proj(decoder_output)


class VanillaGreedyGenerator(nn.Module):
    # smth about multi gpu and model.module?
    def __init__(self, model: VanillaDualTreeTransformer, max_len: int, vocab: SimpleVocab, k: int):
        super(VanillaGreedyGenerator, self).__init__()

        self.model = model
        self.max_len = max_len
        self.vocab = vocab
        self.k = k

    def model_device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]):
        device = self.model_device()

        batch = {k: v.to(device) for k, v in batch.items()}

        l_mask = create_padding_mask(batch["l_ids"])
        r_mask = create_padding_mask(batch["r_ids"])
        l_mem = self.model.l_encode(batch["l_ids"], l_mask)
        r_mem = self.model.r_encode(batch["r_ids"], r_mask)

        fused_mem = self.model.fuse_memories(l_mem, r_mem)
        fused_mask = self.model.fuse_mask(l_mask, r_mask)

        tgt_padding_mask = create_padding_mask(batch["tgt_ids"])
        batch_size = fused_mem.size(0)

        assert self.vocab.pad_token_id == 0

        batch["tgt_ids"] = torch.zeros(batch_size, self.max_len, device=device, dtype=torch.long)
        batch["tgt_ids"][:, 0] = self.vocab.bos_token_id
        batch["tgt_probs"] = torch.zeros(batch_size, self.max_len, device=device, dtype=torch.float)
        finished_flags = torch.full((batch_size,), False).to(device)

        for i in range(self.max_len - 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            decoder_outputs = self.model.decode(
                batch["tgt_ids"], fused_mem, tgt_padding_mask=tgt_padding_mask, memory_padding_mask=fused_mask
            )
            fresh_out = decoder_outputs[:, i, :].squeeze(1)

            prob, next_token = torch.max(fresh_out, dim=-1)

            batch["tgt_ids"][:, i + 1] = next_token
            batch["tgt_probs"][:, i + 1] = torch.exp(prob)

            finished_flags = finished_flags | next_token == self.vocab.eos_token_id
            if torch.all(finished_flags):
                break

        return batch["tgt_ids"], batch["tgt_probs"]
