import math

import torch
from torch import Tensor
from torch import nn

from .layers.encoder import EncoderLayer
from .layers.decoder import DecoderLayer, DualDecoderLayer
from .args import ModelArgs


class DualTransformer(nn.Module):
    def __init__(self, conf: ModelArgs, vocab_size: int, pad_token_id: int = 0):
        super(DualTransformer, self).__init__()

        self.d_model = conf.d_model
        self.pad_token_id = pad_token_id

        self.l_embedding = nn.Embedding(vocab_size, conf.d_model)
        self.target_embedding = nn.Embedding(vocab_size, conf.d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, conf.d_model)

        # Encoders
        self.start_encoder = nn.ModuleList([EncoderLayer(conf) for _ in range(conf.num_layers)])
        self.target_encoder = nn.ModuleList([EncoderLayer(conf) for _ in range(conf.num_layers)])

        # Decoder
        self.decoder = nn.ModuleList([DualDecoderLayer(conf) for _ in range(conf.num_layers)])

        # Output projection
        self.output_proj = nn.Linear(conf.d_model, vocab_size)
        self.output_norm = nn.RMSNorm(conf.d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.l_embedding.weight, mean=0.0, std=conf.d_model**-0.5)
        nn.init.normal_(self.target_embedding.weight, mean=0.0, std=conf.d_model**-0.5)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=conf.d_model**-0.5)

    @torch.compile(fullgraph=True)
    def start_encode(self, start_ids: Tensor, start_mask: Tensor):
        # Embeddings
        start_mem = self.l_embedding(start_ids) * math.sqrt(self.d_model)

        # Compute each RoPE encoder layer
        for layer in self.start_encoder:
            start_mem = layer(start_mem, start_mask)
        return start_mem

    @torch.compile(fullgraph=True)
    def target_encode(self, target_ids: Tensor, target_mask: Tensor):
        # Embeddings
        target_mem = self.target_embedding(target_ids) * math.sqrt(self.d_model)

        # Compute each RoPE encoder layer
        for layer in self.target_encoder:
            target_mem = layer(target_mem, target_mask)
        return target_mem

    @torch.compile(fullgraph=True)
    def decode(
        self,
        guide_ids: Tensor,
        guide_mask: Tensor,
        start_mem: Tensor,
        start_mask: Tensor,
        target_mem: Tensor,
        target_mask: Tensor,
    ):
        """Decode target sequence using fused encoder memories."""

        # Target embeddings
        output = self.tgt_embedding(guide_ids) * math.sqrt(self.d_model)

        # Compute each RoPE decoder layer
        for layer in self.decoder:
            output = layer(output, guide_mask, start_mem, start_mask, target_mem, target_mask)

        return self.output_proj(self.output_norm(output))

    @torch.compile(fullgraph=True)
    def forward(self, batch: dict[str, Tensor]):
        # Encode both source sequences
        start_mem = self.start_encode(batch["start_ids"], batch["start_mask"])
        target_mem = self.target_encode(batch["target_ids"], batch["target_mask"])
        # Decode and project to vocabulary
        return self.decode(
            batch["guide_ids"], batch["guide_mask"], start_mem, batch["start_mask"], target_mem, batch["target_mask"]
        )


class DecoderOnly(nn.Module):
    def __init__(self, conf: ModelArgs, vocab_size: int, pad_token_id: int = 0):
        super(DecoderOnly, self).__init__()

        self.d_model = conf.d_model
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, conf.d_model)

        # Decoder
        self.decoder = nn.ModuleList([DecoderLayer(conf) for _ in range(conf.num_layers)])

        # Output projection
        self.output_proj = nn.Linear(conf.d_model, vocab_size)
        self.output_norm = nn.RMSNorm(conf.d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=conf.d_model**-0.5)

    @torch.compile(fullgraph=True)
    def forward(
        self,
        tgt_ids: Tensor,
        tgt_mask: Tensor,
    ):
        # Encode both source sequences
        # Target embeddings
        output = self.embedding(tgt_ids) * math.sqrt(self.d_model)

        # Compute each RoPE decoder layer
        for layer in self.decoder:
            output = layer(output, tgt_mask)

        return self.output_proj(self.output_norm(output))
