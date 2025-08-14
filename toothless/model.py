import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .layers.decoder import TransformerDecoderLayer
from .layers.encoder import TransformerEncoderLayer
from .args import ModelArguments
from .vocab import SimpleVocab


class DualTreeTransformer(nn.Module):
    def __init__(
        self, conf: ModelArguments, src_vocab_size: int, tgt_vocab_size: int, pad_token_id: int, state_dict=None
    ):
        super(DualTreeTransformer, self).__init__()

        self.conf = conf
        self.pad_token_id = pad_token_id

        self.l_embedding = nn.Embedding(src_vocab_size, conf.d_model)
        self.r_embedding = nn.Embedding(src_vocab_size, conf.d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, conf.d_model)

        # Encoders
        self.l_encoder = nn.ModuleList([TransformerEncoderLayer(conf) for _ in range(conf.num_layers)])
        self.r_encoder = nn.ModuleList([TransformerEncoderLayer(conf) for _ in range(conf.num_layers)])

        # Decoder
        self.decoder = nn.ModuleList([TransformerDecoderLayer(conf) for _ in range(conf.num_layers)])

        # Output projection
        self.output_proj = nn.Linear(conf.d_model, tgt_vocab_size)
        self.output_norm = nn.RMSNorm(conf.d_model)

        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.normal_(self.l_embedding.weight, mean=0.0, std=self.conf.d_model**-0.5)
            nn.init.normal_(self.r_embedding.weight, mean=0.0, std=self.conf.d_model**-0.5)
            nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=self.conf.d_model**-0.5)
        else:
            self.load_state_dict(state_dict)

    @torch.compile()
    def l_encode(self, l_ids: Tensor, l_mask: Tensor):
        # Embeddings
        l_mem = self.l_embedding(l_ids) * math.sqrt(self.conf.d_model)

        # Compute each RoPE encoder layer
        for layer in self.l_encoder:
            l_mem = layer(l_mem, l_mask)
        return l_mem

    @torch.compile()
    def r_encode(self, r_ids: Tensor, r_mask: Tensor):
        # Embeddings
        r_mem = self.r_embedding(r_ids) * math.sqrt(self.conf.d_model)

        # Compute each RoPE encoder layer
        for layer in self.r_encoder:
            r_mem = layer(r_mem, r_mask)
        return r_mem

    @torch.compile()
    def decode(self, tgt: Tensor, l_mem: Tensor, r_mem: Tensor, tgt_mask: Tensor, l_mask: Tensor, r_mask: Tensor):
        """Decode target sequence using fused encoder memories."""

        # Target embeddings
        output = self.tgt_embedding(tgt) * math.sqrt(self.conf.d_model)

        # Compute each RoPE decoder layer
        for layer in self.decoder:
            output = layer(output, l_mem, r_mem, tgt_mask, l_mask, r_mask)

        return self.output_proj(self.output_norm(output))

    @torch.compile()
    def forward(self, tgt_ids: Tensor, l_ids: Tensor, r_ids: Tensor):
        # Encode both source sequences
        l_mask = create_padding_mask(l_ids, pad_token_id=self.pad_token_id)
        l_mem = self.l_encode(l_ids, l_mask)

        r_mask = create_padding_mask(r_ids, pad_token_id=self.pad_token_id)
        r_mem = self.r_encode(r_ids, r_mask)

        tgt_mask = create_padding_mask(tgt_ids, pad_token_id=self.pad_token_id)
        # Decode and project to vocabulary
        return self.decode(tgt_ids, l_mem, r_mem, tgt_mask, l_mask, r_mask)


def create_padding_mask(
    input_ids: Tensor,
    pad_token_id: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Creates a padding mask for attention mechanisms.
    """
    if device is None:
        device = input_ids.device

    mask = (input_ids == pad_token_id).unsqueeze(1).unsqueeze(2)

    return mask.to(device)


@dataclass
class GenerationResult:
    """Container for generation results with tokens and probabilities."""

    tokens: Tensor
    token_probs: Tensor
    sequence_probs: Tensor | None
    attention_weights: Tensor | None

    def __init__(
        self,
        tokens: Tensor,
        token_probs: Tensor,
        sequence_probs: Tensor | None = None,
        attention_weights: Tensor | None = None,
    ):
        self.tokens = tokens  # Generated token IDs [batch_size, seq_len]
        self.token_probs = token_probs  # Probability of each generated token [batch_size, seq_len]
        self.sequence_probs = sequence_probs  # Overall sequence probability [batch_size]
        self.attention_weights = attention_weights  # Optional attention weights

    def to_dict(self) -> dict[str, list]:
        """Convert to dictionary for easy serialization."""
        result = {"tokens": self.tokens.tolist(), "token_probs": self.token_probs.tolist()}
        if self.sequence_probs is not None:
            result["sequence_probs"] = self.sequence_probs.tolist()
        if self.attention_weights is not None:
            result["attention_weights"] = self.attention_weights.tolist()
        return result


def generate_with_probabilities(
    model: DualTreeTransformer | FSDP,
    l_batch: Tensor,
    r_batch: Tensor,
    vocab: SimpleVocab,
    max_len: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: int | None = None,
) -> GenerationResult:
    """
    Generate sequences with token probabilities using various sampling strategies.

    Args:
        model: The VanillaDualTreeTransformer model
        l_batch: First source sequences [batch_size, seq_len]
        r_batch: Second source sequences [batch_size, seq_len]
        max_len: Maximum generation length
        vocab: The used vocabulary
        temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more random, 0.0 = no randomness)
        top_k: Keep only top k tokens for sampling (None = no filtering)
        top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p (None = no filtering)

    Returns:
        GenerationResult object containing tokens and probabilities
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = l_batch.shape[0]

    pad_token = vocab.pad_token_id
    start_token = vocab.bos_token_id
    end_token = vocab.eos_token_id

    l_batch = l_batch.to(device)
    l_mask = create_padding_mask(l_batch, pad_token_id=pad_token)

    r_batch = r_batch.to(device)
    r_mask = create_padding_mask(r_batch, pad_token_id=pad_token)

    # Encode sources once
    with torch.no_grad():
        l_mem = model.l_encode(l_batch, l_mask)
        r_mem = model.r_encode(r_batch, r_mask)

    # Initialize generation with all start tokens
    generated_tokens = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    token_probabilities = torch.zeros((batch_size, max_len + 1), device=device)  # +1 for start token
    token_probabilities[:, 0] = 1.0  # Start token has probability 1.0

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        with torch.no_grad():
            tgt_mask = create_padding_mask(generated_tokens, pad_token_id=pad_token)
            # Get logits for current sequence
            logits = model.decode(generated_tokens, l_mem, r_mem, tgt_mask, l_mask, r_mask)

            next_token_logits = logits[:, -1, :]  # Last position logits

            # Apply temperature
            if temperature != 1.0 and temperature != 0.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # Set non-top-k logits to -inf
                filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
                next_token_logits = filtered_logits

            # Convert to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Set filtered probabilities to 0
            sorted_probs[sorted_indices_to_remove] = 0
            # Renormalize
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # Scatter back to original order
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, sorted_indices, sorted_probs)

        # Sample next token
        if temperature == 0.0:
            # Greedy sampling
            next_token = probs.argmax(dim=-1, keepdim=True)
            selected_probs = probs.gather(-1, next_token).squeeze(-1)
        else:
            # Multinomial sampling
            next_token = torch.multinomial(probs, 1)
            selected_probs = probs.gather(-1, next_token).squeeze(-1)

        # Store probabilities for non-finished sequences
        active_mask = ~finished
        token_probabilities[active_mask, step + 1] = selected_probs[active_mask]

        # Update generated tokens for non-finished sequences
        next_token_masked = next_token.squeeze(-1)
        next_token_masked = torch.where(finished, torch.zeros_like(next_token_masked), next_token_masked)
        generated_tokens = torch.cat([generated_tokens, next_token_masked.unsqueeze(-1)], dim=1)

        # Check for end tokens, set flag if finished
        finished = finished | (next_token.squeeze(-1) == end_token)

        # If all sequences are finished, we are done
        if finished.all():
            break

    # Calculate sequence probabilities (log probability of the entire sequence)
    actual_length = generated_tokens.shape[1]
    sequence_log_probs = torch.log(token_probabilities[:, :actual_length] + 1e-10).sum(dim=1)
    sequence_probs = torch.exp(sequence_log_probs)

    return GenerationResult(
        tokens=generated_tokens, token_probs=token_probabilities[:, :actual_length], sequence_probs=sequence_probs
    )


def beam_search_with_probabilities(
    model: DualTreeTransformer | FSDP,
    l_batch: Tensor,
    r_batch: Tensor,
    vocab: SimpleVocab,
    max_len: int,
    beam_size: int = 3,
    length_penalty: float = 1.0,
) -> GenerationResult:
    """
    Beam search generation with probabilities.

    Args:
        model: The VanillaDualTreeTransformer model
        l_batch: First source sequences [batch_size, seq_len]
        r_batch: Second source sequences [batch_size, seq_len]
        vocab: The used vocabulary
        beam_size: Number of beams to keep
        max_len: Maximum generation length
        length_penalty: Length penalty factor (>1.0 favors longer sequences)

    Returns:
        GenerationResult object with best beam for each batch
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = l_batch.shape[0]

    pad_token = vocab.pad_token_id
    start_token = vocab.bos_token_id
    end_token = vocab.eos_token_id

    l_batch = l_batch.to(device)
    l_mask = create_padding_mask(l_batch, pad_token_id=pad_token)

    r_batch = r_batch.to(device)
    r_mask = create_padding_mask(r_batch, pad_token_id=pad_token)

    # Encode sources once
    with torch.no_grad():
        l_mem = model.l_encode(l_batch, l_mask)
        r_mem = model.r_encode(r_batch, r_mask)

    # Initialize beams for each batch element
    beams = torch.full((batch_size, beam_size, 1), start_token, device=device, dtype=torch.long)
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_token_probs = torch.ones((batch_size, beam_size, 1), device=device)  # Start with probability 1.0

    # Track finished beams
    finished_beams = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        # Expand memory for all beams
        l_expanded_memory = (
            l_mem.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(batch_size * beam_size, l_mem.shape[1], l_mem.shape[2])
        )
        r_expanded_memory = (
            r_mem.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(batch_size * beam_size, r_mem.shape[1], r_mem.shape[2])
        )
        current_beams = beams.reshape(batch_size * beam_size, -1)

        with torch.no_grad():
            tgt_mask = create_padding_mask(current_beams, pad_token_id=pad_token)
            logits = model.decode(current_beams, l_expanded_memory, r_expanded_memory, tgt_mask, l_mask, r_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            probs = torch.softmax(logits[:, -1, :], dim=-1)

        # Reshape back to beam dimension
        log_probs = log_probs.reshape(batch_size, beam_size, -1)
        probs = probs.reshape(batch_size, beam_size, -1)

        # For finished beams, only allow end token to maintain score
        vocab_size = log_probs.shape[-1]
        finished_mask = finished_beams.unsqueeze(-1).expand(-1, -1, vocab_size)
        log_probs = torch.where(finished_mask, torch.full_like(log_probs, float("-inf")), log_probs)
        # Allow end token for finished beams
        if end_token < vocab_size:
            log_probs[finished_beams, :, end_token] = 0.0

        # Add to beam scores
        candidate_scores = beam_scores.unsqueeze(-1) + log_probs

        # Get top beam_size candidates for each batch
        candidate_scores_flat = candidate_scores.reshape(batch_size, -1)
        top_scores, top_indices = candidate_scores_flat.topk(beam_size, dim=-1)

        # Convert flat indices back to beam and token indices
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # Update beams and probabilities
        new_beams = []
        new_token_probs = []

        for b in range(batch_size):
            batch_new_beams = []
            batch_new_probs = []

            for i in range(beam_size):
                old_beam_idx = beam_indices[b, i]
                old_beam = beams[b, old_beam_idx]
                old_probs = beam_token_probs[b, old_beam_idx]

                new_token = token_indices[b, i].unsqueeze(0)
                new_token_prob = probs[b, old_beam_idx, token_indices[b, i]].unsqueeze(0)

                new_beam = torch.cat([old_beam, new_token])
                new_prob_seq = torch.cat([old_probs, new_token_prob])

                batch_new_beams.append(new_beam)
                batch_new_probs.append(new_prob_seq)

            new_beams.append(torch.stack(batch_new_beams))
            new_token_probs.append(torch.stack(batch_new_probs))

        beams = torch.stack(new_beams)
        beam_token_probs = torch.stack(new_token_probs)

        # Apply length penalty to scores
        current_lengths = beams.shape[-1]
        if length_penalty != 1.0:
            length_penalties = torch.pow(
                torch.tensor(current_lengths, device=device, dtype=torch.float), length_penalty
            )
            beam_scores = top_scores / length_penalties
        else:
            beam_scores = top_scores

        # Update finished status
        new_finished = token_indices == end_token
        for b in range(batch_size):
            for i in range(beam_size):
                old_beam_idx = beam_indices[b, i]
                finished_beams[b, i] = finished_beams[b, old_beam_idx] or new_finished[b, i]

        # If all beams are finished, break
        if finished_beams.all():
            break

    # Select best beam for each batch element
    best_beam_indices = beam_scores.argmax(dim=-1)
    result_tokens = []
    result_probs = []
    result_sequence_probs = []

    for b in range(batch_size):
        best_idx = best_beam_indices[b]
        best_tokens = beams[b, best_idx]
        best_token_probs = beam_token_probs[b, best_idx]

        # Calculate sequence probability
        sequence_prob = torch.prod(best_token_probs)

        result_tokens.append(best_tokens)
        result_probs.append(best_token_probs)
        result_sequence_probs.append(sequence_prob)

    # Pad sequences to same length for batching
    max_len = max(len(seq) for seq in result_tokens)
    padded_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    padded_probs = torch.zeros(batch_size, max_len, device=device)

    for i, (tokens, probs) in enumerate(zip(result_tokens, result_probs)):
        padded_tokens[i, : len(tokens)] = tokens
        padded_probs[i, : len(probs)] = probs

    return GenerationResult(
        tokens=padded_tokens, token_probs=padded_probs, sequence_probs=torch.stack(result_sequence_probs)
    )
