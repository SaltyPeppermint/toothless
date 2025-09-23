from pathlib import Path
from dataclasses import dataclass


import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from dataclass_wizard import JSONWizard
from tokenizers import Tokenizer


from eggshell import EggshellException  # type: ignore
from eggshell import rise  # type: ignore

from toothless.model import DualTransformer

from .utils import rank0print
from .data import Triple


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


def create_padding_mask(input_ids: Tensor, pad_token_id: int = 0, device: torch.device | None = None) -> Tensor:
    """
    Creates a padding mask for attention mechanisms.
    """
    if device is None:
        device = input_ids.device

    return (input_ids != pad_token_id).to(device)


def generate_with_probabilities(
    model: DualTransformer | FSDP,
    batch: dict[str, Tensor],
    max_len: int,
    pad_token: int = 0,
    start_token: int = 1,
    end_token: int = 2,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: int | None = None,
) -> GenerationResult:
    """
    Generate sequences with token probabilities using various sampling strategies.

    Args:
        model: The VanillaDualTreeTransformer model
        batch: Batch of source and target sequences and masks dict[str, [batch_size, seq_len]]
        max_len: Maximum generation length
        pad_token:
        start_token:
        end_token:
        temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative, >1.0 = more random, 0.0 = no randomness)
        top_k: Keep only top k tokens for sampling (None = no filtering)
        top_p: Nucleus sampling - keep tokens with cumulative probability <= top_p (None = no filtering)

    Returns:
        GenerationResult object containing tokens and probabilities
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = batch["start_ids"].shape[0]

    batch = {k: v.to(device) for k, v in batch.items()}

    # Encode sources once
    with torch.no_grad():
        start_mem = model.start_encode(batch["start_ids"], batch["start_mask"])
        target_mem = model.target_encode(batch["target_ids"], batch["target_mask"])

    # Initialize generation with all start tokens
    generated_tokens = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    token_probabilities = torch.zeros((batch_size, max_len + 1), device=device)  # +1 for start token
    token_probabilities[:, 0] = 1.0  # Start token has probability 1.0

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        with torch.no_grad():
            tgt_mask = create_padding_mask(generated_tokens, pad_token_id=pad_token)
            # Get logits for current sequence
            logits = model.decode(
                generated_tokens, tgt_mask, start_mem, batch["start_mask"], target_mem, batch["target_mask"]
            )

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
    model: DualTransformer | FSDP,
    batch: dict[str, Tensor],
    tokenizer: Tokenizer,
    max_len: int,
    beam_size: int = 3,
    length_penalty: float = 1.0,
) -> GenerationResult:
    """
    Beam search generation with probabilities.

    Args:
        model: The VanillaDualTreeTransformer model
        start_batch: First source sequences [batch_size, seq_len]
        target_batch: Second source sequences [batch_size, seq_len]
        vocab: The used vocabulary
        beam_size: Number of beams to keep
        max_len: Maximum generation length
        length_penalty: Length penalty factor (>1.0 favors longer sequences)

    Returns:
        GenerationResult object with best beam for each batch
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = batch["start_ids"].shape[0]

    batch = {k: v.to(device) for k, v in batch.items()}

    pad_token = tokenizer.token_to_id("[PAD]")
    start_token = tokenizer.token_to_id("[CLS]")
    end_token = tokenizer.token_to_id("[SEP]")

    # Encode sources once
    with torch.no_grad():
        start_mem = model.start_encode(batch["start_ids"], batch["start_mask"])
        target_mem = model.target_encode(batch["target_ids"], batch["target_mask"])

    # Initialize beams for each batch element
    beams = torch.full((batch_size, beam_size, 1), start_token, device=device, dtype=torch.long)
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_token_probs = torch.ones((batch_size, beam_size, 1), device=device)  # Start with probability 1.0

    # Track finished beams
    finished_beams = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

    for step in range(max_len):
        # Expand memory for all beams
        start_expanded_memory = (
            start_mem.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(batch_size * beam_size, start_mem.shape[1], start_mem.shape[2])
        )
        target_expanded_memory = (
            target_mem.unsqueeze(1)
            .expand(-1, beam_size, -1, -1)
            .reshape(batch_size * beam_size, target_mem.shape[1], target_mem.shape[2])
        )
        current_beams = beams.reshape(batch_size * beam_size, -1)

        with torch.no_grad():
            tgt_mask = create_padding_mask(current_beams, pad_token_id=pad_token)
            logits = model.decode(
                current_beams,
                tgt_mask,
                start_expanded_memory,
                batch["start_mask"],
                target_expanded_memory,
                batch["target_mask"],
            )
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


@dataclass
class InferResult(JSONWizard):
    left: str
    middle: str
    right: str
    generated: str
    tokens: list[int]
    probs: list[float]


def batch_process_result(
    tokenizer: Tokenizer,
    triples: list[Triple],
    batch_ids: list[list[int]],
    batch_probs: list[list[float]],
    path: Path,
    id_offset: int,
    verbose: bool,
) -> list[InferResult]:
    path = path.absolute()

    batch_gen_triples = []
    for i, (triple, ids, token_probs) in enumerate(zip(triples, batch_ids, batch_probs)):
        sample_id = i + id_offset

        start_tokens = tokenizer.decode(triple.start_ids.tolist())
        start = rise.RecExpr(start_tokens)
        guide_tokens = tokenizer.decode(triple.guide_ids.tolist())
        guide = rise.RecExpr(guide_tokens)
        target_tokens = tokenizer.decode(triple.target_ids.tolist())
        target = rise.RecExpr(target_tokens)

        start.to_dot(f"{sample_id} left", str(path / f"{sample_id}_left"))
        guide.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle"))
        guide.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle_t"), transparent=True)
        target.to_dot(f"{sample_id} right", str(path / f"{sample_id}_right"))

        if verbose:
            rank0print("----------")
            rank0print(f"Sample {sample_id}", "blue")
            rank0print("LEFT:", "green")
            rank0print(start_tokens)
            rank0print("MIDDLE:", "green")
            rank0print(guide_tokens)
            rank0print("RIGHT:", "green")
            rank0print(target_tokens)

        raw_generated_tokens = tokenizer.decode(ids, skip_special_tokens=False)
        if verbose:
            rank0print("RAW GENERATED TOKENS:", "yellow")
            rank0print(raw_generated_tokens, "yellow")

        generated_tokens = tokenizer.decode(ids).replace("[var]", "?")
        try:
            generated = rise.Guide(generated_tokens)
        except EggshellException as e:
            rank0print("COULD NOT PROPERLY PARSE GENERATED GUIDE.", "red")
            rank0print(e, "red")
            continue

        generated.to_dot(f"{sample_id} generated", str(path / f"{sample_id}_generated"))
        generated.to_dot(f"{sample_id} generated", str(path / f"{sample_id}_generated_t"), transparent=True)
        batch_gen_triples.append(InferResult(str(start), str(guide), str(target), str(generated), ids, token_probs))

        if verbose:
            rank0print("GENERATED:", "green")
            rank0print(generated)

    return batch_gen_triples


# def print_distance(distances: list[FirstErrorDistance], ds_name: str):
#     rank0print(f"\n### AVERAGE DISTANCE IN {ds_name} DATASET ###", "yellow")
#     n_hits = sum([d.n_hits for d in distances])
#     rank0print(f"Hits: {n_hits}", "yellow")
#     n_misses = sum([d.n_misses for d in distances])
#     rank0print(f"Misses: {n_misses}", "yellow")
#     perfect_matches = sum([1 for d in distances if d.n_misses == 0])
#     rank0print(f"Perfect matches: {perfect_matches}", "yellow")

#     avg_hit_prob = _avg_prob([d.hit_probabilities() for d in distances if d])
#     rank0print(f"Average Hit Probability: {avg_hit_prob}", "yellow")
#     avg_miss_prob = _avg_prob([d.miss_probabilities() for d in distances if d])
#     rank0print(f"Average Miss Probability: {avg_miss_prob}", "yellow")
#     rank0print("\n")


def _avg_prob(probs: list[list[float | None]]):
    avg_prob = 0
    not_none = 0
    for i in probs:
        for j in i:
            if j is not None:
                avg_prob += j
                not_none += 1
    avg_prob = avg_prob / not_none
    return avg_prob
