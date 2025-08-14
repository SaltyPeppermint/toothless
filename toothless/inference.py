from pathlib import Path
from dataclasses import dataclass

from dataclass_wizard import JSONWizard

from eggshell import FirstErrorDistance, EggshellException
from eggshell import rise  # type: ignore

from .vocab import SimpleVocab
from .utils import rank0print
from .data import split_off_special, Tripple


@dataclass
class InferResult(JSONWizard):
    left: str
    right: str
    middle: str
    generated: str
    rules_chain: list[str]


def batch_process_result(
    vocab: SimpleVocab,
    tripples: list[Tripple],
    batch_ids: list[list[int]],
    batch_probs: list[list[float]],
    rule_chains: list[list[str]],
    path: Path,
    id_offset: int,
    verbose: bool,
) -> tuple[list[FirstErrorDistance], list[InferResult]]:
    batch_distances = []
    batch_gen_tripples = []
    for i, (tripple, ids, token_probs, rule_chain) in enumerate(zip(tripples, batch_ids, batch_probs, rule_chains)):
        sample_id = i + id_offset

        rise.RecExpr(tripple.l_str).to_dot(f"{sample_id} left", str(path / f"{sample_id}_left"))
        middle = rise.RecExpr(tripple.tgt_str)
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle"))
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle_t"), transparent=True)
        rise.RecExpr(tripple.r_str).to_dot(f"{sample_id} right", str(path / f"{sample_id}_right"))

        if verbose:
            rank0print("----------")
            rank0print(f"Sample {sample_id}", "blue")
            rank0print("LEFT:", "green")
            rank0print(tripple.r_str)
            rank0print("MIDDLE:", "green")
            rank0print(tripple.tgt_str)
            rank0print("RIGHT:", "green")
            rank0print(tripple.r_str)

        raw_generated_tokens = [vocab.id2token(int(i)) for i in ids if i]
        generated_tokens = split_off_special(raw_generated_tokens, vocab)
        generated = rise.GeneratedRecExpr(generated_tokens, token_probs=token_probs)
        try:
            lowered = generated.lower()
            distance = rise.first_miss_distance(middle, generated)
            batch_distances.append(distance)
            lowered.to_dot(
                f"{sample_id} generated", str(path / f"{sample_id}_generated"), marked_ids=distance.miss_ids()
            )
            lowered.to_dot(
                f"{sample_id} generated",
                str(path / f"{sample_id}_generated_t"),
                marked_ids=distance.miss_ids(),
                transparent=True,
            )
            batch_gen_tripples.append(
                InferResult(tripple.l_str, tripple.r_str, tripple.tgt_str, str(lowered), rule_chain)
            )

            if verbose:
                rank0print("GENERATED:", "green")
                rank0print(lowered)

        except EggshellException as e:
            generated.to_dot(f"{sample_id} generated (damaged)", str(path / f"{sample_id}_generated"))  # type: ignore
            generated.to_dot(
                f"{sample_id} generated (damaged)", str(path / f"{sample_id}_generated_t"), transparent=True
            )
            rank0print("COULD NOT PROPERLY PARSE GENERATED GUIDE.", "red")
            rank0print(e, "red")
            if verbose:
                rank0print("BEST ATTEMPT:", "red")
                rank0print(generated)
                rank0print(f"Used {generated.used_tokens} out of {len(generated_tokens)}", "red")

    return batch_distances, batch_gen_tripples


def print_distance(distances: list[FirstErrorDistance], ds_name: str):
    rank0print(f"\n### AVERAGE DISTANCE IN {ds_name} DATASET ###", "yellow")
    n_hits = sum([d.n_hits for d in distances])
    rank0print(f"Hits: {n_hits}", "yellow")
    n_misses = sum([d.n_misses for d in distances])
    rank0print(f"Misses: {n_misses}", "yellow")
    perfect_matches = sum([1 for d in distances if d.n_misses == 0])
    rank0print(f"Perfect matches: {perfect_matches}", "yellow")

    avg_hit_prob = _avg_prob([d.hit_probabilities() for d in distances if d])
    rank0print(f"Average Hit Probability: {avg_hit_prob}", "yellow")
    avg_miss_prob = _avg_prob([d.miss_probabilities() for d in distances if d])
    rank0print(f"Average Miss Probability: {avg_miss_prob}", "yellow")
    rank0print("\n")


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
