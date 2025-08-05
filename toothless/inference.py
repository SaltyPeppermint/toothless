from pathlib import Path


from eggshell import FirstErrorDistance, EggshellException
from eggshell import rise  # type: ignore

from toothless.vocab import SimpleVocab
from toothless.utils import rank0print
from toothless.data import split_off_special, Tripple


def batch_process_result(
    rank: int,
    vocab: SimpleVocab,
    tripples: list[Tripple],
    batch_ids: list[list[int]],
    batch_probs: list[list[float]],
    path: Path,
    id_offset: int,
    verbose: bool,
) -> tuple[list[FirstErrorDistance], list[dict[str, str]]]:
    batch_distances = []
    batch_gen_tripples = []
    for i, (tripple, ids, token_probs) in enumerate(zip(tripples, batch_ids, batch_probs)):
        sample_id = i + id_offset

        rise.RecExpr(tripple.l_str).to_dot(f"{sample_id} left", str(path / f"{sample_id}_left"))
        middle = rise.RecExpr(tripple.tgt_str)
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle"))
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle_t"), transparent=True)
        rise.RecExpr(tripple.r_str).to_dot(f"{sample_id} right", str(path / f"{sample_id}_right"))

        if verbose:
            rank0print(rank, "----------")
            rank0print(rank, f"Sample {sample_id}", "blue")
            rank0print(rank, "LEFT:", "green")
            rank0print(rank, tripple.r_str)
            rank0print(rank, "MIDDLE:", "green")
            rank0print(rank, tripple.tgt_str)
            rank0print(rank, "RIGHT:", "green")
            rank0print(rank, tripple.r_str)

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
                {"left": tripple.l_str, "middle": tripple.tgt_str, "right": tripple.r_str, "generated": str(lowered)}
            )

            if verbose:
                rank0print(rank, "GENERATED:", "green")
                rank0print(rank, lowered)

        except EggshellException as e:
            generated.to_dot(f"{sample_id} generated (damaged)", str(path / f"{sample_id}_generated"))  # type: ignore
            generated.to_dot(
                f"{sample_id} generated (damaged)", str(path / f"{sample_id}_generated_t"), transparent=True
            )
            rank0print(rank, "COULD NOT PROPERLY PARSE GENERATED GUIDE.", "red")
            rank0print(rank, e, "red")
            if verbose:
                rank0print(rank, "BEST ATTEMPT:", "red")
                rank0print(rank, generated)
                rank0print(rank, f"Used {generated.used_tokens} out of {len(generated_tokens)}", "red")

    return batch_distances, batch_gen_tripples


def print_distance(rank: int, distances: list[FirstErrorDistance], ds_name: str):
    rank0print(rank, f"\n### AVERAGE DISTANCE IN {ds_name} DATASET ###", "yellow")
    n_hits = sum([d.n_hits for d in distances])
    rank0print(rank, f"Hits: {n_hits}", "yellow")
    n_misses = sum([d.n_misses for d in distances])
    rank0print(rank, f"Misses: {n_misses}", "yellow")
    perfect_matches = sum([1 for d in distances if d.n_misses == 0])
    rank0print(rank, f"Perfect matches: {perfect_matches}", "yellow")

    avg_hit_prob = _avg_prob([d.hit_probabilities() for d in distances if d])
    rank0print(rank, f"Average Hit Probability: {avg_hit_prob}", "yellow")
    avg_miss_prob = _avg_prob([d.miss_probabilities() for d in distances if d])
    rank0print(rank, f"Average Miss Probability: {avg_miss_prob}", "yellow")
    rank0print(rank, "\n")


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
