import json
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp
from torch.utils.data import Dataset

import tyro
from tqdm.auto import tqdm

from eggshell import FirstErrorDistance, EggshellException
from eggshell import rise  # type: ignore

from toothless.vocab import SimpleVocab
from toothless.utils import cleanup_process_group, rank0print, setup_process_group
from toothless.collators import DisentangledDictCollator
from toothless.data import TrippleDataSet, split_off_special
from toothless.models.disentangled import DisentangledDualTreeTransformer, DisentangledGreedyGenerator
from toothless.models.utils import count_parameters
from toothless.args import DataArguments, InferenceArguments, ModelArguments


def fsdp_main(rank: int, world_size: int, infer_args: InferenceArguments, dataset: TrippleDataSet):
    setup_process_group(rank, world_size)
    rank0print(rank, "Distributed Network ready")
    torch.cuda.set_device(rank)

    # Load Data
    vocab = SimpleVocab.load(Path(infer_args.folder) / "vocab.json")
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArguments.from_json(f.read())
    with open(Path(infer_args.folder) / "model_args.json", encoding="utf-8") as f:
        model_args = ModelArguments.from_json(f.read())
    assert isinstance(model_args, ModelArguments)
    assert isinstance(data_args, DataArguments)

    # Construct Base Model
    weights = torch.load(infer_args.folder + f"/tree_transformer{infer_args.model_suffix}.pt")
    model = DisentangledDualTreeTransformer(model_args, len(vocab), len(vocab), data_args.k, state_dict=weights)
    model.eval()
    generator = DisentangledGreedyGenerator(model, data_args.max_len, vocab, data_args.k)
    rank0print(rank, "Base Model and Generator ready")

    table, total_params = count_parameters(model)
    if infer_args.verbose:
        rank0print(rank, table)
    rank0print(rank, f"Total Parameters: {total_params}")

    # FSDP model and Mixed Precision Config
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if infer_args.bf16 else None
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    generator = FSDP(generator, sharding_strategy=sharding_strategy, mixed_precision=mixed_precision, device_id=rank)
    generator.eval()

    rank0print(rank, "FSDP Model/Generator loaded to GPU and ready")

    data_loader = DisentangledDictCollator(vocab.pad_token_id, data_args.max_len, data_args.k, vocab)

    # infer.json
    rank0print(rank, "\n=================\nRunning inference on infer_data.json ...")
    with open(infer_args.infer_data, encoding="utf-8") as f:
        tripples = json.load(f)

    batch, _n_tokens = data_loader(tripples)
    batch = {k: v.to(rank) for k, v in batch.items()}
    batch_ids, batch_probs = generator(batch)

    p = Path("viz/asts/infer_data")
    p.mkdir(parents=True, exist_ok=True)
    _batch_process_result(rank, vocab, tripples, batch_ids, batch_probs, p, 0, infer_args.verbose)
    del batch_ids, batch_probs, batch

    # Running inference on dataset samples
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], torch.Generator().manual_seed(data_args.rng_seed)
    )

    n = infer_args.n_train_data if infer_args.n_train_data else len(train_dataset)
    train_distances, train_gen_tripples = _batch_infer(
        rank,
        n,
        data_args.sample_distance,
        infer_args.batch_size,
        vocab,
        generator,
        data_loader,
        train_dataset,
        "train",
        infer_args.verbose,
    )
    with open("train_gen_tripples.json", mode="w", encoding="utf-8") as f:
        json.dump(train_gen_tripples, f)
    del train_gen_tripples
    _print_distance(rank, train_distances, "TRAIN")

    n = infer_args.n_eval_data if infer_args.n_eval_data else len(eval_dataset)
    eval_distances, eval_gen_tripples = _batch_infer(
        rank,
        n,
        data_args.sample_distance,
        infer_args.batch_size,
        vocab,
        generator,
        data_loader,
        eval_dataset,
        "eval",
        infer_args.verbose,
    )

    with open("eval_gen_tripples.json", mode="w", encoding="utf-8") as f:
        json.dump(eval_gen_tripples, f)
    del eval_gen_tripples

    _print_distance(rank, eval_distances, "EVAL")

    cleanup_process_group()


def _batch_infer(
    rank: int,
    n: int,
    sample_distance: int,
    batch_size: int,
    vocab: SimpleVocab,
    generator: FSDP,
    data_loader: DisentangledDictCollator,
    dataset: Dataset,
    ds_name: str,
    verbose: bool,
) -> tuple[list[FirstErrorDistance], list[dict[str, str]]]:
    rank0print(rank, f"\n=================\nRunning inference on {n} samples of {ds_name} dataset ...")
    distances = []
    gen_tripples = []

    for i in tqdm(range(0, n, batch_size), desc=f"Inference Batch (Batch Size {batch_size})"):
        tripples = [dataset[i] for i in range(i, i + batch_size)]
        batch, _n_tokens = data_loader(tripples)
        batch = {k: v.to(rank) for k, v in batch.items()}
        batch_ids, batch_probs = generator(batch)

        p = Path(f"viz/asts/d{sample_distance}/{ds_name}_dataset")
        p.mkdir(parents=True, exist_ok=True)
        batch_distance, batch_gen_tripples = _batch_process_result(
            rank, vocab, tripples, batch_ids, batch_probs, p, i, verbose
        )
        distances.extend(batch_distance)
        gen_tripples.extend(batch_gen_tripples)
        del batch_ids, batch_probs, batch

    return distances, gen_tripples


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


def _print_distance(rank: int, distances: list[FirstErrorDistance], ds_name: str):
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


def _batch_process_result(
    rank: int,
    vocab: SimpleVocab,
    tripples: list[dict[str, str]],
    batch_ids: list[Tensor],
    batch_probs: list[Tensor],
    path: Path,
    id_offset: int,
    verbose: bool,
) -> tuple[list[FirstErrorDistance], list[dict[str, str]]]:
    batch_distances = []
    batch_gen_tripples = []
    for i, (tripple, ids, token_probs) in enumerate(zip(tripples, batch_ids, batch_probs)):
        sample_id = i + id_offset

        rise.RecExpr(tripple["left"]).to_dot(f"{sample_id} left", str(path / f"{sample_id}_left"))
        middle = rise.RecExpr(tripple["middle"])
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle"))
        middle.to_dot(f"{sample_id} middle", str(path / f"{sample_id}_middle_t"), transparent=True)
        rise.RecExpr(tripple["right"]).to_dot(f"{sample_id} right", str(path / f"{sample_id}_right"))

        if verbose:
            rank0print(rank, "----------")
            rank0print(rank, f"Sample {sample_id}", "blue")
            rank0print(rank, "LEFT:", "green")
            rank0print(rank, tripple["left"])
            rank0print(rank, "MIDDLE:", "green")
            rank0print(rank, tripple["middle"])
            rank0print(rank, "RIGHT:", "green")
            rank0print(rank, tripple["right"])

        raw_generated_tokens = [vocab.id2token(int(i)) for i in ids if i]
        generated_tokens = split_off_special(raw_generated_tokens, vocab)
        generated = rise.GeneratedRecExpr(generated_tokens, token_probs=token_probs.tolist())
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
            tripple["generated"] = str(lowered)
            batch_gen_tripples.append(tripple)

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


if __name__ == "__main__":
    infer_args = tyro.cli(InferenceArguments)
    print(vars(infer_args))
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArguments.from_json(f.read())
    assert isinstance(data_args, DataArguments)
    dataset = TrippleDataSet(data_args, True)

    world_size = torch.cuda.device_count()

    if world_size <= 1:
        fsdp_main(0, world_size, infer_args, dataset)
    else:
        mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)  # type: ignore
    print("\nDONE")
