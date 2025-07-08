import json
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp

import transformers

from eggshell import FirstErrorDistance
from eggshell import rise  # type: ignore


from toothless.vocab import SimpleVocab
from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.data import CustomDataset, DictCollator, split_off_special
from toothless.model import ASTTransformer, GreedyGenerator, count_parameters
from toothless.args import DataArguments, InferenceArguments, ModelArguments


def fsdp_main(rank: int, world_size: int, infer_args: InferenceArguments, dataset: CustomDataset):
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
    model = ASTTransformer(model_args, len(vocab), len(vocab), data_args.k, state_dict=weights)
    model.eval()
    generator = GreedyGenerator(model, data_args.max_len, vocab, data_args.k)
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

    data_loader = DictCollator(vocab.pad_token_id, data_args.max_len, data_args.k, vocab)

    # infer.json
    rank0print(rank, "\n=================\nRunning inference on infer_data.json ...")
    with open(infer_args.infer_data, encoding="utf-8") as f:
        tripples = json.load(f)

    batch, _n_tokens = data_loader(tripples)
    batch = {k: v.to(rank) for k, v in batch.items()}
    batch_ids, batch_probs = generator(batch)

    p = Path("viz/asts/infer_data")
    p.mkdir(parents=True, exist_ok=True)
    batch_process_result(rank, vocab, tripples, batch_ids, batch_probs, p, 0)
    del batch_ids, batch_probs, batch

    # Running inference on dataset samples
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], torch.Generator().manual_seed(data_args.rng_seed)
    )

    n = infer_args.n_train_data if infer_args.n_train_data else len(train_dataset)
    batch_infer(rank, n, infer_args.batch_size, vocab, generator, data_loader, train_dataset, "train")

    n = infer_args.n_eval_data if infer_args.n_eval_data else len(eval_dataset)
    batch_infer(rank, n, infer_args.batch_size, vocab, generator, data_loader, eval_dataset, "eval")

    cleanup_process_group()


def batch_infer(rank, n, batch_size, vocab, generator, data_loader, dataset, ds_name):
    rank0print(rank, f"\n=================\nRunning inference on {n} samples of {ds_name} dataset ...")
    avg_distance = FirstErrorDistance()
    for i in range(0, n, batch_size):
        tripples = [dataset[i] for i in range(i, i + batch_size)]
        batch, _n_tokens = data_loader(tripples)
        batch = {k: v.to(rank) for k, v in batch.items()}
        batch_ids, batch_probs = generator(batch)

        p = Path(f"viz/asts/{ds_name}_dataset")
        p.mkdir(parents=True, exist_ok=True)
        batch_distance = batch_process_result(rank, vocab, tripples, batch_ids, batch_probs, p, i)
        avg_distance.combine(batch_distance)
        del batch_ids, batch_probs, batch

    rank0print(rank, "\n### AVERAGE DISTANCE IN DATASET ###", "yellow")
    rank0print(rank, f"Hits: {len(avg_distance.hits)}", "yellow")
    rank0print(rank, f"Misses: {len(avg_distance.misses)}", "yellow")
    avg_hit_prob = _avg_prob(avg_distance.hit_probabilities)
    rank0print(rank, f"Average Hit Probability: {avg_hit_prob}", "yellow")
    avg_miss_prob = _avg_prob(avg_distance.miss_probabilities)
    rank0print(rank, f"Average Miss Probability: {avg_miss_prob}", "yellow")
    rank0print(rank, "\n")


def _avg_prob(probs: list[float | None]):
    avg_prob = 0
    not_none = 0
    for i in probs:
        if i is not None:
            avg_prob += i
            not_none += 1
    avg_prob = avg_prob / not_none
    return avg_prob


def batch_process_result(
    rank: int,
    vocab: SimpleVocab,
    tripples: list[dict[str, str]],
    batch_ids: list[Tensor],
    batch_probs: list[Tensor],
    path: Path,
    id_offset: int,
) -> FirstErrorDistance:
    batch_distance = FirstErrorDistance()
    for i, (tripple, ids, token_probs) in enumerate(zip(tripples, batch_ids, batch_probs)):
        id = i + id_offset
        rank0print(rank, "----------")
        rank0print(rank, f"Sample {id}", "blue")
        rank0print(rank, "LEFT:", "green")
        rank0print(rank, tripple["left"])
        rise.RecExpr(tripple["left"]).to_dot(f"{id} left", str(path / f"{id}_left"))
        rank0print(rank, "MIDDLE:", "green")
        rank0print(rank, tripple["middle"])
        middle = rise.RecExpr(tripple["middle"])
        middle.to_dot(f"{id} middle", str(path / f"{id}_middle"))
        middle.to_dot(f"{id} middle", str(path / f"{id}_middle_t"), transparent=True)
        rank0print(rank, "RIGHT:", "green")
        rank0print(rank, tripple["right"])
        rise.RecExpr(tripple["right"]).to_dot(f"{i} right", str(path / f"{i}_right"))

        raw_generated_tokens = [vocab.id2token(int(id)) for id in ids if id]
        generated_tokens = split_off_special(raw_generated_tokens, vocab)
        generated = rise.GeneratedRecExpr(generated_tokens, token_probs=token_probs.tolist())
        if len(generated_tokens) == generated.used_tokens:
            rank0print(rank, "GENERATED:", "green")
            lowered = generated.lower()
            distance = rise.first_miss_distance(middle, generated)
            batch_distance.combine(distance)
            rank0print(rank, lowered)
            lowered.to_dot(f"{i} generated", str(path / f"{i}_generated"), marked_ids=distance.misses)
            lowered.to_dot(
                f"{i} generated", str(path / f"{i}_generated_t"), marked_ids=distance.misses, transparent=True
            )

        else:
            rank0print(rank, "COULD NOT PROPERLY PARSE GENERATED GUIDE. BEST ATTEMPT:", "red")
            rank0print(rank, generated)
            rank0print(rank, f"Used {generated.used_tokens} out of {len(generated_tokens)}", "red")
            generated.to_dot(f"{i} generated", str(path / f"{i}_generated"))  # type: ignore
            generated.to_dot(f"{i} generated", str(path / f"{i}_generated_t"), transparent=True)

    return batch_distance


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(InferenceArguments)  # type: ignore
    infer_args = parser.parse_args_into_dataclasses()[0]
    assert isinstance(infer_args, InferenceArguments)
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArguments.from_json(f.read())
    assert isinstance(data_args, DataArguments)
    dataset = CustomDataset(data_args)

    world_size = torch.cuda.device_count()

    if world_size <= 1:
        fsdp_main(0, world_size, infer_args, dataset)
    else:
        mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)  # type: ignore
    print("\nDONE")
