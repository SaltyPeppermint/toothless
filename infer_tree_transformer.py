import json
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp

import transformers

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
    pretty_print_result(rank, vocab, tripples, batch_ids, batch_probs, p)
    del batch_ids, batch_probs, batch

    # Running inference on dataset samples
    rng = torch.Generator().manual_seed(data_args.rng_seed)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], rng
    )

    rank0print(rank, f"\n=================\nRunning inference on first {infer_args.n_train_data} of train dataset ...")
    tripples = [train_dataset[i] for i in range(0, infer_args.n_train_data)]
    batch, _n_tokens = data_loader(tripples)  # type: ignore
    batch = {k: v.to(rank) for k, v in batch.items()}
    batch_ids, batch_probs = generator(batch)

    p = Path("viz/asts/train_dataset")
    p.mkdir(parents=True, exist_ok=True)
    pretty_print_result(rank, vocab, tripples, batch_ids, batch_probs, p)  # type: ignore
    del batch_ids, batch_probs, batch

    rank0print(rank, f"\n=================\nRunning inference on first {infer_args.n_eval_data} of train dataset ...")
    tripples = [eval_dataset[i] for i in range(0, infer_args.n_train_data)]
    batch, _n_tokens = data_loader(tripples)  # type: ignore
    batch = {k: v.to(rank) for k, v in batch.items()}
    batch_ids, batch_probs = generator(batch)

    p = Path("viz/asts/eval_dataset")
    p.mkdir(parents=True, exist_ok=True)
    pretty_print_result(rank, vocab, tripples, batch_ids, batch_probs, p)  # type: ignore
    del batch_ids, batch_probs, batch

    cleanup_process_group()


def pretty_print_result(
    rank: int,
    vocab: SimpleVocab,
    tripples: list[dict[str, str]],
    batch_ids: list[Tensor],
    batch_probs: list[Tensor],
    path: Path,
):
    for i, (tripple, ids, token_probs) in enumerate(zip(tripples, batch_ids, batch_probs)):
        rank0print(rank, "----------")
        rank0print(rank, f"Example {i}", "blue")
        rank0print(rank, "START:", "green")
        rank0print(rank, tripple["left"])
        rise.RecExpr(tripple["left"]).to_dot(tripple["left"], str(path / f"{i}_left.svg"))  # type: ignore
        rank0print(rank, "GOAL:", "green")
        rank0print(rank, tripple["middle"])
        ground_truth = rise.RecExpr(tripple["middle"])
        ground_truth.to_dot(tripple["middle"], str(path / f"{i}_middle.svg"))  # type: ignore
        ground_truth.to_dot(tripple["middle"], str(path / f"{i}_middle_t.svg"), transparent=True)
        rank0print(rank, "GROUND TRUTH:", "green")
        rank0print(rank, tripple["right"])
        rise.RecExpr(tripple["right"]).to_dot(tripple["right"], str(path / f"{i}_right.svg"))  # type: ignore

        raw_generated_tokens = [vocab.id2token(int(id)) for id in ids if id]
        generated_tokens = split_off_special(raw_generated_tokens, vocab)
        generated = rise.GeneratedRecExpr(generated_tokens, token_probs=token_probs.tolist())
        if len(generated_tokens) == generated.used_tokens:
            rank0print(rank, "GENERATED GUIDE:", "green")
            lowered = generated.lower()
            distance = rise.first_miss_distance(ground_truth, generated)
            rank0print(rank, lowered)
            generated.to_dot(str(generated), str(path / f"{i}_generated"), marked_ids=distance.misses)  # type: ignore #marked_ids=distance.misses
            generated.to_dot(
                str(generated), str(path / f"{i}_generated_t"), marked_ids=distance.misses, transparent=True
            )

        else:
            rank0print(rank, "COULD NOT PROPERLY PARSE GENERATED GUIDE. BEST ATTEMPT:", "yellow")
            rank0print(rank, generated)
            rank0print(rank, f"Used {generated.used_tokens} out of {len(generated_tokens)}", "yellow")
            generated.to_dot(str(generated), str(path / f"{i}_generated"))  # type: ignore
            generated.to_dot(str(generated), str(path / f"{i}_generated_t"), transparent=True)


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
