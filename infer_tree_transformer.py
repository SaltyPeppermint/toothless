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
        assert type(data_args) is DataArguments
    with open(Path(infer_args.folder) / "model_args.json", encoding="utf-8") as f:
        model_args = ModelArguments.from_json(f.read())
        assert type(model_args) is ModelArguments

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
    rank0print(rank, "FSDP Model/Generator loaded to GPU and ready")

    data_loader = DictCollator(vocab.pad_token_id, data_args.max_len, data_args.k, vocab)

    # infer.json
    rank0print(rank, "\n=================\nRunning inference on infer_data.json ...")
    with open(infer_args.infer_data, encoding="utf-8") as f:
        tripples = json.load(f)

    batch, _n_tokens = data_loader(tripples)

    batch = {k: v.to(rank) for k, v in batch.items()}
    generator.eval()
    generated_ids = generator(batch)

    rank0print(rank, "Inference done!")
    p = Path("viz/asts/infer_data")
    p.mkdir(parents=True, exist_ok=True)
    pretty_print_result(rank, vocab, tripples, generated_ids, p)

    # Running inference on dataset samples
    rng = torch.Generator().manual_seed(data_args.rng_seed)
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], rng
    )

    rank0print(rank, f"\n=================\nRunning inference on first {infer_args.n_train_data} of train dataset ...")
    tripples = [train_dataset[i] for i in range(0, infer_args.n_train_data)]
    batch, _n_tokens = data_loader(tripples)  # type: ignore
    batch = {k: v.to(rank) for k, v in batch.items()}

    generator.eval()
    generated_ids = generator(batch)

    rank0print(rank, "Inference done!")
    p = Path("viz/asts/train_dataset")
    p.mkdir(parents=True, exist_ok=True)
    pretty_print_result(rank, vocab, tripples, generated_ids, p)  # type: ignore

    rank0print(rank, f"\n=================\nRunning inference on first {infer_args.n_eval_data} of train dataset ...")
    tripples = [eval_dataset[i] for i in range(0, infer_args.n_train_data)]
    batch, _n_tokens = data_loader(tripples)  # type: ignore
    batch = {k: v.to(rank) for k, v in batch.items()}

    generator.eval()
    generated_ids = generator(batch)

    rank0print(rank, "Inference done!")
    p = Path("viz/asts/eval_dataset")
    p.mkdir(parents=True, exist_ok=True)
    pretty_print_result(rank, vocab, tripples, generated_ids, p)  # type: ignore

    cleanup_process_group()


def pretty_print_result(
    rank: int, vocab: SimpleVocab, tripples: list[dict[str, str]], generated_ids: list[Tensor], svg_path: Path
):
    for i, (tripple, generated_id) in enumerate(zip(tripples, generated_ids)):
        rank0print(rank, "----------")
        rank0print(rank, f"Example {i}", "blue")
        rank0print(rank, "START:", "green")
        rank0print(rank, tripple["left"])
        rise.RecExpr(tripple["left"]).to_dot(tripple["left"], str(svg_path / f"{i}_left.svg"))  # type: ignore
        rank0print(rank, "GOAL:", "green")
        rank0print(rank, tripple["middle"])
        rise.RecExpr(tripple["middle"]).to_dot(tripple["middle"], str(svg_path / f"{i}_middle.svg"))  # type: ignore
        rise.RecExpr(tripple["middle"]).to_dot(tripple["middle"], str(svg_path / f"{i}_middle_t.svg"), transparent=True)
        rank0print(rank, "GROUND TRUTH:", "green")
        rank0print(rank, tripple["right"])
        rise.RecExpr(tripple["right"]).to_dot(tripple["right"], str(svg_path / f"{i}_right.svg"))  # type: ignore

        raw_guide_tokens = [vocab.id2token(int(id)) for id in generated_id if id]
        guide_tokens = split_off_special(raw_guide_tokens, vocab)
        guide = rise.PartialRecExpr(guide_tokens)
        if len(guide_tokens) == guide.used_tokens:
            rank0print(rank, "GENERATED GUIDE:", "green")
            rank0print(rank, guide.lower_meta_level())
        else:
            rank0print(rank, "COULD NOT PROPERLY PARSE GENERATED GUIDE. BEST ATTEMPT:", "yellow")
            rank0print(rank, guide)
            rank0print(rank, f"Used {guide.used_tokens} out of {len(guide_tokens)}", "yellow")
        guide.to_dot(str(guide), str(svg_path / f"{i}_generated.svg"))  # type: ignore
        guide.to_dot(str(guide), str(svg_path / f"{i}_generated_t.svg"), transparent=True)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(InferenceArguments)  # type: ignore
    infer_args = parser.parse_args_into_dataclasses()[0]
    world_size = torch.cuda.device_count()
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArguments.from_json(f.read())
        assert type(data_args) is DataArguments
    dataset = CustomDataset(data_args)

    if world_size <= 1:
        fsdp_main(0, world_size, infer_args, dataset)
    else:
        mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)  # type: ignore
    print("\nDONE")
