import json
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp

import transformers

from eggshell import rise  # type: ignore

from toothless.tree_model.vocab import SimpleVocab
from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.tree_model.data import CustomDataset, DictCollator, pyrec_to_tensor, split_off_special
from toothless.tree_model.model import ASTTransformer, GreedyGenerator, count_parameters
from toothless.tree_model.args import DataArguments, InferenceArguments, ModelArguments


def fsdp_main(rank: int, world_size: int, infer_args: InferenceArguments):
    setup_process_group(rank, world_size)
    rank0print(rank, "Distributed Network ready")
    torch.cuda.set_device(rank)

    # Load Data
    vocab = SimpleVocab.load(Path(infer_args.folder) / "vocab.json")
    with open(Path(infer_args.folder) / "data_args.json") as f:
        data_args = DataArguments.from_json(f.read())
        assert type(data_args) is DataArguments
    with open(Path(infer_args.folder) / "model_args.json") as f:
        model_args = ModelArguments.from_json(f.read())
        assert type(model_args) is ModelArguments

    # Construct Base Model
    weights = torch.load(infer_args.folder + f"/tree_transformer_{infer_args.model_suffix}.pt")
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

    data_loader = DictCollator(vocab.pad_token_id, data_args.max_len)

    # INFER JSON
    rank0print(rank, "\n=======================\nRunning inference on infer_data.json ...")
    with open(infer_args.infer_data, encoding="utf-8") as f:
        tripples = json.load(f)
    data, tripple_ids = pp_for_inference(vocab, data_args.k, tripples)

    batch, _n_tokens = data_loader(data)

    batch = {k: v.to(rank) for k, v in batch.items()}
    generator.eval()
    generated_ids = generator(batch)

    rank0print(rank, "Inference done!")
    pretty_print_result(rank, vocab, tripples, tripple_ids, generated_ids, infer_args.verbose)

    # INFER DATASET
    first_n = 4
    rank0print(rank, f"\n=======================\nRunning inference on first {first_n} of dataset ...")

    tripples = [dataset.get_sample_json(i) for i in range(0, first_n)]
    data, tripple_ids = pp_for_inference(vocab, data_args.k, tripples)

    batch, _n_tokens = data_loader(data)
    batch = {k: v.to(rank) for k, v in batch.items()}

    generator.eval()
    generated_ids = generator(batch)

    rank0print(rank, "Inference done!")
    pretty_print_result(rank, vocab, tripples, tripple_ids, generated_ids, infer_args.verbose)

    cleanup_process_group()


def pp_for_inference(
    vocab: SimpleVocab, k: int, tripples: list[dict[str, str]]
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    data = []
    tripple_ids = []
    for tripple in tripples:
        l_ids, l_anc, l_sib = pyrec_to_tensor(rise.PyRecExpr(tripple["left"]), vocab, k)
        r_ids, r_anc, r_sib = pyrec_to_tensor(rise.PyRecExpr(tripple["middle"]), vocab, k)
        tgt_ids, _, _ = pyrec_to_tensor(rise.PyRecExpr(tripple["right"]), vocab, k)
        tripple_ids.append({"l_ids": l_ids, "tgt_ids": tgt_ids, "r_ids": r_ids})
        data.append(
            {
                "l_ids": l_ids,
                "l_anc": l_anc,
                "l_sib": l_sib,
                "r_ids": r_ids,
                "r_anc": r_anc,
                "r_sib": r_sib,
            }
        )

    return data, tripple_ids


def pretty_print_result(
    rank: int,
    vocab: SimpleVocab,
    tripples: list[dict[str, str]],
    tripple_ids: list[dict[str, Tensor]],
    generated_ids: list[Tensor],
    verbose: bool,
):
    for i, (tripple, tripple_id, generated_id) in enumerate(zip(tripples, tripple_ids, generated_ids)):
        rank0print(rank, f"----------\nExample {i}")
        rank0print(rank, f"START:\n{tripple['left']}")
        if verbose:
            l_tokens = [vocab.id2token(int(id)) for id in tripple_id["l_ids"]]
            rank0print(rank, l_tokens)

        rank0print(rank, f"GOAL:\n{tripple['middle']}")
        if verbose:
            r_tokens = [vocab.id2token(int(id)) for id in tripple_id["r_ids"]]
            rank0print(rank, r_tokens)

        rank0print(rank, f"GROUND TRUTH:\n{tripple['right']}")
        if verbose:
            ground_truth_tokens = [vocab.id2token(int(id)) for id in tripple_id["tgt_ids"]]
            rank0print(rank, ground_truth_tokens)

        raw_guide_tokens = [vocab.id2token(int(id)) for id in generated_id if id]
        guide_tokens = split_off_special(raw_guide_tokens, vocab)
        if len(guide_tokens) == rise.count_expected_tokens(guide_tokens):
            guide_s_expr = rise.lower_meta_level([tok for tok in guide_tokens])
            rank0print(rank, f"GENERATED GUIDE:\n{guide_s_expr}")
        else:
            rank0print(rank, f"COULD NOT PROPERLY DISPLAY GENERATED GUIDE:\n{raw_guide_tokens}")

        if verbose:
            rank0print(rank, guide_tokens)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(InferenceArguments)  # type: ignore
    infer_args = parser.parse_args_into_dataclasses()[0]
    world_size = torch.cuda.device_count()
    with open(Path(infer_args.folder) / "data_args.json") as f:
        data_args = DataArguments.from_json(f.read())
        assert type(data_args) is DataArguments
    dataset = CustomDataset(data_args)

    mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)  # type: ignore
    print("\nDONE")
