import json
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp

import transformers

from eggshell import rise  # type: ignore

from toothless.tree_model.vocab import SimpleVocab
from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.tree_model.data import DictCollator, pyrec_to_tensor  # , CustomDataset
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
    weights = torch.load(infer_args.folder + "/" + "tree_transformer.pt")
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

    with open(infer_args.infer_data, encoding="utf-8") as f:
        pairs = json.load(f)

    data = []
    ground_truths = []
    for pair in pairs:
        l_ids, l_anc, l_sib = pyrec_to_tensor(rise.PyRecExpr(pair["start"]), vocab, data_args.k)
        r_ids, r_anc, r_sib = pyrec_to_tensor(rise.PyRecExpr(pair["goal"]), vocab, data_args.k)
        guide_ids, _, _ = pyrec_to_tensor(rise.PyRecExpr(pair["guide"]), vocab, data_args.k)
        ground_truths.append(guide_ids)
        data.append(
            {
                "l_ids": l_ids,
                "l_anc": l_anc,
                "l_sib": l_sib,
                "r_ids": r_ids,
                "r_anc": r_anc,
                "r_sib": r_sib,
                "distance": torch.tensor(0),
            }
        )

    data_loader = DictCollator(vocab.pad_token_id, data_args.max_len)

    rank0print(rank, f"Data of batch size {len(data)}")
    batch, _n_tokens = data_loader(data)
    # dataset = CustomDataset(data_args, rank)
    # batch, _n_tokens= data_loader([dataset[0], dataset[1]])

    rank0print(rank, "Data ready!\nNow running autoregressive inference...")

    batch = {k: v.to(rank) for k, v in batch.items()}
    generator.eval()

    tgt_ids = generator(batch)

    rank0print(rank, "Inference done!\n----------")
    for i, (entry) in enumerate(tgt_ids):
        rank0print(rank, f"Example {i}")
        rank0print(rank, f"\nSTART:\n{pairs[i]['start']}")
        if infer_args.verbose:
            start_tokens = [vocab.id2token(int(id)) for id in batch["l_ids"][i]]
            rank0print(rank, start_tokens)

        rank0print(rank, f"\nGOAL:\n{pairs[i]['goal']}")
        if infer_args.verbose:
            goal_tokens = [vocab.id2token(int(id)) for id in batch["r_ids"][i]]
            rank0print(rank, goal_tokens)

        rank0print(rank, f"\nGROUND TRUTH:\n{pairs[i]['guide']}")
        if infer_args.verbose:
            ground_truth_tokens = [vocab.id2token(int(id)) for id in ground_truths[i]]
            rank0print(rank, ground_truth_tokens)

        guide_tokens = [vocab.id2token(int(id)) for id in entry]
        rank0print(rank, f"\nGENERATED GUIDE:\n{rise.lower_meta_level(guide_tokens)}")
        if infer_args.verbose:
            rank0print(rank, guide_tokens)

        rank0print(rank, "----------")

    cleanup_process_group()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(InferenceArguments)  # type: ignore
    infer_args = parser.parse_args_into_dataclasses()[0]
    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, infer_args), nprocs=world_size, join=True)  # type: ignore
    print("\nDONE")
