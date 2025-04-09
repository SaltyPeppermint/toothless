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
from toothless.tree_model.model import ASTTransformer, GreedyGenerator
from toothless.tree_model.args import DataArguments, InferenceArguments, ModelArguments


def fsdp_main(
    rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, infer_args: InferenceArguments
):
    setup_process_group(rank, world_size)

    rank0print(rank, "Distributed Network ready")

    torch.cuda.set_device(rank)

    # Load Data

    vocab = SimpleVocab.load(Path(infer_args.vocab_path))
    rank0print(rank, "DataLoaders ready")

    # Construct Base Model
    weights = torch.load(infer_args.weights_path)
    model = ASTTransformer(model_args, len(vocab), len(vocab), data_args.k, state_dict=weights)
    model.eval()
    generator = GreedyGenerator(model, data_args.max_len, vocab, data_args.k)
    rank0print(rank, "Base Model and Generator ready")

    # FSDP model and Mixed Precision Config
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if infer_args.bf16 else None
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    generator = FSDP(generator, sharding_strategy=sharding_strategy, mixed_precision=mixed_precision, device_id=rank)
    rank0print(rank, "FSDP Model/Generator loaded to GPU and ready")

    with open(infer_args.infer_data, encoding="utf-8") as f:
        pairs = json.load(f)

    data = []
    for pair in pairs:
        l_ids, l_anc, l_sib = pyrec_to_tensor(rise.PyRecExpr(pair["left"]), vocab, data_args.k)
        r_ids, r_anc, r_sib = pyrec_to_tensor(rise.PyRecExpr(pair["right"]), vocab, data_args.k)
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

    rank0print(rank, "Inference done!\n")
    for i, entry in enumerate(tgt_ids):
        rank0print(rank, f"RESULT: {i}")
        start = [vocab.id2token(int(id)) for id in batch["l_ids"][i]]
        rank0print(rank, f"START: {start}")
        guide = [vocab.id2token(int(id)) for id in entry]
        rank0print(rank, f"GUIDE {guide}")
        end = [vocab.id2token(int(id)) for id in batch["r_ids"][i]]
        rank0print(rank, f"END: {end}")
        rank0print(rank, "---")

    cleanup_process_group()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))  # type: ignore
    (
        model_args,
        data_args,
        infer_args,
    ) = parser.parse_args_into_dataclasses()
    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, model_args, data_args, infer_args), nprocs=world_size, join=True)  # type: ignore
    print("DONE")
