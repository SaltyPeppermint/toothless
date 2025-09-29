from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcps

import tyro

import toothless.inference as infer
from toothless.utils import count_parameters, cleanup_process_group, rank0print, setup_process_group
from toothless.collators import TripleDualCollator
from toothless.data import TripleDataSet
from toothless.model import DualTransformer
from toothless.args import DataArgs, InferArgs, ModelArgs

torch.set_float32_matmul_precision("high")


def fsdp_main(rank: int, world_size: int, infer_args: InferArgs, dataset: TripleDataSet):
    setup_process_group(rank, world_size)
    rank0print("Distributed Network ready")
    torch.cuda.set_device(rank)

    # Load Data
    vocab_size = dataset.tokenizer.get_vocab_size()
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArgs.from_json(f.read())
    with open(Path(infer_args.folder) / "model_args.json", encoding="utf-8") as f:
        model_args = ModelArgs.from_json(f.read())
    assert isinstance(model_args, ModelArgs)
    assert isinstance(data_args, DataArgs)

    eval_folder = Path(infer_args.folder) / "eval"
    eval_folder.mkdir(exist_ok=True, parents=True)

    # Construct Base Model

    model = DualTransformer(model_args, vocab_size)
    rank0print("Base model ready")

    # FSDP model and Mixed Precision Config
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16) if model_args.bf16 else None
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        device_id=rank,
        use_orig_params=True,  # ALLOWS FULL GRAPH CAPTURE
    )

    model_state_dict = dcps.get_model_state_dict(model)
    dcp.load(state_dict=model_state_dict, checkpoint_id=infer_args.folder + f"/weights/{infer_args.model_suffix}")
    dcps.set_model_state_dict(model, model_state_dict)

    model.eval()

    table, total_params = count_parameters(model)
    if infer_args.verbose:
        rank0print(table)
    rank0print(f"Total Parameters: {total_params}")

    rank0print("FSDP Model/Generator loaded to GPU and ready")

    collator = TripleDualCollator(data_args.max_len)

    _, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], torch.Generator().manual_seed(data_args.rng_seed)
    )
    unmod_triple = eval_dataset[0]
    unmod_triple.target_ids = torch.tensor(
        dataset._tokenize(
            "(lam (>> f1 (>> (>> transpose (>> (>> transpose transpose) (>> (>> (>> transpose transpose) transpose) transpose))) (>> (>> transpose transpose) transpose))) (>> (lam f2 (>> (>> (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (app iterateStream (var f4)) (app (app map (lam mfu26 (app (var f3) (app (var f2) (app (var f1) (var mfu26)))))) (var x3)))))))) transpose) (>> transpose (>> (>> (>> transpose (>> (>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose))) (>> (>> transpose (>> transpose transpose)) (>> (>> transpose transpose) (>> transpose transpose))))) (>> transpose transpose)) (>> transpose transpose))))) (>> (>> (>> transpose transpose) transpose) transpose)))"
        ),
        dtype=torch.long,
    )
    triple = [unmod_triple]
    batch = collator(triple)
    result = infer.generate_with_probabilities(model, batch, data_args.max_len)

    processed = infer.batch_process_result(
        dataset.tokenizer, triple, result.tokens.tolist(), result.token_probs.tolist(), 99999, verbose=False
    )
    rank0print(f"---\n{processed}")

    cleanup_process_group()


if __name__ == "__main__":
    infer_args = tyro.cli(InferArgs)
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArgs.from_json(f.read())
    assert isinstance(data_args, DataArgs)

    dataset = TripleDataSet(data_args, tokenizer_path=Path(infer_args.folder) / "tokenizer.json")
    print("Dataset ready")

    world_size = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)
    print("DONE")
