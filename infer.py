from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
import torch.multiprocessing as mp
from torch.utils.data import Subset
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcps

import tyro
from tqdm.auto import tqdm
from tokenizers import Tokenizer

from eggshell import FirstErrorDistance

import toothless.inference as infer
from toothless.utils import cleanup_process_group, rank0print, setup_process_group
from toothless.collators import DictCollator
from toothless.data import TripleDataSet, Triple
from toothless.model import DualTreeTransformer, generate_with_probabilities
from toothless.utils import count_parameters
from toothless.args import DataArgs, InferArgs, ModelArgs

torch.set_float32_matmul_precision("high")


def fsdp_main(rank: int, world_size: int, infer_args: InferArgs, dataset: TripleDataSet):
    setup_process_group(rank, world_size)
    rank0print("Distributed Network ready")
    torch.cuda.set_device(rank)

    pad_token_id = dataset.tokenizer.token_to_id("<PAD>")
    assert pad_token_id == 0

    # Load Data
    vocab = Tokenizer.from_file(str(Path(infer_args.folder) / "vocab.json"))
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArgs.from_json(f.read())
    with open(Path(infer_args.folder) / "model_args.json", encoding="utf-8") as f:
        model_args = ModelArgs.from_json(f.read())
    assert isinstance(model_args, ModelArgs)
    assert isinstance(data_args, DataArgs)

    eval_folder = Path(infer_args.folder) / "eval"
    eval_folder.mkdir(exist_ok=True, parents=True)

    # Construct Base Model

    model = DualTreeTransformer(model_args, len(vocab), len(vocab), pad_token_id=pad_token_id)
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

    model.eval()

    table, total_params = count_parameters(model)
    if infer_args.verbose:
        rank0print(table)
    rank0print(f"Total Parameters: {total_params}")

    rank0print("FSDP Model/Generator loaded to GPU and ready")

    collator = DictCollator(data_args.max_len, pad_token_id=pad_token_id)

    # Running inference on dataset samples
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], torch.Generator().manual_seed(data_args.rng_seed)
    )

    train_distances, train_gen_triples = _batch_infer(
        data_args, infer_args, dataset.tokenizer, model, collator, train_dataset, "train", eval_folder
    )
    with open(eval_folder / "train_gen_triples_vanilla.json", mode="w", encoding="utf-8") as f:
        f.write(infer.InferResult.list_to_json(train_gen_triples))
    del train_gen_triples
    infer.print_distance(train_distances, "TRAIN")

    eval_distances, eval_gen_triples = _batch_infer(
        data_args, infer_args, dataset.tokenizer, model, collator, eval_dataset, "eval", eval_folder
    )
    with open(eval_folder / "eval_gen_triples_vanilla.json", mode="w", encoding="utf-8") as f:
        f.write(infer.InferResult.list_to_json(eval_gen_triples))
    del eval_gen_triples
    infer.print_distance(eval_distances, "EVAL")

    cleanup_process_group()


def _batch_infer(
    data_args: DataArgs,
    infer_args: InferArgs,
    tokenizer: Tokenizer,
    model: FSDP,
    collator: DictCollator,
    dataset: Subset[Triple],
    ds_name: str,
    eval_folder: Path,
) -> tuple[list[FirstErrorDistance], list[infer.InferResult]]:
    if ds_name == "eval":
        n = infer_args.n_eval_data if infer_args.n_eval_data else len(dataset)
    elif ds_name == "train":
        n = infer_args.n_train_data if infer_args.n_train_data else len(dataset)
    else:
        raise ValueError("Unknown Dataset name")

    rank0print(f"\n=================\nRunning inference on {n} samples of {ds_name} dataset ...")
    distances = []
    gen_triples = []

    n = infer_args.n_eval_data if infer_args.n_eval_data else len(dataset)

    for i in tqdm(range(0, n, infer_args.batch_size), desc=f"Inference Batch (Batch Size {infer_args.batch_size})"):
        triples = [dataset[i] for i in range(i, i + infer_args.batch_size)]
        batch, _n_tokens = collator(triples)
        result = generate_with_probabilities(model, batch["l_ids"], batch["r_ids"], tokenizer, data_args.max_len)

        p = Path(eval_folder / "viz/asts/")
        p.mkdir(parents=True, exist_ok=True)
        batch_distance, batch_gen_triples = infer.batch_process_result(
            tokenizer, triples, result.tokens.tolist(), result.token_probs.tolist(), p, i, infer_args.verbose
        )
        distances.extend(batch_distance)
        gen_triples.extend(batch_gen_triples)
        del batch, result

    return distances, gen_triples


if __name__ == "__main__":
    infer_args = tyro.cli(InferArgs)
    with open(Path(infer_args.folder) / "data_args.json", encoding="utf-8") as f:
        data_args = DataArgs.from_json(f.read())
    assert isinstance(data_args, DataArgs)
    dataset = TripleDataSet(data_args)

    world_size = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(world_size, infer_args, dataset), nprocs=world_size, join=True)
