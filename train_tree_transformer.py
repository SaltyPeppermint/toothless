import copy
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm.auto import tqdm
import transformers

from toothless.tree_model.vocab import SimpleVocab
from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.tree_model.data import CustomDataset, mk_loaders
from toothless.tree_model.model import ASTTransformer, count_parameters
from toothless.tree_model.args import DataArguments, TrainingArguments, ModelArguments


def fsdp_main(
    rank: int,
    world_size: int,
    model_args: ModelArguments,
    data_args: DataArguments,
    train_args: TrainingArguments,
    dataset: CustomDataset,
):
    start_time = datetime.now()
    setup_process_group(rank, world_size)

    writer = SummaryWriter() if rank == 0 else None

    torch.cuda.set_device(rank)

    # Load Data
    vocab_size = len(dataset.vocab)
    train_dataloader, eval_dataloader = mk_loaders(rank, world_size, dataset, data_args)
    rank0print(rank, "DataLoaders ready")

    # Construct Base Model
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ASTTransformer(model_args, vocab_size, vocab_size, data_args.k)

    if writer and train_args.trace:
        example_batch, _ = next(iter(copy.deepcopy(train_dataloader)))
        writer.add_graph(model, example_batch)

    table, total_params = count_parameters(model)
    rank0print(rank, table)
    rank0print(rank, f"Total Parameters: {total_params}")

    # FSDP model and Mixed Precision Config
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True) if train_args.bf16 else None
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    model = FSDP(model, sharding_strategy=sharding_strategy, mixed_precision=mixed_precision, device_id=rank)

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.weight_decay,
    )
    total_steps = train_args.epochs * len(train_dataloader)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,  # Start from 1% of max_lr
        end_factor=1.0,
        total_iters=train_args.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - train_args.warmup_steps,  # Cosine After Warmup
        eta_min=train_args.min_lr,
    )

    criterion = CrossEntropyLoss(ignore_index=dataset.vocab.pad_token_id, label_smoothing=0.1)

    rank0print(rank, "Starting training!")
    init_start_event.record(torch.cuda.current_stream())

    for epoch in range(train_args.epochs):
        train(
            rank,
            model,
            copy.deepcopy(train_dataloader),
            criterion,
            optimizer,
            warmup_scheduler,
            cosine_scheduler,
            epoch,
            train_args,
            writer,
        )

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(rank, model, copy.deepcopy(eval_dataloader), criterion, epoch, train_args.epochs, writer)

        cosine_scheduler.step()

    init_end_event.record(torch.cuda.current_stream())
    rank0print(rank, "Training finished!")

    if rank == 0:
        init_end_event.synchronize()
    rank0print(rank, f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000} sec")

    if train_args.save_model_end:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            states = model.state_dict()
            if rank == 0:
                save(model_args, data_args, train_args, dataset.vocab, states, start_time)

    cleanup_process_group()


def train(
    rank: int,
    model: FSDP,
    dataloader: DataLoader[dict[str, Tensor]],
    criterion: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    warmup_scheduler: LinearLR,
    cosine_scheduler: CosineAnnealingLR,
    epoch: int,
    train_args: TrainingArguments,
    writer: SummaryWriter | None = None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    for batch_idx, (batch, num_tokens) in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{train_args.epochs}")
    ):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}

        # Forward pass
        out = model(batch)
        loss = criterion(out.view(-1, out.size(-1)), batch["tgt_ids_y"].view(-1))

        # Backwards pass
        loss.backward()
        optimizer.step()
        # Dont forget...
        optimizer.zero_grad()

        # LR Scheduling
        if batch_idx + epoch * len(dataloader) < train_args.warmup_steps:
            warmup_scheduler.step()  # Linear warmup
            last_lr = warmup_scheduler.get_last_lr()
        else:
            cosine_scheduler.step()  # Cosine decay
            last_lr = cosine_scheduler.get_last_lr()

        if writer:
            writer.add_scalar("Train Loss/batch", loss, batch_idx + epoch * len(dataloader))
            writer.add_scalar("Train LR/batch", last_lr[-1], batch_idx + epoch * len(dataloader))
            writer.add_scalar("Toks/in-batch", num_tokens, batch_idx + epoch * len(dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Train Loss/epoch", train_loss, epoch + 1)

    rank0print(rank, f"Epoch: {epoch + 1}/{train_args.epochs} \tTrain Loss: {train_loss:.6f}")


def eval(
    rank: int,
    model: nn.Module,
    dataloader: DataLoader[dict[str, Tensor]],
    criterion: CrossEntropyLoss,
    epoch: int,
    max_epochs: int,
    writer: SummaryWriter | None = None,
):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)
    for batch, _num_tokens in dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
            loss = criterion(out.view(-1, out.size(-1)), batch["tgt_ids_y"].view(-1))

            ddp_loss[0] += loss
            ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Eval loss/epoch", eval_loss, epoch + 1)

    rank0print(rank, f"Epoch: {epoch + 1}/{max_epochs} \tValidation loss: {eval_loss:.4f}")


def save(
    model_args: ModelArguments,
    data_args: DataArguments,
    train_args: TrainingArguments,
    vocab: SimpleVocab,
    states: dict[str, Any],
    start_time: datetime,
):
    folder = Path(model_args.output_dir) / start_time.strftime("%d-%m-%y-%Y_%H:%M:%S")
    folder.mkdir(exist_ok=True, parents=True)

    with open(folder / "model_args.json", mode="w", encoding="utf-8") as f:
        f.write(model_args.to_json())
    with open(folder / "data_args.json", mode="w", encoding="utf-8") as f:
        f.write(data_args.to_json())
    with open(folder / "train_args.json", mode="w", encoding="utf-8") as f:
        f.write(train_args.to_json())
    vocab.save(folder / "vocab.json")
    torch.save(states, f"{folder}/tree_transformer.pt")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    (
        model_args,
        data_args,
        train_args,
    ) = parser.parse_args_into_dataclasses()

    world_size = torch.cuda.device_count()
    dataset = CustomDataset(data_args)

    mp.spawn(fsdp_main, args=(world_size, model_args, data_args, train_args, dataset), nprocs=world_size, join=True)  # type: ignore
    print("DONE")
