import copy
import dataclasses
import json
from pathlib import Path
from datetime import datetime
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm.auto import tqdm
import transformers

from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.tree_model.data import CustomDataset, mk_loaders
from toothless.tree_model.model import ASTTransformer
from toothless.tree_model.args import DataArguments, TrainingArguments, ModelArguments


def fsdp_main(
    rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments
):
    start_time = datetime.now()
    setup_process_group(rank, world_size)

    rank0print(rank, "Distributed Network ready")

    writer = SummaryWriter() if rank == 0 else None

    torch.cuda.set_device(rank)

    # Load Data
    dataset = CustomDataset(data_args, len_limit=100)
    train_dataloader, eval_dataloader = mk_loaders(rank, world_size, dataset, data_args)

    rank0print(rank, "DataLoaders ready")

    vocab_size = len(dataset.vocab)
    # Load Model

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ASTTransformer(
        model_args,
        vocab_size,
        vocab_size,
        data_args.k,
    )

    if writer:
        example_batch, _ = next(iter(copy.deepcopy(train_dataloader)))
        writer.add_graph(model, example_batch)
        model.to(rank)

    model.to(rank)
    rank0print(rank, "Model loaded")

    # FSDP model
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    model = FullyShardedDataParallel(model, sharding_strategy=sharding_strategy, mixed_precision=mixed_precision)
    rank0print(rank, "FSDP Model ready")

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.tmax)
    criterion = CrossEntropyLoss(ignore_index=dataset.vocab.pad_token_id, label_smoothing=0.1)
    rank0print(rank, "Optimizer and LR Scheduler ready")

    rank0print(rank, "Starting training!")
    init_start_event.record(torch.cuda.current_stream())

    for epoch in range(train_args.num_train_epochs):
        train(rank, model, copy.deepcopy(train_dataloader), criterion, optimizer, epoch, train_args, writer)

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(rank, model, copy.deepcopy(eval_dataloader), criterion, epoch, train_args.num_train_epochs, writer)

        lr_scheduler.step()

    rank0print(rank, "Training finished!")
    # eval(rank, model, eval_dataloader, train_args.num_train_epochs, writer=writer)

    init_end_event.record(torch.cuda.current_stream())

    if rank == 0:
        init_end_event.synchronize()
    rank0print(rank, f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000} sec")
    rank0print(rank, f"{model}")

    if train_args.save_model_end:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            save(model_args, data_args, train_args, states, start_time)

    cleanup_process_group()


def train(
    rank: int,
    model: FullyShardedDataParallel,
    dataloader: DataLoader[dict[str, Tensor]],
    criterion: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    epoch: int,
    train_args: TrainingArguments,
    writer: SummaryWriter | None = None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    for batch_idx, (batch, num_tokens) in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{train_args.num_train_epochs}")
    ):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}

        # Forward pass
        out = model(batch)
        loss = criterion(out.view(-1, out.size(-1)), batch["tgt_ids_y"].view(-1))

        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar("Train Loss/batch", loss, batch_idx + epoch * len(dataloader))
            writer.add_scalar("Toks/in-batch", num_tokens, batch_idx + epoch * len(dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Train Loss/epoch", train_loss, epoch + 1)

    rank0print(rank, f"Epoch: {epoch + 1}/{train_args.num_train_epochs} \tTrain Loss: {train_loss:.6f}")


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
    states: dict[str, Any],
    start_time: datetime,
):
    folder = Path(model_args.output_dir) / start_time.strftime("%d-%m-%y-%Y_%H:%M:%S")
    folder.mkdir(exist_ok=True, parents=True)

    with open(folder / "model_args.json", mode="w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(model_args), f)
    with open(folder / "data_args.json", mode="w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(data_args), f)
    with open(folder / "train_args.json", mode="w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(train_args), f)
    torch.save(states, f"{folder}/tree_transformer.pt")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    (
        model_args,
        data_args,
        train_args,
    ) = parser.parse_args_into_dataclasses()

    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, model_args, data_args, train_args), nprocs=world_size, join=True)  # type: ignore
    print("DONE")
