import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.profiler import profile, ProfilerActivity
from torch.distributed.checkpoint.state_dict import get_state_dict

from tqdm.auto import tqdm
import tyro

from toothless.collators import DictCollator, mk_loaders
from toothless.utils import cleanup_process_group, rank0print, setup_process_group
from toothless.data import TripleDataSet, Triple
from toothless.model import DualTreeTransformer
from toothless.utils import count_parameters
from toothless.args import DataArguments, TrainingArguments, ModelArguments, TrainRunArgs


def fsdp_main(
    rank: int,
    world_size: int,
    model_args: ModelArguments,
    train_args: TrainingArguments,
    data_args: DataArguments,
    save_folder: Path,
):
    setup_process_group(rank, world_size)
    torch.cuda.set_device(rank)

    writer = SummaryWriter(log_dir=train_args.run_log_dir) if rank == 0 else None

    dataset = TripleDataSet(data_args)
    if rank == 0:
        dataset.vocab.save(save_folder / "vocab.json")
    rank0print("Dataset ready")

    # Load Data
    vocab_size = len(dataset.vocab)
    collator = DictCollator(dataset.vocab.pad_token_id, data_args.max_len, dataset.vocab)
    train_dataloader, eval_dataloader = mk_loaders(
        rank, world_size, dataset, collator, data_args, train_args.batch_size
    )
    rank0print("DataLoaders ready")

    # Construct Base Model
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = DualTreeTransformer(model_args, vocab_size, vocab_size, dataset.vocab.pad_token_id)

    if writer and train_args.trace:
        example_batch, _ = next(iter(copy.deepcopy(train_dataloader)))
        writer.add_graph(model, example_batch)

    table, total_params = count_parameters(model)
    rank0print(table)
    rank0print(f"Total Parameters: {total_params}")

    # FSDP model and Mixed Precision Config
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16) if model_args.bf16 else None
    sharding_strategy = ShardingStrategy.FULL_SHARD if world_size > 1 else ShardingStrategy.NO_SHARD

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        device_id=rank,
        use_orig_params=True,  # ALLOWS FULL GRAPH CAPTURE BUT WE DONT HAVE CHONKY GPU
    )

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=torch.tensor(train_args.learning_rate),
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

    best_eval_loss = float("inf")
    if rank == 0:
        (save_folder / "weights").mkdir(exist_ok=True, parents=True)

    rank0print("Starting training!")
    init_start_event.record(torch.cuda.current_stream())

    # profil_model(rank, model, copy.deepcopy(train_dataloader), criterion)

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

        # use a barrier to make sure training is done on all ranks
        model_state_dict, _optimizer_state_dict = get_state_dict(model, optimizer)
        _checkpoint_future = dcp.async_save(model_state_dict, checkpoint_id=save_folder / "weights" / f"{epoch}.pt")

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval_loss = evalulate(
                rank, model, copy.deepcopy(eval_dataloader), criterion, epoch, train_args.epochs, writer
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model_state_dict, _optimizer_state_dict = get_state_dict(model, optimizer)
                _checkpoint_future = dcp.async_save(
                    model_state_dict, checkpoint_id=save_folder / "weights" / "best_eval.pt"
                )

        cosine_scheduler.step()

    init_end_event.record(torch.cuda.current_stream())
    rank0print("Training finished!")

    if rank == 0:
        init_end_event.synchronize()
    rank0print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000} sec")

    model_state_dict, _optimizer_state_dict = get_state_dict(model, optimizer)
    _checkpoint_future = dcp.async_save(model_state_dict, checkpoint_id=save_folder / "weights" / "final.pt")

    cleanup_process_group()


def profil_model(rank: int, model: FSDP, dataloader: DataLoader[Triple], criterion: CrossEntropyLoss):
    model.train()
    dl_iter = iter(dataloader)

    for batch_idx in tqdm(range(2), desc="Profiling... "):
        batch, _, _ = next(dl_iter)

        # Move batch to device
        tgt_ids, l_ids, r_ids = batch["tgt_ids"].to(rank), batch["l_ids"].to(rank), batch["r_ids"].to(rank)

        # Create input and target for teacher forcing
        tgt_input = tgt_ids[:, :-1]  # All tokens except last
        tgt_output = tgt_ids[:, 1:]  # All tokens except first (shifted by 1)

        # Forward pass

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            _ = model(tgt_input, l_ids, r_ids)
            logits = model(tgt_input, l_ids, r_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

            # Backwards pass
            loss.backward()

        prof.export_chrome_trace(f"trace{batch_idx}.json")


def train(
    rank: int,
    model: FSDP,
    dataloader: DataLoader[Triple],
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

    for batch_idx, (batch, _, num_tokens) in enumerate(
        tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{train_args.epochs}")
    ):
        # Move batch to device
        tgt_ids, l_ids, r_ids = batch["tgt_ids"].to(rank), batch["l_ids"].to(rank), batch["r_ids"].to(rank)

        # Create input and target for teacher forcing
        tgt_input = tgt_ids[:, :-1]  # All tokens except last
        tgt_output = tgt_ids[:, 1:]  # All tokens except first (shifted by 1)

        # Forward pass
        logits = model(tgt_input, l_ids, r_ids)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

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

        if writer is not None:
            writer.add_scalar("Train Loss/batch", loss, batch_idx + epoch * len(dataloader))
            writer.add_scalar("Train LR/batch", last_lr[-1], batch_idx + epoch * len(dataloader))
            writer.add_scalar("Toks/in-batch", num_tokens, batch_idx + epoch * len(dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer is not None:
        writer.add_scalar("Train Loss/epoch", train_loss, epoch + 1)

    rank0print(f"Epoch: {epoch + 1}/{train_args.epochs} \tTrain Loss: {train_loss:.6f}")


def evalulate(
    rank: int,
    model: nn.Module,
    dataloader: DataLoader[Triple],
    criterion: CrossEntropyLoss,
    epoch: int,
    max_epochs: int,
    writer: SummaryWriter | None = None,
) -> float:
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)
    for batch, _, _ in dataloader:
        # Move batch to device
        tgt_ids, l_ids, r_ids = batch["tgt_ids"].to(rank), batch["l_ids"].to(rank), batch["r_ids"].to(rank)

        # Create input and target for teacher forcing
        tgt_input = tgt_ids[:, :-1]  # All tokens except last
        tgt_output = tgt_ids[:, 1:]  # All tokens except first (shifted by 1)

        # Forward pass
        with torch.no_grad():
            logits = model(tgt_input, l_ids, r_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            ddp_loss[0] += loss
            ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss[0] / ddp_loss[1]
    if writer is not None:
        writer.add_scalar("Eval loss/epoch", eval_loss, epoch + 1)

    rank0print(f"Epoch: {epoch + 1}/{max_epochs} \tValidation loss: {eval_loss:.4f}")
    return float(eval_loss)


if __name__ == "__main__":
    args = tyro.cli(TrainRunArgs)

    start_time = datetime.now()
    save_folder = Path(args.model.output_dir) / start_time.strftime("%y-%m-%d-%H:%M:%S")
    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / "model_args.json", mode="w", encoding="utf-8") as f:
        f.write(args.model.to_json())
    with open(save_folder / "data_args.json", mode="w", encoding="utf-8") as f:
        f.write(args.data.to_json())
    with open(save_folder / "train_args.json", mode="w", encoding="utf-8") as f:
        f.write(args.train.to_json())

    world_size = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(world_size, args.model, args.train, args.data, save_folder), nprocs=world_size, join=True)
    print("DONE")
