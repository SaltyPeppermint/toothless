import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter


from tqdm.auto import tqdm
import transformers

from toothless.utils.dist_helper import cleanup_process_group, setup_process_group, rank0_print
from toothless.tree_model.data import CustomDataset
from toothless.tree_model.model import FastASTTrans
from toothless.tree_model.args import DataArguments, TrainingArguments, ModelArguments


def mk_loaders(rank: int, world_size: int, dataset: CustomDataset, data_args: DataArguments):
    # Create and load dataset
    # split_idx = int(data_args.split_size * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size]
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_args.batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


def train(
    rank: int,
    model: FSDP,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    epoch: int,
    writer: SummaryWriter | None = None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_args.num_train_epochs}")):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}
        # Reset Optimizer
        optimizer.zero_grad()

        # Forward pass
        z_s = model.encode(batch["x_s"], batch["edge_index_s"], batch["x_s_batch"])
        z_t = model.encode(batch["x_t"], batch["edge_index_t"], batch["x_t_batch"])
        pred = model.distance(z_s, z_t)
        loss = (
            model.recon_loss(z_s, batch["x_s"], batch["edge_index_s"])
            + model.recon_loss(z_t, batch["x_t"], batch["edge_index_t"])
            + model.distance_loss(pred, batch["distance"])
        )
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar("Loss/train-batch", loss, batch_idx + epoch * len(train_dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Loss/train-epoch", train_loss, epoch + 1)
    rank0_print(rank, "Epoch: {} \tLoss: {:.6f}".format(epoch + 1, train_loss))


def eval(
    rank: int,
    model: nn.Module,
    eval_dataloader: DataLoader,
    epoch: int,
    writer: SummaryWriter | None = None,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    for batch in eval_dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            ddp_loss[0] += outputs.loss
            ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Loss/eval-epoch", eval_loss, epoch + 1)
    rank0_print(rank, f"Epoch {epoch + 1}: Validation loss: {eval_loss:.4f}")


def fsdp_main(
    rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments
):
    setup_process_group(rank, world_size)
    rank0_print(rank, "Distributed Network ready")

    torch.cuda.set_device(rank)

    # Load Data
    dataset = CustomDataset(data_args.data_path, 5, data_args.random_state)
    train_dataloader, eval_dataloader = mk_loaders(rank, world_size, dataset, data_args)
    rank0_print(rank, "DataLoaders ready")

    for s in train_dataloader:
        rank0_print(rank, s)
        break

    # Load Model
    rank0_print(rank, "Creating model...")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = FastASTTrans(0, 0, 0, 4, 4, 0, "0", 0, 0, 0.0)  # FIXME
    model.to(rank)
    rank0_print(rank, "Model loaded")

    # FSDP model
    model = FSDP(model)
    rank0_print(rank, "FSDP Model ready")

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.tmax)
    rank0_print(rank, "Optimizer and LR Scheduler ready")

    if rank == 0:
        writer = SummaryWriter()
    else:
        writer = None

    rank0_print(rank, "Starting training!")

    init_start_event.record(torch.cuda.current_stream())

    for epoch in range(train_args.num_train_epochs):
        train(rank, model, optimizer, train_dataloader, epoch, writer)

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(rank, model, eval_dataloader, epoch, writer)

        lr_scheduler.step()

    rank0_print(rank, "Training finished!")
    # eval(rank, model, eval_dataloader, train_args.num_train_epochs, writer=writer)

    init_end_event.record(torch.cuda.current_stream())

    if rank == 0:
        init_end_event.synchronize()
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if train_args.save_model_end:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, f"{model_args.output_dir}/graph_autoencoder.pt")

    cleanup_process_group()


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
