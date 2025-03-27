import copy
import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm.auto import tqdm
import transformers

from toothless.utils.dist_helper import cleanup_process_group, rank0print, setup_process_group
from toothless.tree_model.data import CustomDataset, DictCollator
from toothless.tree_model.model import ASTTransformer
from toothless.tree_model.args import DataArguments, TrainingArguments, ModelArguments


def mk_loaders(
    rank: int, world_size: int, dataset: CustomDataset, data_args: DataArguments
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
    # Create and load dataset
    # split_idx = int(data_args.split_size * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size]
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    pad_id = dataset.vocab.pad_token_id
    assert pad_id == 0

    collator = DictCollator(pad_id, data_args.max_len)

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_args.batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    return train_dataloader, test_dataloader


def train(
    rank: int,
    model: FSDP,
    optimizer: optim.Optimizer,
    criterion: CrossEntropyLoss,
    train_dataloader: DataLoader,
    epoch: int,
    train_args: TrainingArguments,
    writer: SummaryWriter | None = None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    for batch_idx, (batch, num_tokens) in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_args.num_train_epochs}")
    ):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}

        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch["tgt_ids_y"].view(-1))

        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar("Loss/train-batch", loss, batch_idx + epoch * len(train_dataloader))
            writer.add_scalar("Loss/tokens-in-batch", num_tokens, batch_idx + epoch * len(train_dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Loss/train-epoch", train_loss, epoch + 1)

    rank0print(rank,   f"Epoch: {epoch + 1} \tLoss: {train_loss:.6f}")


def eval(
    rank: int,
    model: nn.Module,
    eval_dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    epoch: int,
    writer: SummaryWriter | None = None,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    for batch in eval_dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch["tgt_ids_y"].view(-1))
            ddp_loss[0] += loss
            ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Loss/eval-epoch", eval_loss, epoch + 1)

    rank0print(rank,   f"Epoch {epoch + 1}: Validation loss: {eval_loss:.4f}")


def fsdp_main(
    rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments
):
    setup_process_group(rank, world_size)

    rank0print(rank,"Distributed Network ready")

    writer = SummaryWriter() if rank == 0 else None

    torch.cuda.set_device(rank)

    # Load Data
    dataset = CustomDataset(data_args.data_path, 5, data_args.max_rel_distance, random_state=data_args.random_state)
    train_dataloader, eval_dataloader = mk_loaders(rank, world_size, dataset, data_args)

    rank0print(rank, "DataLoaders ready")

    vocab_size = len(dataset.vocab)
    # Load Model
    rank0print(rank, "Creating model...")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ASTTransformer(
        vocab_size,
        vocab_size,
        model_args.d_model,
        model_args.num_layers,  # 1 for testing
        model_args.dim_feed_forward,
        model_args.dropout,
        "p2q_p2k",
        model_args.anc_heads,
        model_args.sib_heads,
        dataset.max_rel_distance,
    )

    if writer:
        example_batch, _ = next(iter(copy.deepcopy(eval_dataloader)))
        writer.add_graph(model, example_batch)
        model.to(rank)

    model.to(rank)
    rank0print(rank,"Model loaded")

    # FSDP model
    model = FSDP(model)
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
    rank0print(rank,"Optimizer and LR Scheduler ready")

    rank0print(rank, "Starting training!")
    init_start_event.record(torch.cuda.current_stream())

    for epoch in range(train_args.num_train_epochs):
        train(rank, model, optimizer, criterion, train_dataloader, epoch, train_args, writer)

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(rank, model, eval_dataloader, criterion, epoch, writer)

        lr_scheduler.step()

    rank0print(rank, "Training finished!")
    # eval(rank, model, eval_dataloader, train_args.num_train_epochs, writer=writer)

    init_end_event.record(torch.cuda.current_stream())

    if rank == 0:
        init_end_event.synchronize()
    rank0print(rank, f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000} sec")
    rank0print(rank,  f"{model}")

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

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    logger = logging.getLogger(__name__)

    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, model_args, data_args, train_args), nprocs=world_size, join=True)  # type: ignore
    logger.info("DONE")
