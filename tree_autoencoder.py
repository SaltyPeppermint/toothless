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
    token_criterion: CrossEntropyLoss,
    distance_criterion: CrossEntropyLoss,
    train_dataloader: DataLoader,
    epoch: int,
    train_args: TrainingArguments,
    with_anc: bool = False,
    with_sib: bool = False,
    writer: SummaryWriter | None = None,
):
    model.train()
    ddp_loss = torch.zeros(5).to(rank)

    for batch_idx, (batch, num_tokens) in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_args.num_train_epochs}")
    ):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}

        # Forward pass
        tok_out, anc_out, sib_out = model(batch)

        tok_loss = token_criterion(tok_out.view(-1, tok_out.size(-1)), batch["tgt_ids_y"].view(-1))
        ddp_loss[2] += tok_loss.item()
        loss = tok_loss
        scale_factor = 1
        if writer:
            writer.add_scalar("Train Loss Batch/tok", loss, batch_idx + epoch * len(train_dataloader))

        if with_anc:
            anc_loss = distance_criterion(anc_out.view(-1, anc_out.size(-1)), batch["tgt_anc_y"].view(-1))
            anc_loss = anc_loss / anc_out.size(-2)  # Scale for dim
            ddp_loss[3] += anc_loss.item()
            loss = loss + anc_loss
            scale_factor = scale_factor + 1
            if writer:
                writer.add_scalar("Train Loss Batch/anc", loss, batch_idx + epoch * len(train_dataloader))

        if with_sib:
            sib_loss = distance_criterion(sib_out.view(-1, anc_out.size(-1)), batch["tgt_sib_y"].view(-1))
            sib_loss = sib_loss / sib_out.size(-2)  # Scale for dim
            ddp_loss[4] += sib_loss.item()
            loss = loss + sib_loss
            scale_factor = scale_factor + 1
            if writer:
                writer.add_scalar("Train Loss Batch/sib", loss, batch_idx + epoch * len(train_dataloader))

        loss = loss / scale_factor
        # Backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar("Train Loss Batch/total", loss, batch_idx + epoch * len(train_dataloader))
            writer.add_scalar("Toks/in-batch", num_tokens, batch_idx + epoch * len(train_dataloader))

        # Record loss
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Train Loss Epoch/total", train_loss, epoch + 1)
        writer.add_scalar("Train Loss Epoch/tok", ddp_loss[2] / ddp_loss[1], epoch + 1)
        if with_anc:
            writer.add_scalar("Train Loss Epoch/anc", ddp_loss[3] / ddp_loss[1], epoch + 1)
        if with_anc:
            writer.add_scalar("Train Loss Epoch/sib", ddp_loss[4] / ddp_loss[1], epoch + 1)

    rank0print(rank, f"Epoch: {epoch + 1} \tLoss: {train_loss:.6f}")


def eval(
    rank: int,
    model: nn.Module,
    eval_dataloader: DataLoader,
    token_criterion: CrossEntropyLoss,
    distance_criterion: CrossEntropyLoss,
    epoch: int,
    with_anc: bool = False,
    with_pos: bool = False,
    writer: SummaryWriter | None = None,
):
    model.eval()
    ddp_loss = torch.zeros(5).to(rank)
    for batch in eval_dataloader:
        batch = {k: v.to(rank) for k, v in batch.items()}
        with torch.no_grad():
            tok_out, anc_out, sib_out = model(batch)

            tok_loss = token_criterion(tok_out.view(-1, tok_out.size(-1)), batch["tgt_ids_y"].view(-1))
            ddp_loss[2] += tok_loss

            loss = tok_loss
            scale_factor = 1

            if with_anc:
                anc_loss = distance_criterion(anc_out.view(-1, anc_out.size(-1)), batch["tgt_anc_y"].view(-1))
                ddp_loss[3] += anc_loss
                loss = loss + anc_loss
                scale_factor = scale_factor + 1

            if with_pos:
                sib_loss = distance_criterion(sib_out.view(-1, anc_out.size(-1)), batch["tgt_sib_y"].view(-1))
                ddp_loss[4] += sib_loss
                loss = loss + sib_loss
                scale_factor = scale_factor + 1

            ddp_loss[0] += loss / scale_factor
            ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss[0] / ddp_loss[1]
    if writer:
        writer.add_scalar("Loss Eval/total", eval_loss, epoch + 1)
        writer.add_scalar("Loss Eval/tok", ddp_loss[2] / ddp_loss[1], epoch + 1)
        if with_anc:
            writer.add_scalar("Loss Eval/anc", ddp_loss[3] / ddp_loss[1], epoch + 1)
        if with_anc:
            writer.add_scalar("Loss Eval/sib", ddp_loss[4] / ddp_loss[1], epoch + 1)

    rank0print(rank, f"Epoch {epoch + 1}: Validation loss: {eval_loss:.4f}")


def fsdp_main(
    rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments
):
    setup_process_group(rank, world_size)

    rank0print(rank, "Distributed Network ready")

    writer = SummaryWriter() if rank == 0 else None

    torch.cuda.set_device(rank)

    # Load Data
    dataset = CustomDataset(data_args)
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
        data_args.max_len,
    )

    if writer:
        example_batch, _ = next(iter(copy.deepcopy(eval_dataloader)))
        writer.add_graph(model, example_batch)
        model.to(rank)

    model.to(rank)
    rank0print(rank, "Model loaded")

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
    token_criterion = CrossEntropyLoss(ignore_index=dataset.vocab.pad_token_id, label_smoothing=0.1)
    distance_criterion = CrossEntropyLoss()
    rank0print(rank, "Optimizer and LR Scheduler ready")

    with_anc = model_args.anc_heads > 0
    with_sib = model_args.sib_heads > 0
    rank0print(rank, "Starting training!")
    init_start_event.record(torch.cuda.current_stream())

    for epoch in range(train_args.num_train_epochs):
        train(
            rank,
            model,
            optimizer,
            token_criterion,
            distance_criterion,
            train_dataloader,
            epoch,
            train_args,
            with_anc,
            with_sib,
            writer,
        )

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(
                rank,
                model,
                copy.deepcopy(eval_dataloader),
                token_criterion,
                distance_criterion,
                epoch,
                with_anc,
                with_sib,
                writer,
            )

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
