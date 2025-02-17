from pathlib import Path
import os

from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import polars as pl
import sklearn.model_selection
from tqdm.auto import tqdm


from toothless.utils import loading
from toothless.utils.args import ModelArguments, DataArguments, TrainingArguments
from toothless.utils.train import cleanup_process_group, setup_process_group, rank0_print
from toothless.utils.consts import VAR_NAMES, IGNORE_UNKNOWN
from toothless.gnn import data, model


# def data_loaders(
#     rank: int,
#     world_size: int,
#     data_args: DataArguments,
#     train_args: TrainingArguments,
# ):
#     # Load dataset
#     # Create data loaders

#     train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
#     eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

#     train_kwargs = {"batch_size": train_args.per_device_train_batch_size, "sampler": train_sampler}
#     eval_kwargs = {"batch_size": train_args.per_device_eval_batch_size, "sampler": eval_sampler}
#     cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
#     train_kwargs.update(cuda_kwargs)
#     eval_kwargs.update(cuda_kwargs)

#     train_dataloader = DataLoader(train_dataset, **train_kwargs)
#     eval_dataloader = DataLoader(eval_dataset, **eval_kwargs)
#     return train_dataloader, eval_dataloader


# def train(rank, model, optimizer, train_args, train_dataloader, epoch, sampler=None, writer=None):
#     model.train()
#     ddp_loss = torch.zeros(2).to(rank)
#     if sampler:
#         sampler.set_epoch(epoch)
#     for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_args.num_train_epochs}")):
#         # Move batch to device
#         batch = {k: v.to(rank) for k, v in batch.items()}
#         # Reset Optimizer
#         optimizer.zero_grad()

#         # Forward pass
#         output = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
#         loss = output.loss

#         # Backward pass
#         loss.backward()

#         # Update weights
#         optimizer.step()

#         if writer:
#             writer.add_scalar("Loss/train-batch", loss, batch_idx + epoch * len(train_dataloader))

#         # Record loss
#         ddp_loss[0] += loss.item()
#         ddp_loss[1] += len(batch)

#     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
#     train_loss = ddp_loss[0] / ddp_loss[1]
#     if writer:
#         writer.add_scalar("Loss/train-epoch", train_loss, epoch + 1)
#     rank0_print(rank, "Epoch: {} \tLoss: {:.6f}".format(epoch + 1, train_loss))


# def eval(rank, model, eval_dataloader, epoch, writer=None):
#     model.eval()
#     ddp_loss = torch.zeros(3).to(rank)
#     for batch in eval_dataloader:
#         batch = {k: v.to(rank) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)
#             ddp_loss[0] += outputs.loss
#             ddp_loss[1] += len(batch)

#     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
#     eval_loss = ddp_loss[0] / ddp_loss[1]
#     if writer:
#         writer.add_scalar("Loss/eval-epoch", eval_loss, epoch + 1)
#     rank0_print(rank, f"Epoch {epoch + 1}: Validation loss: {eval_loss:.4f}")


# def fsdp_main(
#     rank: int, world_size: int, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments
# ):
#     setup_process_group(rank, world_size)
#     rank0_print(rank, "Distributed Network ready")

#     torch.cuda.set_device(rank)

#     # Load Tokenizer

#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=train_args.cache_dir,
#         model_max_length=train_args.model_max_length,
#         padding_side="right",
#         use_fast=False,
#         trust_remote_code=True,
#         pad_token="<|endoftext|>",
#     )
#     tokenizer.pad_token_id = tokenizer.pad_token_id

#     # Load data
#     train_dataloader, eval_dataloader = data_loaders(rank, world_size, data_args, train_args, tokenizer)
#     rank0_print(rank, "DataLoaders ready")

#     # Load Model
#     rank0_print(rank, "Loading model from disk...")
#     model_config = transformers.AutoConfig.from_pretrained(
#         model_args.model_name_or_path, cache_dir=train_args.cache_dir, trust_remote_code=True
#     )
#     model_config.use_cache = False

#     init_start_event = torch.cuda.Event(enable_timing=True)
#     init_end_event = torch.cuda.Event(enable_timing=True)

#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         config=model_config,
#         cache_dir=train_args.cache_dir,
#         trust_remote_code=True,
#     )
#     model.to(rank)
#     rank0_print(rank, "Model loaded")

#     # FSDP model
#     model = FSDP(model)
#     rank0_print(rank, "FSDP Model ready")

#     # Define optimizer and loss function
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=train_args.learning_rate,
#         betas=(train_args.adam_beta1, train_args.adam_beta2),
#         weight_decay=train_args.weight_decay,
#     )
#     lr_scheduler = CosineAnnealingLR(optimizer, T_max=train_args.tmax)
#     rank0_print(rank, "Optimizer and LR Scheduler ready")

#     if rank == 0:
#         writer = SummaryWriter()
#     else:
#         writer = None

#     rank0_print(rank, "Starting training!")

#     init_start_event.record(torch.cuda.current_stream())

#     for epoch in range(train_args.num_train_epochs):
#         train(rank, model, optimizer, train_args, train_dataloader, epoch, writer=writer)

#         # Optionally, evaluate the model on the validation set after each epoch
#         if train_args.eval_each_epoch:
#             eval(rank, model, eval_dataloader, epoch, writer=writer)

#         lr_scheduler.step()

#     rank0_print(rank, "Training finished!")
#     # eval(rank, model, eval_dataloader, train_args.num_train_epochs, writer=writer)

#     init_end_event.record(torch.cuda.current_stream())

#     if rank == 0:
#         init_end_event.synchronize()
#         print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
#         print(f"{model}")

#     if train_args.save_model_end:
#         # use a barrier to make sure training is done on all ranks
#         dist.barrier()
#         states = model.state_dict()
#         if rank == 0:
#             torch.save(states, f"{model_args.output_dir}/mygpt2.pt")

#     cleanup_process_group()


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    data_path = Path("data/start_goal_with_expl/start_and_goal-2025-01-29-b33b4ba4-ee88-48b5-981b-c2b809d6504f/0")

    dataset = data.PairDataset(data_path, 5, VAR_NAMES, IGNORE_UNKNOWN, 42)
    # print(len(dataset))
    train_loader, test_loader = dataset.to_dataloader(8)
    print(train_loader)
    print(test_loader)
    # parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    # (
    #     model_args,
    #     data_args,
    #     train_args,
    # ) = parser.parse_args_into_dataclasses()
    # world_size = torch.cuda.device_count()
    # mp.spawn(fsdp_main, args=(world_size, model_args, data_args, train_args), nprocs=world_size, join=True)  # type: ignore
