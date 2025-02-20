from typing import Dict, Tuple
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

import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from huggingface_hub import login

import polars as pl
import sklearn.model_selection
from tqdm.auto import tqdm


import utils.loading as loading
from toothless.language_model.args import ModelArguments, DataArguments, TrainingArguments
from toothless.utils.dist_helper import cleanup_process_group, setup_process_group, rank0_print
from utils.consts import VAR_NAMES, IGNORE_UNKNOWN


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    examples_df: pl.DataFrame,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant. Provide an intermediate term in a sequence of rewrites from the Start Expression to the Goal Expression.",
) -> Dict:
    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    system_message = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    for row in examples_df.iter_rows(named=True):
        input_id, target = tokenize_single(tokenizer, max_len, system_message, im_start, im_end, nl_tokens, row)
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.int64)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # type: ignore
    )


def tokenize_single(tokenizer, max_len: int, system_message, im_start, im_end, nl_tokens, row):
    input_id, target = [], []
    input_id += system_message
    target += [im_start] + [IGNORE_TOKEN_ID] * (len(system_message) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)

    # First user asks
    role = "<|im_start|>user"
    # goal_term = example["goal"]  # TODO ACTUAL
    user_input_id = (
        tokenizer(role).input_ids
        + nl_tokens
        + tokenizer("Start Term: " + row["start_expr"]).input_ids
        + nl_tokens
        + tokenizer("Goal Term: " + row["goal_expr"]).input_ids
        + [im_end]
        + nl_tokens
    )
    input_id += user_input_id
    target += [im_start] + [IGNORE_TOKEN_ID] * (len(user_input_id) - 3) + [im_end] + nl_tokens

    # Then asssistant responds
    role = "<|im_start|>assistant"
    assistant_input_id = (
        tokenizer(role).input_ids + nl_tokens + tokenizer(row["middle_expr"]).input_ids + [im_end] + nl_tokens
    )
    input_id += assistant_input_id

    target += (
        [im_start]
        + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids)
        + assistant_input_id[len(tokenizer(role).input_ids) + 1 : -2]
        + [im_end]
        + nl_tokens
    )

    assert len(input_id) == len(target)
    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    return input_id, target


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: pl.DataFrame, tokenizer: transformers.PreTrainedTokenizer, max_len: int, rank: int):
        super(SupervisedDataset, self).__init__()

        rank0_print(rank, "Formatting inputs...")
        data_dict = preprocess(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: pl.DataFrame, tokenizer: transformers.PreTrainedTokenizer, max_len: int, rank: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print(rank, "Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(pl.from_dict(self.raw_data.row(i, named=True)), self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def load_df(data_path) -> Tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(data_path).select(["goal_expr", "middle_expr", "start_expr"])

    # print(df.head())

    test_size = 0.2
    random_state = 42
    train, eval = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=random_state)

    return train, eval


def make_supervised_data_module(
    tokenizer,
    data_args,
    max_len,
    rank,
) -> Tuple[Dataset, Dataset]:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset

    rank0_print(rank, "Loading raw data...")
    train_df, eval_df = load_df(data_args.data_path)
    rank0_print(rank, "Raw data loaded")

    rank0_print(rank, "Tokenizing and preprocessing data...")
    train_dataset = dataset_cls(train_df, tokenizer=tokenizer, max_len=max_len, rank=rank)
    eval_dataset = dataset_cls(eval_df, tokenizer=tokenizer, max_len=max_len, rank=rank)
    rank0_print(rank, "Data tokenized and preprocessed")

    return train_dataset, eval_dataset


def data_loaders(
    rank: int,
    world_size: int,
    data_args: DataArguments,
    train_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
):
    train_dataset, eval_dataset = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=train_args.model_max_length, rank=rank
    )
    # Load dataset
    # Create data loaders

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": train_args.per_device_train_batch_size, "sampler": train_sampler}
    eval_kwargs = {"batch_size": train_args.per_device_eval_batch_size, "sampler": eval_sampler}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    eval_kwargs.update(cuda_kwargs)

    train_dataloader = DataLoader(train_dataset, **train_kwargs)
    eval_dataloader = DataLoader(eval_dataset, **eval_kwargs)
    return train_dataloader, eval_dataloader


def train(rank, model, optimizer, train_args, train_dataloader, epoch, sampler=None, writer=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{train_args.num_train_epochs}")):
        # Move batch to device
        batch = {k: v.to(rank) for k, v in batch.items()}
        # Reset Optimizer
        optimizer.zero_grad()

        # Forward pass
        output = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"])
        loss = output.loss

        # Backward pass
        loss.backward()

        # Update weights
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


def eval(rank, model, eval_dataloader, epoch, writer=None):
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

    # Load Tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=train_args.cache_dir,
        model_max_length=train_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        pad_token="<|endoftext|>",
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id

    # Load data
    train_dataloader, eval_dataloader = data_loaders(rank, world_size, data_args, train_args, tokenizer)
    rank0_print(rank, "DataLoaders ready")

    # Load Model
    rank0_print(rank, "Loading model from disk...")
    model_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, cache_dir=train_args.cache_dir, trust_remote_code=True
    )
    model_config.use_cache = False

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=train_args.cache_dir,
        trust_remote_code=True,
    )
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
        train(rank, model, optimizer, train_args, train_dataloader, epoch, writer=writer)

        # Optionally, evaluate the model on the validation set after each epoch
        if train_args.eval_each_epoch:
            eval(rank, model, eval_dataloader, epoch, writer=writer)

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
            torch.save(states, f"{model_args.output_dir}/mygpt2.pt")

    cleanup_process_group()


if __name__ == "__main__":
    if "INVALIDATE_CACHE" in os.environ:
        loading.update_cache(VAR_NAMES, IGNORE_UNKNOWN)

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    login(token=hf_token)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    (
        model_args,
        data_args,
        train_args,
    ) = parser.parse_args_into_dataclasses()
    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(world_size, model_args, data_args, train_args), nprocs=world_size, join=True)  # type: ignore
