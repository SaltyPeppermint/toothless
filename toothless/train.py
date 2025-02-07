import json
from typing import Dict, Tuple
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import transformers
from transformers.trainer_pt_utils import LabelSmoother

import polars as pl
import polars.selectors as cs
import sklearn.model_selection
from tqdm.auto import tqdm


import loading
from args import ModelArguments, DataArguments, TrainingArguments

local_rank = None


NAMES_BLINDED = True
IGNORE_UNKNOWN = True
VAR_NAMES = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "x0",
    "x1",
    "x2",
    "x3",
]

COLS_TO_DROP = ["generation", "expression", "explanation_chain"]

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
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

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


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data: pl.DataFrame, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
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

    def __init__(self, raw_data: pl.DataFrame, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data.select(cs.by_index(i)), self.tokenizer, self.max_len)
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
    print("Data loaded")

    test_size = 0.2
    random_state = 42
    train, eval = sklearn.model_selection.train_test_split(df, test_size=test_size, random_state=random_state)

    return train, eval


def make_supervised_data_module(
    tokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    rank0_print("Loading data...")

    train_df, eval_df = load_df(data_args.data_path)

    train_dataset = dataset_cls(train_df, tokenizer=tokenizer, max_len=max_len)
    eval_dataset = dataset_cls(eval_df, tokenizer=tokenizer, max_len=max_len)
    print("Data tokenized and preprocessed")

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# # Load dataset from JSON file
# class JSONDataset(Dataset):
#     def __init__(self, file_path, tokenizer):
#         with open(file_path, "r") as f:
#             self.data = json.load(f)
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data[idx]["text"]
#         label = self.data[idx]["label"]
#         encoding = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
#         return {
#             "input_ids": encoding["input_ids"].squeeze(),
#             "attention_mask": encoding["attention_mask"].squeeze(),
#             "label": torch.tensor(label, dtype=torch.long),
#         }


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # Load Tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        pad_token="<|endoftext|>",
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    # Load dataset
    # Create data loaders
    train_dataloader = DataLoader(data_module["train_dataset"], batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(data_module["eval_dataset"], batch_size=8)

    # Load Model

    model_config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    model_config.use_cache = False

    sharded_module = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    # FSDP model
    sharded_module = FSDP(sharded_module)

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        sharded_module.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.weight_decay,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=training_args.tmax)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sharded_module.train()
    for epoch in range(training_args.epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{training_args.epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = sharded_module(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Optionally, evaluate the model on the validation set after each epoch
        sharded_module.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = sharded_module(**batch)
            eval_loss += outputs.loss.item()

        eval_loss /= len(eval_dataloader)
        print(f"Epoch {epoch + 1}: Validation Loss = {eval_loss}")

        sharded_module.train()


if __name__ == "__main__":
    if "INVALIDATE_CACHE" in os.environ:
        loading.update_cache(VAR_NAMES, IGNORE_UNKNOWN)

    train()
