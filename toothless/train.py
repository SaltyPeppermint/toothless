import sys
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from deepspeed.runtime.lr_schedules import WarmupLR
import deepspeed

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
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for row in examples_df.iter_rows(named=True):
        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)

        # First user asks
        role = "<|im_start|>user"
        # goal_term = example["goal"]  # TODO ACTUAL
        user_input_id = (
            tokenizer(role).input_ids + nl_tokens + tokenizer(row["expression"]).input_ids + [im_end] + nl_tokens
        )
        input_id += user_input_id
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(user_input_id) - 3) + [im_end] + nl_tokens

        # Then asssistant responds
        role = "<|im_start|>assistant"
        # middle_sketch = example["sketch"]  # TODO EXPAND RESPONSE
        assistant_input_id = (
            tokenizer(role).input_ids + nl_tokens + tokenizer(row["middle_item"]).input_ids + [im_end] + nl_tokens
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
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # type: ignore
    )


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
    df = pl.read_parquet(data_path).select(["expression", "middle_item"])

    print(df.head())
    print("Data loading goal done")

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


def train():
    global local_rank
    # Load model and tokenizer

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    if data_args.update_cache:
        loading.update_cache(VAR_NAMES, IGNORE_UNKNOWN)

    # # This serves for single-gpu qlora.
    # if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
    #     training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    # Load DeepSpeed config
    with open(training_args.ds_config) as f:
        ds_config = json.load(f)

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        # device_map=device_map,
        trust_remote_code=True,
        # quantization_config=GPTQConfig(bits=4, disable_exllama=True)
        # if training_args.use_lora and lora_args.q_lora
        # else None,
        # **model_load_kwargs,
    )

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

    # def tokenize_fn(examples):
    #     return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Create data loaders
    train_dataloader = DataLoader(data_module["train_dataset"], batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(data_module["eval_dataset"], batch_size=8)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = WarmupLR(optimizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DeepSpeed
    ds_config["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )

    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)

    model.train()
    for epoch in range(training_args.epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{training_args.epochs}"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Optionally, evaluate the model on the validation set after each epoch
        model.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            eval_loss += outputs.loss.item()

        eval_loss /= len(eval_dataloader)
        print(f"Epoch {epoch + 1}: Validation Loss = {eval_loss}")

        model.train()


if __name__ == "__main__":
    # model_dir = snapshot_download("Qwen/Qwen-1_8B-Chat", cache_dir="cache", revision="master")
    update_cache = sys.argv[1] == "--update-cache" if len(sys.argv) >= 2 else False

    train()
