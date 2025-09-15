import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn, Tensor

from termcolor import cprint
from prettytable import PrettyTable

from toothless.args import ModelArgs


def get_save_folder(model_args: ModelArgs, start_time_str: str) -> Path:
    return Path(model_args.output_dir) / start_time_str


def count_parameters(model: nn.Module) -> tuple[PrettyTable, int]:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return table, total_params


def create_padding_mask(
    input_ids: Tensor,
    pad_token_id: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Creates a padding mask for attention mechanisms.
    """
    if device is None:
        device = input_ids.device

    # Create mask (1 for padding, 0 for actual tokens)

    return (input_ids == pad_token_id).to(device)


def rank0print(message, color: str | tuple[int, int, int] | None = None):
    if dist.get_rank() == 0:
        cprint(message, color)


def rank0eprint(message):
    if dist.get_rank() == 0:
        print(message, file=sys.stderr)


def setup_process_group(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6601"

    # initialize the process group
    # cpu for async save
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    print("Process group created")


def cleanup_process_group():
    dist.destroy_process_group()
