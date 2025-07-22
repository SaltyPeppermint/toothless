import os
import sys
import torch.distributed as dist

from termcolor import cprint


def rank0print(rank: int, message, color: str | tuple[int, int, int] | None = None):
    if rank == 0:
        cprint(message, color)


def rank0eprint(rank: int, message):
    if rank == 0:
        print(message, file=sys.stderr)


def setup_process_group(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6601"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Process group created")


def cleanup_process_group():
    dist.destroy_process_group()
