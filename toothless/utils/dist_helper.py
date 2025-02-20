import torch.distributed as dist


def rank0_print(rank, *args):
    if rank == 0:
        print(*args)


def setup_process_group(rank, world_size):
    rank0_print(rank, "Setting up process group...")
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Process group created")


def cleanup_process_group():
    dist.destroy_process_group()
