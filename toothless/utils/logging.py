def rank0_print(rank, *args):
    if rank == 0:
        print(*args)
