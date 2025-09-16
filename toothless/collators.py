from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


from .data import TripleDataSet, Triple
from .args import DataArgs


class TripleCollator:
    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        raise NotImplementedError


class TripleDualCollator(TripleCollator):
    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        # Use defaultdict to automatically create lists for each key

        stack_dicts = {key: torch.stack([d.tensor_dict[key] for d in triples]) for key in triples[0].tensor_dict.keys()}

        return stack_dicts


def mk_loaders(
    rank: int,
    world_size: int,
    dataset: TripleDataSet,
    collator: TripleCollator,
    data_args: DataArgs,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[DataLoader[Triple], DataLoader[Triple]]:
    # Create and load dataset
    rng = torch.Generator().manual_seed(data_args.rng_seed)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], rng
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    return train_dataloader, eval_dataloader
