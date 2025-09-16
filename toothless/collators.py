from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from .data import TripleDataSet, Triple
from .args import DataArgs


class TripleCollator:
    def __init__(self, max_len: int, tokenizer: Tokenizer):
        assert tokenizer.token_to_id("[PAD]") == 0
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __call__(self, triples: Sequence[Triple]) -> tuple[dict[str, Tensor], int]:
        assert type(triples[0]) is Triple

        start = self.tokenizer.encode_batch([t.start for t in triples])
        guide = self.tokenizer.encode_batch([t.guide for t in triples])
        target = self.tokenizer.encode_batch([t.target for t in triples])

        for s, g, t in zip(start, guide, target):
            if max(len(s.ids), len(g.ids), len(t.ids)) > self.max_len:
                print(s)
                print(g)
                print(t)
                raise ValueError("too long")

        batch = {
            "start": torch.tensor([i.ids for i in start], dtype=torch.long),
            "start_mask": torch.tensor([i.attention_mask for i in guide], dtype=torch.bool),
            "guide": torch.tensor([i.ids for i in guide], dtype=torch.long),
            "guide_mask": torch.tensor([i.attention_mask for i in guide], dtype=torch.bool),
            "target": torch.tensor([i.ids for i in target], dtype=torch.long),
            "target_mask": torch.tensor([i.attention_mask for i in guide], dtype=torch.bool),
        }

        return batch, sum([len(seq.ids) for seq in target])


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
