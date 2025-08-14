from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from eggshell import rise  # type: ignore

from .data import TrippleDataSet, Tripple
from .args import DataArguments
from .vocab import SimpleVocab


class DictCollator:
    def __init__(self, pad_id: int, max_len: int, vocab: SimpleVocab):
        self.pad_id = pad_id
        self.max_len = max_len
        self.vocab = vocab

    def __call__(self, tripples: Sequence[Tripple]) -> tuple[dict[str, Tensor], list[list[str]], int]:
        assert type(tripples[0]) is Tripple

        l_batch = []
        r_batch = []
        tgt_batch = []
        rules_chains = []

        for tripple in tripples:
            l_batch.append(tripple.l_ids)
            r_batch.append(tripple.r_ids)
            tgt_batch.append(tripple.tgt_ids)
            rules_chains.append(tripple.rules_chain)

            assert max(len(tripple.l_ids), len(tripple.r_ids), len(tripple.tgt_ids)) <= self.max_len

        batch = {
            "tgt_ids": torch.nested.nested_tensor(tgt_batch, layout=torch.jagged).to_padded_tensor(self.pad_id),
            "l_ids": torch.nested.nested_tensor(l_batch, layout=torch.jagged).to_padded_tensor(self.pad_id),
            "r_ids": torch.nested.nested_tensor(r_batch, layout=torch.jagged).to_padded_tensor(self.pad_id),
        }

        return batch, rules_chains, sum([len(seq) for seq in tgt_batch])

    def _pyrec_to_tensor(self, expr: rise.RecExpr) -> Tensor:
        tree_data = expr.to_data()

        return torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )


def mk_loaders(
    rank: int,
    world_size: int,
    dataset: TrippleDataSet,
    collator: DictCollator,
    data_args: DataArguments,
    shuffle: bool = True,
) -> tuple[DataLoader[Tripple], DataLoader[Tripple]]:
    # Create and load dataset
    rng = torch.Generator().manual_seed(data_args.rng_seed)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size], rng
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

    pad_id = dataset.vocab.pad_token_id
    assert pad_id == 0

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_args.batch_size,
        sampler=eval_sampler,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    return train_dataloader, eval_dataloader
