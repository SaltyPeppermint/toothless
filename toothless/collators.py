from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .data import TripleDataSet, Triple
from .args import DataArgs


class TripleCollator:
    def __init__(self, target_length: int, pad_token_id: int = 0):
        self.target_length = target_length
        self.pad_token_id = pad_token_id

    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        raise NotImplementedError

    def pad_stack(self, tensor_list: list[Tensor]) -> tuple[Tensor, Tensor]:
        padded_tensors = []
        padding_masks = []

        for tensor in tensor_list:
            current_length = tensor.shape[0]

            if current_length > self.target_length:
                raise ValueError(f"Tensor length {current_length} exceeds target length {self.target_length}")

            mask = torch.ones(self.target_length, dtype=torch.bool)
            mask[current_length:] = False
            padding_masks.append(mask)

            pad_amount = self.target_length - current_length
            padded_tensor = F.pad(tensor, (0, pad_amount), value=self.pad_token_id)
            padded_tensors.append(padded_tensor)

        # stacked_tensors = torch.stack(padded_tensors, dim=0)
        # stacked_masks = stacked_tensors != self.pad_token_id
        # Stack all padded tensors and masks
        return torch.stack(padded_tensors, dim=0), torch.stack(padding_masks, dim=0)


class TripleDualCollator(TripleCollator):
    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        start_ids, start_mask = self.pad_stack([t.start_ids for t in triples])
        target_ids, target_mask = self.pad_stack([t.target_ids for t in triples])
        guide_ids, guide_mask = self.pad_stack([t.guide_ids for t in triples])

        return {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "guide_ids": guide_ids,
            "guide_mask": guide_mask,
        }


class DecoderOnlyCollator(TripleCollator):
    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        tgt_ids, tgt_mask = self.pad_stack(
            [torch.cat((t.start_ids, t.target_ids[1:], t.guide_ids[1:])) for t in triples]
        )

        return {"tgt_ids": tgt_ids, "tgt_mask": tgt_mask}


class EncoderOnlyCollator(TripleCollator):
    def __call__(self, triples: Sequence[Triple]) -> dict[str, Tensor]:
        tgt_ids, tgt_mask = self.pad_stack(
            [torch.cat((t.start_ids, t.target_ids[1:], t.guide_ids[1:])) for t in triples]
        )
        pos_ids, _ = self.pad_stack(
            [
                torch.cat(
                    (
                        torch.full_like(t.start_ids, 0)[1:],
                        torch.full_like(t.target_ids, 1)[1:],
                        torch.full_like(t.guide_ids, 2)[1:],
                    )
                )
                for t in triples
            ]
        )

        return {"tgt_ids": tgt_ids, "tgt_mask": tgt_mask, "pos_ids": pos_ids}


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
