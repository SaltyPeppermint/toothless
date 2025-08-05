from typing import Sequence

from eggshell import rise  # type: ignore

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .models.utils import make_tgt_mask
from .data import TrippleDataSet, Tripple
from .args import DataArguments
from .vocab import SimpleVocab


class BaseCollator:
    def __init__(self, pad_id: int, max_len: int, k: int, vocab: SimpleVocab):
        self.pad_id = pad_id
        self.max_len = max_len
        self.k = k
        self.vocab = vocab

    def __call__(self, tripples: Sequence[Tripple]) -> tuple[dict[str, Tensor], int]:
        raise NotImplementedError


class VanillaDictCollator(BaseCollator):
    def __call__(self, tripples: Sequence[Tripple]) -> tuple[dict[str, Tensor], int]:
        l_batch = []
        r_batch = []
        tgt_batch = []
        for tripple in tripples:
            l_batch.append(tripple.l_ids)
            r_batch.append(tripple.r_ids)
            tgt_batch.append(tripple.tgt_ids)

            assert max(len(tripple.l_ids), len(tripple.r_ids), len(tripple.tgt_ids)) <= self.max_len

        # Get lengths for sorting/bucketing
        l_lengths = [len(seq) for seq in l_batch]
        r_lengths = [len(seq) for seq in r_batch]
        tgt_lengths = [len(seq) for seq in tgt_batch]

        batch_size = len(tripples)

        # Pad sequences to max length in batch
        l_padded = torch.nn.utils.rnn.pad_sequence(l_batch, batch_first=True, padding_value=0)
        r_padded = torch.nn.utils.rnn.pad_sequence(r_batch, batch_first=True, padding_value=0)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)

        l_mask = torch.zeros(batch_size, l_padded.size(1), dtype=torch.bool)
        r_mask = torch.zeros(batch_size, r_padded.size(1), dtype=torch.bool)
        tgt_mask = torch.zeros(batch_size, tgt_padded.size(1), dtype=torch.bool)

        for i, (l_len, r_len, tgt_len) in enumerate(zip(l_lengths, r_lengths, tgt_lengths)):
            l_mask[i, l_len:] = True  # Mask padding positions
            r_mask[i, r_len:] = True
            tgt_mask[i, tgt_len:] = True

        return {
            "l_ids": l_padded,
            "r_ids": r_padded,
            "tgt_ids": tgt_padded,
            "l_mask": l_mask,
            "r_mask": r_mask,
            "tgt_mask": tgt_mask,
            "l_lengths": torch.tensor(l_lengths),
            "r_lengths": torch.tensor(r_lengths),
            "tgt_lengths": torch.tensor(tgt_lengths),
        }, sum(tgt_lengths)

    def _pyrec_to_tensor(self, expr: rise.RecExpr) -> Tensor:
        tree_data = expr.to_data()

        return torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )


class DisentangledDictCollator(BaseCollator):
    def __call__(self, tripples: Sequence[Tripple]) -> tuple[dict[str, Tensor], int]:
        batch = {}

        # batched_data["distance"] = torch.stack([sample["distance"] for sample in batch])

        batch["l_ids"] = self._pad_1d([sample.l_ids for sample in tripples], False)
        batch["l_anc"] = self._pad_2d([sample.l_anc for sample in tripples if sample.l_anc is not None], False)
        batch["l_sib"] = self._pad_2d([sample.l_sib for sample in tripples if sample.l_sib is not None], False)
        batch["l_mask"] = (batch["l_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        batch["r_ids"] = self._pad_1d([sample.r_ids for sample in tripples], False)
        batch["r_anc"] = self._pad_2d([sample.r_anc for sample in tripples if sample.r_anc is not None], False)
        batch["r_sib"] = self._pad_2d([sample.r_sib for sample in tripples if sample.r_sib is not None], False)
        batch["r_mask"] = (batch["r_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        n_tokens = 0
        # The _y versions are always shifted right.
        # For matrices this means right and down.
        full_tgt_ids = self._pad_1d([sample.tgt_ids for sample in tripples], True)
        batch["tgt_ids"] = full_tgt_ids[:, :-1]
        batch["tgt_ids_y"] = full_tgt_ids[:, 1:]
        batch["tgt_mask"] = make_tgt_mask(batch["tgt_ids"], self.pad_id)

        full_tgt_anc = self._pad_2d([sample.tgt_anc for sample in tripples if sample.tgt_anc is not None], True)
        batch["tgt_anc"] = full_tgt_anc[:, :-1, :-1]
        batch["tgt_anc_y"] = full_tgt_anc[:, 1:, 1:]

        full_tgt_sib = self._pad_2d([sample.tgt_sib for sample in tripples if sample.tgt_sib is not None], True)
        batch["tgt_sib"] = full_tgt_sib[:, :-1, :-1]
        batch["tgt_sib_y"] = full_tgt_sib[:, 1:, 1:]

        n_tokens = int((full_tgt_ids != self.pad_id).data.sum())

        return batch, n_tokens

    def _pad_1d(self, samples: list[Tensor], extra_pad: bool) -> Tensor:
        """
        Pad sequences to the same length along one simple dimension
        Generate padding directions depending on max_len

        :param samples: List of input tensors to pad and stack
        :return: Padded and stacked samples
        """
        pad_len = self.max_len
        if extra_pad:
            pad_len += 1  # Extra padding since tgt will be shifted
        paddings = [(0, pad_len - s.size(-1)) for s in samples]
        padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]
        return torch.stack(padded_elements)

    def _pad_2d(self, samples: list[Tensor], extra_pad: bool) -> Tensor:
        """
        Find largest dimensions of the square 2-dimensional matrices (they are always square)
        Generate padding directions depending on largest element in batch

        :param samples: List of input tensors to pad and stack
        :return: Padded and stacked samples
        """
        pad_len = self.max_len
        if extra_pad:
            pad_len += 1  # Extra padding since tgt will be shifted
        paddings = [(0, pad_len - s.size(-1), 0, pad_len - s.size(-2)) for s in samples]
        padded_elements = [F.pad(s, p, "constant", self.pad_id) for s, p in zip(samples, paddings)]
        return torch.stack(padded_elements)


def mk_loaders(
    rank: int,
    world_size: int,
    dataset: TrippleDataSet,
    collator: BaseCollator,
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
