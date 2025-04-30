from pathlib import Path

import polars as pl

from eggshell import rise  # type: ignore

# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.normalizers import BertNormalizer
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
# from tokenizers.pre_tokenizers import Split
# from tokenizers.normalizers import Strip
# from tokenizers.normalizers import Sequence as NormalizerSequence
# import matplotlib.pyplot as plt


import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader


from toothless.tree_model.args import DataArguments
from toothless.tree_model.vocab import BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN, SimpleVocab
from toothless.utils import loading
from toothless.utils.dist_helper import rank0print

CHUNK_SIZE = 128


class CustomDataset(data.Dataset):
    def __init__(self, conf: DataArguments, rank: int):
        """
        :param k represents the max relative distance
        """
        self.json_root = Path(conf.data_path)
        self.sample_distance = conf.sample_distance
        self.k = conf.k
        self.force_reload = conf.force_reload
        self.len_limit = conf.data_limit
        torch.manual_seed(conf.random_state)

        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._process_raw(rank)
        self.vocab = self._build_vocab(rank)
        self.samples = self._process(rank)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> dict[str, Tensor]:
        sample = self.get_tuple_as_str(idx)

        vectorized = self._vectorize(sample["start"], sample["guide"], sample["goal"])
        return vectorized

    def get_tuple_as_str(self, idx) -> dict[str, str]:
        # FIXME THIS IS A MIXUP OF START AND GUIDE
        sample = self.samples[idx]
        goal = sample["start"].item()
        start = sample["goal"].item()
        guide = sample["guide"].item()
        return {"start": start, "goal": goal, "guide": guide}

    def _process_raw(self, rank: int):
        if not self.force_reload and self.raw_path.is_file():
            return

        df = loading.load_df(self.json_root, rank)
        df.write_parquet(self.raw_path)

    def _process(self, rank: int) -> pl.DataFrame:
        if not self.force_reload and self.processed_path.is_file():
            df = pl.read_parquet(self.processed_path)
            if self.len_limit:
                return df.limit(self.len_limit)
            return df

        raw_data = pl.read_parquet(self.raw_path)
        expl_chains = raw_data.get_column("explanation_chain")

        # picked_tripples = [self._pick_recursive_indices(len(chain) - 1) for chain in expl_chains]
        picked_tripples = [self._pick_fixed_distance_indices(len(chain) - 1) for chain in expl_chains]
        length = sum([len(chain_pairs) for chain_pairs in picked_tripples])
        rank0print(rank, f"Total pairs: {length}")

        total_samples = {"start": [], "goal": [], "guide": [], "distance": []}
        for chain, tripple in zip(expl_chains, picked_tripples):
            for left, middle, right in tripple:
                right = int(right)
                left = int(left)
                middle = int(middle)
                total_samples["start"].append(str(chain[left]))
                total_samples["goal"].append(str(chain[right]))
                total_samples["guide"].append(str(chain[middle]))
                total_samples["distance"].append(middle / (right - left))

        df = pl.DataFrame(total_samples)
        df.write_parquet(self.processed_path)
        rank0print(rank, f"Total samples: {len(df)}")

        if self.len_limit:
            df = df.limit(self.len_limit)

        rank0print(rank, f"Using {len(df)} samples!")

        rank0print(rank, "Data processed!")
        return df

    def _build_vocab(self, _rank: int) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    # def _gen(self, expl_chains, picked_tripples) -> Iterator[list[str]]:
    #     for chain, tripple in zip(expl_chains, picked_tripples):
    #         for left, middle, right in tripple:
    #             yield chain[left].to_data().values()
    #             yield chain[middle].to_data().values()
    #             yield chain[right].to_data().values()
    def _pick_fixed_distance_indices(self, max_index: int) -> set[tuple[int, int, int]]:
        s = set()
        starts = range(0, max_index - self.sample_distance)
        ends = range(self.sample_distance, max_index)
        for start, end in zip(starts, ends):
            midpoint = start + (self.sample_distance // 2)
            s.add((start, midpoint, end))
        return s

    def _pick_recursive_indices(self, max_index: int) -> set[tuple[int, int, int]]:
        def rec(start: int, end: int, acc: set[tuple[int, int, int]], min_distance):
            distance = end - start
            if distance < min_distance:
                return
            else:
                midpoint = start + (distance // 2)
                acc.add((start, midpoint, end))
                rec(start, midpoint, acc, min_distance)
                rec(midpoint, end, acc, min_distance)

        acc = set()
        rec(0, max_index, acc, self.sample_distance)
        return acc

    def _vectorize(self, left: str, middle: str, right: str) -> dict:
        l_ids, l_anc, l_sib = pyrec_to_tensor(rise.PyRecExpr(left), self.vocab, self.k)
        tgt_ids, tgt_anc, tgt_sib = pyrec_to_tensor(rise.PyRecExpr(middle), self.vocab, self.k)
        r_ids, r_anc, r_sib = pyrec_to_tensor(rise.PyRecExpr(right), self.vocab, self.k)

        return {
            "l_ids": l_ids,
            "l_anc": l_anc,
            "l_sib": l_sib,
            "tgt_ids": tgt_ids,
            "tgt_anc": tgt_anc,
            "tgt_sib": tgt_sib,
            "r_ids": r_ids,
            "r_anc": r_anc,
            "r_sib": r_sib,
        }

    @property
    def raw_path(self) -> Path:
        return self.cache_dir / Path("df_raw.parquet")

    @property
    def vocab_path(self) -> Path:
        return self.cache_dir / Path("vocab.json")

    @property
    def processed_path(self) -> Path:
        return self.cache_dir / Path("processed.parquet")


def pyrec_to_tensor(expr: rise.PyRecExpr, vocab: SimpleVocab, k: int) -> tuple[Tensor, Tensor, Tensor]:
    tree_data = expr.to_data()

    ids = torch.tensor(
        [vocab.bos_token_id] + [vocab.token2id(node.name) for node in tree_data.nodes()] + [vocab.eos_token_id],
        dtype=torch.long,
    )

    anc_matrix = torch.tensor(tree_data.anc_matrix(k, double_pad=True), dtype=torch.long)
    sib_matrix = torch.tensor(tree_data.sib_matrix(k, double_pad=True), dtype=torch.long)

    return ids, anc_matrix, sib_matrix


def make_std_mask(tgt: Tensor, pad_id: int):
    "Create a mask to hide padding and future words."
    # unsqueeze to (16,1,128)
    tgt_mask = (tgt == pad_id).unsqueeze(-2)
    # print(f"padding mask dims {tgt_mask.size()}")
    # plt.imsave("padding_mask.png", tgt_mask.squeeze(1))
    # unsqueeze to (1,128,128)
    triangle_mask = triangle_matrix(tgt.size(-1), device=tgt.device).unsqueeze(0)
    # print(f"triangle mask dimes {triangle_mask.size()}")
    # plt.imsave("triangle_mask.png", triangle_mask[0])
    # unsqueeze to (16,1,128,128)
    tgt_mask = (tgt_mask | triangle_mask).unsqueeze(1)
    # plt.imsave("combined_mask.png", tgt_mask[0][0])
    # print(f"tgt mask dims {tgt_mask.size()}")
    return tgt_mask


def triangle_matrix(sz: int, device: torch.device | None = None) -> Tensor:
    m = torch.full((sz, sz), True, device=device, dtype=torch.bool)
    return torch.triu(m, diagonal=1)


class DictCollator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(self, batch: list[dict[str, Tensor]]) -> tuple[dict[str, Tensor], int]:
        # batch is a list of dictionaries
        batched_data = {}

        # batched_data["distance"] = torch.stack([sample["distance"] for sample in batch])

        batched_data["l_ids"] = self.pad_1d([sample["l_ids"] for sample in batch], False)
        batched_data["l_anc"] = self.pad_2d([sample["l_anc"] for sample in batch], False)
        batched_data["l_sib"] = self.pad_2d([sample["l_sib"] for sample in batch], False)
        batched_data["l_mask"] = (batched_data["l_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        batched_data["r_ids"] = self.pad_1d([sample["r_ids"] for sample in batch], False)
        batched_data["r_anc"] = self.pad_2d([sample["r_anc"] for sample in batch], False)
        batched_data["r_sib"] = self.pad_2d([sample["r_sib"] for sample in batch], False)
        batched_data["r_mask"] = (batched_data["r_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        n_tokens = 0
        if all(["tgt_ids" in batch[0], "tgt_anc" in batch[0], "tgt_sib" in batch[0]]):
            # The _y versions are always shifted right.
            # For matrices this means right and down.
            full_tgt_ids = self.pad_1d([sample["tgt_ids"] for sample in batch], True)
            batched_data["tgt_ids"] = full_tgt_ids[:, :-1]
            batched_data["tgt_ids_y"] = full_tgt_ids[:, 1:]
            batched_data["tgt_mask"] = make_std_mask(batched_data["tgt_ids"], self.pad_id)

            full_tgt_anc = self.pad_2d([sample["tgt_anc"] for sample in batch], True)
            batched_data["tgt_anc"] = full_tgt_anc[:, :-1, :-1]
            batched_data["tgt_anc_y"] = full_tgt_anc[:, 1:, 1:]

            full_tgt_sib = self.pad_2d([sample["tgt_sib"] for sample in batch], True)
            batched_data["tgt_sib"] = full_tgt_sib[:, :-1, :-1]
            batched_data["tgt_sib_y"] = full_tgt_sib[:, 1:, 1:]

            n_tokens = int((full_tgt_ids != self.pad_id).data.sum())
        return batched_data, n_tokens

    def pad_1d(self, samples: list[Tensor], extra_pad: bool) -> Tensor:
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

    def pad_2d(self, samples: list[Tensor], extra_pad: bool) -> Tensor:
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
    rank: int, world_size: int, dataset: CustomDataset, data_args: DataArguments, shuffle: bool = True
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
    # Create and load dataset
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [data_args.split_size, 1 - data_args.split_size]
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)
    eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)

    pad_id = dataset.vocab.pad_token_id
    assert pad_id == 0

    collator = DictCollator(pad_id, data_args.max_len)

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=data_args.batch_size,
        sampler=eval_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    return train_dataloader, eval_dataloader


def partial_to_matrices(partial_tok: list[str], k: int) -> tuple[Tensor, Tensor]:
    tree_data = rise.partial_parse(partial_tok)
    anc_matrix = torch.tensor(tree_data.anc_matrix(k, double_pad=True), dtype=torch.long)
    sib_matrix = torch.tensor(tree_data.sib_matrix(k, double_pad=True), dtype=torch.long)
    return anc_matrix[:-1, :-1], sib_matrix[:-1, :-1]


def split_off_special(partial_tok: list[str], vocab: SimpleVocab) -> list[str]:
    partial_tok = partial_tok[1:]
    for i, j in enumerate(partial_tok):
        if j in vocab.special_tokens:
            return partial_tok[:i]
    return partial_tok
