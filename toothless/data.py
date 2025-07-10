from pathlib import Path
from typing import Sequence
import json

import polars as pl
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.normalizers import BertNormalizer
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
# from tokenizers.pre_tokenizers import Split
# from tokenizers.normalizers import Strip
# from tokenizers.normalizers import Sequence as NormalizerSequence
# import matplotlib.pyplot as plt

from eggshell import rise  # type: ignore

import torch
from torch import Tensor
from torch import nn
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .args import DataArguments
from .vocab import BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN, SimpleVocab
from .utils import loading


class CustomDataset(data.Dataset):
    def __init__(self, conf: DataArguments):
        """
        :param k represents the max relative distance
        """
        self.json_root = Path(conf.data_path)
        self.sample_distance = conf.sample_distance
        self.k = conf.k
        self.force_reload = conf.force_reload
        self.sample_limit = conf.sample_limit
        torch.manual_seed(conf.rng_seed)

        self.cache = Path(conf.cache_dir) / Path(*self.json_root.parts[-2:])
        self.cache.mkdir(parents=True, exist_ok=True)

        self.raw_path = self.cache / "df_raw.parquet"
        self.vocab_path = self.cache / "vocab.json"

        if conf.sample_cache_dir is None:
            self.sample_cache = self.cache / "samples" / f"d{conf.sample_distance}"
            self.sample_cache_metadata_path = self.cache / "samples" / f"d{conf.sample_distance}_cache_metadata.json"
        else:
            self.sample_cache = Path(conf.sample_cache_dir) / f"d{conf.sample_distance}"
            self.sample_cache_metadata_path = (
                Path(conf.sample_cache_dir) / f"d{conf.sample_distance}_cache_metadata.json"
            )
        self.sample_cache.mkdir(parents=True, exist_ok=True)

        self._process_raw()
        self.vocab = self._build_vocab()
        self.len = self._process()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> dict[str, str]:
        # with ZipFile(self.zipped_samples) as zip_file:
        #     b = zip_file.read(f"{idx}.json")
        #     s = json.loads(b)
        with open(self.sample_cache / f"{idx}.json", encoding="utf-8") as f:
            s = json.load(f)
        return s

    def _process_raw(self):
        if not self.force_reload and self.raw_path.is_file():
            return

        df = loading.load_df(self.json_root)
        df.write_parquet(self.raw_path)

    def _process(self) -> int:
        if not self.force_reload and self.sample_cache_metadata_path.is_file():  # and self.zipped_samples.is_file():
            with open(self.sample_cache_metadata_path, encoding="utf-8") as p:
                metadata = json.load(p)

            json_files = list(self.sample_cache.glob("*.json"))
            if len(json_files) == metadata["n_samples"] and self.sample_distance == metadata["sample_distance"]:
                print("JSON Cache Usable!")
                return _min_none(self.sample_limit, len(json_files))

        print("JSON Cache *not* usable!")
        self.sample_cache.rmdir()
        self.sample_cache.mkdir()

        raw_data = pl.read_parquet(self.raw_path)
        expl_chains = raw_data.get_column("explanation_chain")

        index_tripples = [self._pick_fixed_distance_indices(len(chain) - 1) for chain in expl_chains]
        length = sum([len(chain_pairs) for chain_pairs in index_tripples])
        print(f"Total tripples: {length}")

        samples = []
        with tqdm(total=length, desc="Creating tripples...") as pbar:
            for chain, index_tripple in zip(expl_chains, index_tripples):
                for left_idx, middle_idx, right_idx in index_tripple:
                    sample = {
                        "left": str(chain[left_idx]),
                        "middle": str(chain[middle_idx]),
                        "right": str(chain[right_idx]),
                    }

                    samples.append(sample)
                    pbar.update()

        print(f"Total samples: {len(samples)} saved to disk")
        with open(self.sample_cache_metadata_path, mode="w", encoding="utf-8") as p:
            json.dump({"n_samples": len(samples), "sample_distance": self.sample_distance}, p)

        for i, sample in enumerate(tqdm(samples, desc="Saving to cache...")):
            with open(self.sample_cache / f"{i}.json", mode="w", encoding="utf-8") as p:
                json.dump(sample, p)

        print("Data processed!")

        return _min_none(self.sample_limit, len(samples))

    def _build_vocab(self) -> SimpleVocab:
        normal_tokens = rise.operators() + ["[constant]", "[variable]"]
        vocab = SimpleVocab(PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN, normal_tokens)
        vocab.save(self.vocab_path)
        return vocab

    def _pick_fixed_distance_indices(self, max_index: int) -> set[tuple[int, int, int]]:
        s = set()
        for start in range(0, max_index - self.sample_distance):
            end = start + self.sample_distance
            mid = start + (self.sample_distance // 2)
            s.add((start, mid, end))
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


def make_std_mask(tgt: Tensor, pad_id: int):
    "Create a mask to hide padding and future words."
    # unsqueeze to (16,1,128)
    tgt_mask = (tgt == pad_id).unsqueeze(-2)
    # print(f"padding mask dims {tgt_mask.size()}")
    # plt.imsave("padding_mask.png", tgt_mask.squeeze(1))
    # unsqueeze to (1,128,128)
    triangle_mask = _triangle_matrix(tgt.size(-1), device=tgt.device).unsqueeze(0)
    # print(f"triangle mask dimes {triangle_mask.size()}")
    # plt.imsave("triangle_mask.png", triangle_mask[0])
    # unsqueeze to (16,1,128,128)
    tgt_mask = (tgt_mask | triangle_mask).unsqueeze(1)
    # plt.imsave("combined_mask.png", tgt_mask[0][0])
    # print(f"tgt mask dims {tgt_mask.size()}")
    return tgt_mask


def _triangle_matrix(sz: int, device: torch.device | None = None) -> Tensor:
    m = torch.full((sz, sz), True, device=device, dtype=torch.bool)
    return torch.triu(m, diagonal=1)


class DictCollator:
    def __init__(self, pad_id: int, max_len: int, k: int, vocab: SimpleVocab):
        self.pad_id = pad_id
        self.max_len = max_len
        self.k = k
        self.vocab = vocab

    def __call__(self, tripples: Sequence[dict[str, str]]) -> tuple[dict[str, Tensor], int]:
        unpadded = [self._vectorize(sample["left"], sample["middle"], sample["right"]) for sample in tripples]
        # batch is a list of dictionaries
        batch = {}

        # batched_data["distance"] = torch.stack([sample["distance"] for sample in batch])

        batch["l_ids"] = self._pad_1d([sample["l_ids"] for sample in unpadded], False)
        batch["l_anc"] = self._pad_2d([sample["l_anc"] for sample in unpadded], False)
        batch["l_sib"] = self._pad_2d([sample["l_sib"] for sample in unpadded], False)
        batch["l_mask"] = (batch["l_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        batch["r_ids"] = self._pad_1d([sample["r_ids"] for sample in unpadded], False)
        batch["r_anc"] = self._pad_2d([sample["r_anc"] for sample in unpadded], False)
        batch["r_sib"] = self._pad_2d([sample["r_sib"] for sample in unpadded], False)
        batch["r_mask"] = (batch["r_ids"] == self.pad_id).unsqueeze(1).unsqueeze(1)

        n_tokens = 0
        # The _y versions are always shifted right.
        # For matrices this means right and down.
        full_tgt_ids = self._pad_1d([sample["tgt_ids"] for sample in unpadded], True)
        batch["tgt_ids"] = full_tgt_ids[:, :-1]
        batch["tgt_ids_y"] = full_tgt_ids[:, 1:]
        batch["tgt_mask"] = make_std_mask(batch["tgt_ids"], self.pad_id)

        full_tgt_anc = self._pad_2d([sample["tgt_anc"] for sample in unpadded], True)
        batch["tgt_anc"] = full_tgt_anc[:, :-1, :-1]
        batch["tgt_anc_y"] = full_tgt_anc[:, 1:, 1:]

        full_tgt_sib = self._pad_2d([sample["tgt_sib"] for sample in unpadded], True)
        batch["tgt_sib"] = full_tgt_sib[:, :-1, :-1]
        batch["tgt_sib_y"] = full_tgt_sib[:, 1:, 1:]

        n_tokens = int((full_tgt_ids != self.pad_id).data.sum())

        return batch, n_tokens

    def _vectorize(self, left: str, middle: str, right: str) -> dict:
        l_ids, l_anc, l_sib = self._pyrec_to_tensor(rise.RecExpr(left))
        tgt_ids, tgt_anc, tgt_sib = self._pyrec_to_tensor(rise.RecExpr(middle))
        r_ids, r_anc, r_sib = self._pyrec_to_tensor(rise.RecExpr(right))

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

    def _pyrec_to_tensor(self, expr: rise.RecExpr) -> tuple[Tensor, Tensor, Tensor]:
        tree_data = expr.to_data()

        ids = torch.tensor(
            [self.vocab.bos_token_id]
            + [self.vocab.token2id(node.name) for node in tree_data.nodes()]
            + [self.vocab.eos_token_id],
            dtype=torch.long,
        )
        padder = nn.ConstantPad2d(1, 0)
        anc_matrix = padder(torch.tensor(tree_data.anc_matrix(self.k), dtype=torch.long))
        sib_matrix = padder(torch.tensor(tree_data.sib_matrix(self.k), dtype=torch.long))

        return ids, anc_matrix, sib_matrix

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
    rank: int, world_size: int, dataset: CustomDataset, data_args: DataArguments, shuffle: bool = True
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
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

    collator = DictCollator(pad_id, data_args.max_len, data_args.k, dataset.vocab)

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


def partial_to_matrices(partial_tok: list[str], k: int) -> tuple[Tensor, Tensor]:
    tree_data = rise.GeneratedRecExpr(partial_tok).to_data()

    padder = nn.ConstantPad2d((1, 0, 1, 0), 0)
    anc_matrix = padder(torch.tensor(tree_data.anc_matrix(k), dtype=torch.long))
    sib_matrix = padder(torch.tensor(tree_data.sib_matrix(k), dtype=torch.long))
    return anc_matrix, sib_matrix


def split_off_special(partial_tok: list[str], vocab: SimpleVocab) -> list[str]:
    partial_tok = partial_tok[1:]
    for i, j in enumerate(partial_tok):
        if j in vocab.special_tokens:
            return partial_tok[:i]
    return partial_tok


def _min_none(a: None | int, b: None | int) -> int:
    match (a, b):
        case (None, None):
            raise ValueError((a, b))
        case (None, x) | (x, None):
            return x
        case (x, y):
            return min(x, y)
