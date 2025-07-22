from pathlib import Path
import shutil
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

from tqdm.auto import tqdm

from .args import DataArguments
from .vocab import BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN, SimpleVocab
from . import loading


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
        shutil.rmtree(self.sample_cache)
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
