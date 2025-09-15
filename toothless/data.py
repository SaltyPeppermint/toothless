import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm
import polars as pl

from .args import DataArgs

# import matplotlib.pyplot as plt
from tokenizers import models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import tokenizers


SCHEMA = {"from": pl.UInt64, "to": pl.UInt64, "path": pl.String}


@dataclass
class Triple:
    l_ids: list[int]
    l_str: str
    tgt_ids: list[int]
    tgt_str: str
    r_ids: list[int]
    r_str: str


class TripleDataSet(Dataset[Triple]):
    def __init__(self, conf: DataArgs):
        """
        :param k represents the max relative distance
        """
        self.json_root = Path(conf.data_path)
        self.force_reload = conf.force_reload
        torch.manual_seed(conf.rng_seed)

        self.cache = Path(conf.cache_dir) / Path(*self.json_root.parts[-2:])
        self.cache.mkdir(parents=True, exist_ok=True)

        self.index_table_cache_path = self.cache / "index_table_cache.csv"
        self.index_table = self._iterate_samples()

        self.n_samples = conf.n_samples

        self.vocab_path = self.cache / "vocab.json"
        self.tokenizer = self._build_vocab(self.vocab_path, self.index_table, conf.tokenizer_samples)

    def _iterate_samples(self) -> pl.DataFrame:
        if self.index_table_cache_path.is_file():
            return pl.read_csv(self.index_table_cache_path, schema=SCHEMA)

        json_files = list(self.json_root.glob("*.json"))
        json_files.sort()

        entries = []
        i = 0

        for batch_file in tqdm(json_files, desc="Enumerating tripples"):
            with open(batch_file, mode="r", encoding="utf-8") as f:
                file = json.load(f)
            j = i + len(file["midpoint"]["goals"])
            entries.append([i, j, str(batch_file)])

            i = j
        df = pl.DataFrame(entries, schema=SCHEMA, orient="row")
        df.write_csv(self.index_table_cache_path)

        return df

    def _load_str_triple(self, idx: int) -> tuple[str, str, str]:
        result = self.index_table.filter((pl.col("from") <= idx) & (idx < pl.col("to"))).row(0, named=True)

        with open(str(result["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        start = file["start_expr"]
        guide = file["midpoint"]["midpoint"]["expression"]
        target = file["midpoint"]["goals"][idx - result["from"]]["expression"]
        return start, guide, target

    def _build_vocab(self, vocab_path: Path, index_table: pl.DataFrame, n_samples: int) -> Tokenizer:
        if self.vocab_path.is_file():
            return Tokenizer.from_file(str(self.vocab_path))

        tokenizer = Tokenizer(models.BPE())

        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), normalizers.Replace(tokenizers.Regex(r"mf(i|u)\d*"), "[var]")]  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Split(")", behavior="isolated"),
                pre_tokenizers.Split("(", behavior="isolated"),
            ]
        )  # pyright: ignore[reportAttributeAccessIssue]
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<CLS> $A <SEP>", pair="<CLS> $A <SEP> $B:1 <SEP>:1", special_tokens=[("<CLS>", 1), ("<SEP>", 2)]
        )  # pyright: ignore[reportAttributeAccessIssue]

        trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<PAD>", "<CLS>", "<SEP>"])  # pyright: ignore[reportCallIssue]

        tokenizer.train_from_iterator(get_training_corpus(index_table, n_samples), trainer=trainer, length=n_samples)
        tokenizer.save(str(vocab_path))
        return tokenizer

    def __getitem__(self, idx: int) -> Triple:
        left, middle, right = self._load_str_triple(idx)

        l_ids = self.tokenizer.encode(left).ids
        tgt_ids = self.tokenizer.encode(middle).ids
        r_ids = self.tokenizer.encode(right).ids

        return Triple(l_ids, left, tgt_ids, middle, r_ids, right)

    def __len__(self) -> int:
        total_samples = self.index_table["to"].max()
        if self.n_samples is not None:
            return min(total_samples, self.n_samples)  # pyright: ignore[reportArgumentType]
        return total_samples  # type: ignore


def get_training_corpus(index_table: pl.DataFrame, n_samples: int):
    i = 0
    for entry in index_table.iter_rows(named=True):
        if i > n_samples:
            break
        with open(str(entry["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        if i == 0:
            i += 1
            yield file["start_expr"]

        i += 1
        yield file["midpoint"]["midpoint"]["expression"]
        for goal in file["midpoint"]["goals"]:
            i += 1
            yield goal["expression"]
