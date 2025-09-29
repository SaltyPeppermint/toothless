import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizers import models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
import tokenizers
from tqdm.auto import tqdm
import polars as pl

from .args import DataArgs


SCHEMA = {"from": pl.UInt64, "to": pl.UInt64, "path": pl.String}


@dataclass
class Triple:
    start_ids: torch.Tensor
    guide_ids: torch.Tensor
    target_ids: torch.Tensor


class TripleDataSet(Dataset[Triple]):
    def __init__(self, conf: DataArgs, tokenizer_path: Path | None = None):
        """
        :param k represents the max relative distance
        """
        self.json_root = Path(conf.data_path)
        self.conf = conf
        torch.manual_seed(conf.rng_seed)

        self.cache = Path(conf.cache_dir) / Path(*self.json_root.parts[-2:])
        self.cache.mkdir(parents=True, exist_ok=True)

        self.index_table_cache_path = self.cache / "index_table_cache.csv"
        self.index_table = self._iterate_samples()

        if tokenizer_path:
            print("LOADING TOKENIZER FROM FILE")
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            self.tokenizer = _build_tokenizer(self.index_table, conf.tokenizer_samples, conf.force_reload)
        self.tokenizer_path = self.cache / "tokenizer.json"

        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_token_id = self.tokenizer.token_to_id("[CLS]")
        self.eos_token_id = self.tokenizer.token_to_id("[SEP]")
        assert self.pad_token_id == 0
        assert self.bos_token_id == 1
        assert self.eos_token_id == 2

    def _iterate_samples(self) -> pl.DataFrame:
        if self.index_table_cache_path.is_file() and not self.conf.force_reload:
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

    def _tokenize(self, text: str) -> list[int]:
        """Prepare a sequence with proper tokenization and padding."""
        ids = self.tokenizer.encode(text).ids

        # Error if too long
        if len(ids) > self.conf.max_len:
            raise ValueError(f"Too long: {text}")

        return ids

    def __getitem__(self, idx: int) -> Triple:
        result = self.index_table.filter((pl.col("from") <= idx) & (idx < pl.col("to"))).row(0, named=True)

        with open(str(result["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        start = file["start_expr"]
        guide = file["midpoint"]["midpoint"]["expression"]
        target = file["midpoint"]["goals"][idx - result["from"]]["expression"]

        return Triple(
            torch.tensor(self._tokenize(start), dtype=torch.long),
            torch.tensor(self._tokenize(guide), dtype=torch.long),
            torch.tensor(self._tokenize(target), dtype=torch.long),
        )

    def __len__(self) -> int:
        total_samples = self.index_table["to"].max()
        if self.conf.n_samples:
            return min(total_samples, self.conf.n_samples)  # pyright: ignore[reportArgumentType]
        return total_samples  # type: ignore


def _get_tokenizer_training_corpus(index_table: pl.DataFrame, n_samples: int):
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


def _build_tokenizer(index_table: pl.DataFrame, n_samples: int, force_reload: bool) -> Tokenizer:
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
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )  # pyright: ignore[reportAttributeAccessIssue]

    trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["[PAD]", "[CLS]", "[SEP]"])  # pyright: ignore[reportCallIssue]

    tokenizer.train_from_iterator(
        _get_tokenizer_training_corpus(index_table, n_samples), trainer=trainer, length=n_samples
    )

    return tokenizer
