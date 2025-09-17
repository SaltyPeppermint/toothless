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
    start: str
    guide: str
    target: str
    tensor_dict: dict[str, torch.Tensor]


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
        self.max_len = conf.max_len

        self.tokenizer_path = self.cache / "tokenizer.json"
        self.tokenizer = _build_tokenizer(self.index_table, self.tokenizer_path, conf.tokenizer_samples)
        assert self.tokenizer.token_to_id("[PAD]") == 0
        assert self.tokenizer.token_to_id("[CLS]") == 1
        assert self.tokenizer.token_to_id("[SEP]") == 2
        self.tokenizer.save(str(self.tokenizer_path))

        self.tokenizer.enable_padding(length=self.max_len)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_token_id = self.tokenizer.token_to_id("[CLS]")
        self.eos_token_id = self.tokenizer.token_to_id("[SEP]")

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

    def _prepare_sequence(self, text: str) -> tuple[list[int], list[bool]]:
        """Prepare a sequence with proper tokenization and padding."""
        ids = self.tokenizer.encode(text).ids

        # Truncate if too long
        if len(ids) > self.max_len:
            raise ValueError(f"Too long: {text}")

        # Create attention mask
        mask = [token_id != self.pad_token_id for token_id in ids]

        return ids, mask

    def __getitem__(self, idx: int) -> Triple:
        result = self.index_table.filter((pl.col("from") <= idx) & (idx < pl.col("to"))).row(0, named=True)

        with open(str(result["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        start = file["start_expr"]
        guide = file["midpoint"]["midpoint"]["expression"]
        target = file["midpoint"]["goals"][idx - result["from"]]["expression"]

        start_ids, start_mask = self._prepare_sequence(start)
        guide_ids, guide_mask = self._prepare_sequence(guide)
        target_ids, target_mask = self._prepare_sequence(target)

        tensor_dict = {
            "start_ids": torch.tensor(start_ids, dtype=torch.long),
            "start_mask": torch.tensor(start_mask, dtype=torch.bool),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "guide_ids": torch.tensor(guide_ids, dtype=torch.long),
            "guide_mask": torch.tensor(guide_mask, dtype=torch.bool),
        }
        return Triple(start, guide, target, tensor_dict)

    def __len__(self) -> int:
        total_samples = self.index_table["to"].max()
        if self.n_samples is not None:
            return min(total_samples, self.n_samples)  # pyright: ignore[reportArgumentType]
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


def _build_tokenizer(index_table: pl.DataFrame, tokenizer_path: Path, n_samples: int) -> Tokenizer:
    if tokenizer_path.is_file():
        return Tokenizer.from_file(str(tokenizer_path))

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
