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

        self.tokenizer = _build_tokenizer(self.index_table, self.cache, conf.tokenizer_samples)

        self.tokenizer.enable_padding(length=self.max_len)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_token_id = self.tokenizer.token_to_id("[CLS]")
        self.eos_token_id = self.tokenizer.token_to_id("[SEP]")
        assert self.pad_token_id == 0
        assert self.bos_token_id == 1
        assert self.eos_token_id == 2

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

        # Pad if too short
        if len(ids) < self.max_len:
            ids.extend([self.pad_token_id] * (self.max_len - len(ids)))

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

        l_ids, l_mask = self._prepare_sequence(start)
        tgt_ids, _ = self._prepare_sequence(guide)
        r_ids, r_mask = self._prepare_sequence(target)

        decoder_input_ids = [self.bos_token_id] + tgt_ids[1:-1] if len(tgt_ids) > 1 else [self.bos_token_id]

        # Ensure decoder input has correct length
        if len(decoder_input_ids) < self.max_len:
            decoder_input_ids.extend([self.pad_token_id] * (self.max_len - len(decoder_input_ids)))
        elif len(decoder_input_ids) > self.max_len:
            decoder_input_ids = decoder_input_ids[: self.max_len]

        tgt_mask = [token_id != self.pad_token_id for token_id in decoder_input_ids]

        tensor_dict = {
            "l_ids": torch.tensor(l_ids, dtype=torch.long),
            "l_mask": torch.tensor(l_mask, dtype=torch.bool),
            "r_ids": torch.tensor(r_ids, dtype=torch.long),
            "r_mask": torch.tensor(r_mask, dtype=torch.bool),
            "tgt_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.bool),
            "labels": torch.tensor(tgt_ids, dtype=torch.long),
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


def _build_tokenizer(index_table: pl.DataFrame, cache: Path, n_samples: int) -> Tokenizer:
    tokenizer_path = cache / "tokenizer.json"
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
    tokenizer.save(str(tokenizer_path))
    return tokenizer
