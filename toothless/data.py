import json
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm
import polars as pl

from .args import DataArgs


SCHEMA = {"from": pl.UInt64, "to": pl.UInt64, "path": pl.String}


@dataclass
class Triple:
    start: str
    guide: str
    target: str


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

    def __getitem__(self, idx: int) -> Triple:
        result = self.index_table.filter((pl.col("from") <= idx) & (idx < pl.col("to"))).row(0, named=True)

        with open(str(result["path"]), mode="r", encoding="utf-8") as f:
            file = json.load(f)
        start = file["start_expr"]
        guide = file["midpoint"]["midpoint"]["expression"]
        target = file["midpoint"]["goals"][idx - result["from"]]["expression"]
        return Triple(start, guide, target)

    def __len__(self) -> int:
        total_samples = self.index_table["to"].max()
        if self.n_samples is not None:
            return min(total_samples, self.n_samples)  # pyright: ignore[reportArgumentType]
        return total_samples  # type: ignore

    def get_tokenizer_training_corpus(self, n_samples: int):
        i = 0
        for entry in self.index_table.iter_rows(named=True):
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
