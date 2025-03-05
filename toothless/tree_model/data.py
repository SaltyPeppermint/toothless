from functools import lru_cache
from pathlib import Path
import json
from typing import Iterator

import torch
from torch import Tensor
from torch.utils import data

from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence as PretokenizerSequence, Split
from tokenizers.normalizers import Sequence as NormalizerSequence, Replace, BertNormalizer, Strip

import polars as pl

from eggshell import rise  # type: ignore

# from eggshell import TreeData
from toothless.utils import loading


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SEP_TOKEN = "<sep>"
MASK_TOKEN = "<mask>"


class WrappedTokenizer:
    def __init__(self, var_names: list[str], ignore_unknown: bool):
        self.ignore_unknown = ignore_unknown
        self.var_names = var_names
        self.offset = rise.num_symbols() + len(var_names) + int(ignore_unknown)
        self.inner_tokenizer = None

    def load(self, path: Path):
        self.inner_tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)).save(path)

    def save(self, path: Path):
        if self.inner_tokenizer:
            self.inner_tokenizer.save(path)
        else:
            raise ValueError("Cant save unless trained or loaded")


class CustomDataset(data.Dataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        random_state: int = 0,
        shards: int = 42,
        transform=None,
        force_reload: bool = False,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl

        self.random_state = random_state
        self.shards = shards

        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transform
        self.force_reload = force_reload

        self._download()
        self.length, self.max_shard_size, self.tokenizer = self._process()
        self.max_chunk_size = _max_shard_size(self.length, self.shards)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        # TODO Implement an iterator for this
        chunk_path = Path(self.processed_paths[idx // self.max_chunk_size])
        chunk = _load_shard(chunk_path)
        sample = chunk[idx % self.max_chunk_size]
        sample = {k: v.to_dense() for k, v in sample.items()}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _download(self):
        if not self.force_reload and all([f.is_file() for f in self.raw_paths]):
            return

        df = loading.load_df(self.json_root)
        df.write_parquet(self.raw_paths[0])
        return

    def _process(self) -> tuple[int, int, Tokenizer]:
        if self.is_processed():
            with open(Path(self.processed_dir) / Path("meta.json")) as f:
                j = json.load(f)
                length = j["len"]
            tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
            tokenizer.from_file(self.tokenizer_path)
            return j["len"], j["max_shard_size"], tokenizer

        raw_expl_chains = pl.read_parquet(self.raw_paths[0]).get_column("explanation_chain")
        torch.manual_seed(self.random_state)

        print("Starting parallel processing")
        expl_chains = [rise.PyRecExpr.batch_new(i) for i in raw_expl_chains]
        print("Parallel processing done")

        picked_tripples = [self._pick_indices(len(chain)) for chain in expl_chains]
        length = sum([len(chain_pairs) for chain_pairs in picked_tripples])
        print(f"Total pairs: {length}")
        max_shard_size = _max_shard_size(length, self.shards)
        print(f"Chunk size: {max_shard_size}")

        tokenizer = self._train_tokenizer(expl_chains, picked_tripples)

        save_buffer = []
        spill_buffer = []
        idx = 0
        for chain, tripple in zip(expl_chains, picked_tripples):
            pairs = [
                self._vectorize(chain[left], chain[middle], chain[right], middle / (right - left), tokenizer)
                for left, middle, right in tripple
            ]

            remaining_space = max_shard_size - len(save_buffer)
            save_buffer.extend(pairs[:remaining_space])
            spill_buffer.extend(pairs[remaining_space:])

            if remaining_space == 0:
                print(f"Saving {len(save_buffer)} pairs in {self.processed_paths[idx]}")
                torch.save(save_buffer, self.processed_paths[idx])

                save_buffer = spill_buffer
                spill_buffer = []
                idx += 1

        # Save last few in the buffer if exist
        if len(save_buffer) != 0:
            print(f"Saving the last {len(save_buffer)} pairs in {self.processed_paths[idx]}")
            torch.save(save_buffer, self.processed_paths[idx])

        meta_path = Path(self.processed_dir) / Path("meta.json")
        with open(meta_path, mode="w", encoding="utf-8") as f:
            json.dump({"len": length, "max_shard_size": max_shard_size}, f)

        print("Data processed!")
        return length, max_shard_size, tokenizer

    def _train_tokenizer(self, expl_chains, picked_tripples) -> Tokenizer:
        if self.tokenizer_path.exists() and not self.force_reload:
            return Tokenizer.from_file(str(self.tokenizer_path))

        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.normalizer = NormalizerSequence(
            [
                Strip(),
                BertNormalizer(clean_text=True, strip_accents=True, lowercase=True),
            ]  # type: ignore
        )  # type: ignore
        tokenizer.pre_tokenizer = PretokenizerSequence([Split(" ", behavior="removed")])  # type: ignore

        trainer = BpeTrainer(
            vocab_size=1000,  # type: ignore
            special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, PAD_TOKEN, MASK_TOKEN],  # type: ignore
        )
        iterator = self.gen(expl_chains, picked_tripples)
        tokenizer.train_from_iterator(
            iterator=iterator, trainer=trainer, length=len(expl_chains) * 3 * self.pairs_per_expl * 50
        )
        tokenizer.add_tokens(rise.operators())
        tokenizer.add_tokens(["[constant]", "[variable]"])
        tokenizer.save(str(self.tokenizer_path))
        return tokenizer

    def gen(self, expl_chains, picked_tripples) -> Iterator[list[str]]:
        for chain, tripple in zip(expl_chains, picked_tripples):
            for left, middle, right in tripple:
                yield chain[left].to_data().values()
                yield chain[middle].to_data().values()
                yield chain[right].to_data().values()

    def _pick_indices(self, max_index: int) -> set[tuple[int, int, int]]:
        xs = torch.randint(0, max_index, (self.pairs_per_expl,))
        ys = torch.randint(0, max_index, (self.pairs_per_expl,))
        r = set()
        for x, y in zip(xs, ys):
            distance = abs(x - y)
            if distance < 2:
                continue
            if x < y:
                r.add((x, distance // 2, y))
            else:
                r.add((y, distance // 2, x))

        return r

    def _vectorize(
        self, left: rise.PyRecExpr, middle: rise.PyRecExpr, right: rise.PyRecExpr, distance: float, tokenizer: Tokenizer
    ) -> dict:
        x_l, v_l, adjacency_s = self._pyrec_to_tensor(left, tokenizer)
        x_m, v_m, adjacency_m = self._pyrec_to_tensor(middle, tokenizer)
        x_r, v_r, adjacency_r = self._pyrec_to_tensor(right, tokenizer)

        return {
            "x_l": x_l,
            "v_l": v_l,
            "adjacency_l": adjacency_s,
            "x_m": x_m,
            "v_m": v_m,
            "adjacency_m": adjacency_m,
            "x_r": x_r,
            "v_r": v_r,
            "adjacency_r": adjacency_r,
            "distance": torch.tensor([distance]),
        }

    def _pyrec_to_tensor(self, expr: rise.PyRecExpr, tokenizer: Tokenizer) -> tuple[Tensor, Tensor, Tensor]:
        graph_data = expr.to_data()
        node_ids = torch.tensor([node.id for node in graph_data.nodes], dtype=torch.int32)
        tokenized_values = [
            tokenizer.encode(BOS_TOKEN + node.name + SEP_TOKEN + node.value + EOS_TOKEN)
            if node.value
            else tokenizer.encode(BOS_TOKEN + node.name + EOS_TOKEN)
            for node in graph_data.nodes
        ]

        adjacency = graph_data.transposed_adjacency()
        n_edges = len(adjacency[0])
        adjacency_matrix = torch.sparse_coo_tensor(
            torch.tensor(adjacency),
            torch.full((n_edges,), True, dtype=torch.bool),
            dtype=torch.bool,
            size=torch.Size((150, 150)),
        )

        return node_ids, tokenized_values, adjacency_matrix

    def is_processed(self) -> bool:
        return (
            not self.force_reload
            and all([f.is_file() for f in self.processed_paths])
            and self.tokenizer_path.is_file()
            and self.meta_path.is_file()
        )

    @property
    def raw_paths(self) -> list[Path]:
        return [self.cache_dir / Path("df_raw.parquet")]

    @property
    def processed_dir(self) -> Path:
        processed_dir = self.cache_dir / Path("processed")
        processed_dir.mkdir(exist_ok=True)
        return processed_dir

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / Path("meta.json")

    @property
    def tokenizer_path(self) -> Path:
        return self.cache_dir / Path("tokenizer.json")

    @property
    def processed_paths(self) -> list[Path]:
        return [self.processed_dir / Path(f"data_{i}.pt") for i in range(self.shards)]


@lru_cache(maxsize=100)
def _load_shard(chunk_path: Path):
    return torch.load(chunk_path, weights_only=False)


def _max_shard_size(total_length_int, num_shards):
    # Get number of full shards -> round up
    return -(-total_length_int // num_shards)


if __name__ == "__main__":
    # expr = rise.PyRecExpr("(app (app zip (var a)) 1)))")
    # x_s, edge_index_s = PairData._pyrec_to_feature_vec(expr, ["a"], True)
    # print(x_s)
    # print(edge_index_s)
    a = 2
