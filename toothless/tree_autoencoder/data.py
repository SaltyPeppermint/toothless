from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import json

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import nn

from tensordict import TensorDict

import polars as pl


from eggshell import rise  # type: ignore
from toothless.utils import loading


class PairDataset(Dataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        var_names: list[str],
        ignore_unknown: bool,
        random_state: int = 0,
        shards: int = 50,
        transform=None,
        force_reload: bool = False,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl

        self.var_names = var_names
        self.ignore_unknown = ignore_unknown
        self.random_state = random_state
        self.shards = shards

        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transform
        self.force_reload = force_reload
        self.len_memo = None

        self._download()
        self._process()

    def __len__(self) -> int:
        if self.len_memo is None:
            with open(Path(self.processed_dir) / Path("meta.json")) as f:
                j = json.load(f)
                self.len_memo = j["len"]
                return j["len"]
        else:
            return self.len_memo

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # samples = []
        # for i in range(len(self))[idx]:  # type: ignore
        #     max_chunk_size = self._max_shard_size()
        #     chunk_path = Path(self.processed_paths[i // max_chunk_size])
        #     chunk = _load_shard(chunk_path)
        #     sample = chunk[i % max_chunk_size]
        #     samples.append(sample)
        max_chunk_size = self._max_shard_size()
        chunk_path = Path(self.processed_paths[idx // max_chunk_size])
        chunk = _load_shard(chunk_path)
        sample = chunk[idx % max_chunk_size]
        sample = {k: v.to_dense() for k, v in sample.items()}

        if self.transform:
            sample = self.transform(sample)
        return sample

    # TODO Implement an iterator for this
    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:  # single-process data loading, return the full iterator
    #         iter_start = 0
    #         iter_end = len(self)
    #     else:  # in a worker process
    #         # split workload
    #         per_worker = int(math.ceil((len(self)) / float(worker_info.num_workers)))
    #         worker_id = worker_info.id
    #         iter_start = worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, len(self))
    #     return iter(range(iter_start, iter_end))

    def _download(self):
        if not self.force_reload and all([f.is_file() for f in self.raw_paths]):
            return

        df = loading.load_df(self.json_root, self.var_names)
        df.write_parquet(self.raw_paths[0])
        return

    def _process(self):
        # Read data into huge `Data` list.
        if (
            not self.force_reload
            and all([f.is_file() for f in self.processed_paths])
            and self.processed_meta_path.is_file()
        ):
            return

        raw_expl_chains = pl.read_parquet(self.raw_paths[0]).get_column("explanation_chain")
        torch.manual_seed(self.random_state)

        expl_chains = [rise.PyRecExpr.batch_new(i) for i in raw_expl_chains]
        print("Parallel processing done")

        picked_pairs = [self._pick_indices(len(chain)) for chain in expl_chains]

        self.len_memo = sum([len(chain_pairs) for chain_pairs in picked_pairs])

        print(f"Total pairs: {self.len_memo}")
        print(f"Chunk size: {self._max_shard_size()}")

        save_buffer = []
        spill_buffer = []
        idx = 0

        for chain, picked_pairs in zip(expl_chains, picked_pairs):
            pairs = [self._from_tripple(chain[a], chain[b], b - a) for a, b in picked_pairs]

            remaining_space = self._max_shard_size() - len(save_buffer)
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
            json.dump({"len": len(self), "max_shard_size": self._max_shard_size()}, f)

        print("Data processed!")
        return

    # def _load(self) -> list[TensorDict]:
    #     big_buf = []
    #     for shard_path in self.processed_paths:
    #         print(f"Loading batch {shard_path}...")
    #         big_buf.extend(_load_shard(shard_path))
    #     print("Data loaded!")
    #     return big_buf

    def _max_shard_size(self):
        # Get number of full shards -> round up
        return -(-len(self) // self.shards)

    def _pick_indices(self, max_index: int) -> set[tuple[int, int]]:
        xs = torch.randint(0, max_index, (self.pairs_per_expl,))
        ys = torch.randint(0, max_index, (self.pairs_per_expl,))
        return set((int(x), int(y)) if x < y else (int(y), int(x)) for x, y in zip(xs, ys))

    def _from_tripple(self, a: rise.PyRecExpr, b: rise.PyRecExpr, distance: int) -> dict:
        "Alternative constructor. Cannot annotate return type because Python reasons"
        x_s, adjacency_s = pyrec_to_tensor(a, self.var_names, self.ignore_unknown)
        x_t, adjacency_t = pyrec_to_tensor(a, self.var_names, self.ignore_unknown)
        return {
            "x_s": x_s,
            "adjacency_s": adjacency_s,
            "x_t": x_t,
            "adjacency_t": adjacency_t,
            "distance": torch.tensor([distance]),
        }

    @property
    def raw_file_names(self) -> list[Path]:
        return [Path("df.parquet")]

    @property
    def raw_dir(self) -> Path:
        raw_dir = self.cache_dir / Path("raw")
        raw_dir.mkdir(exist_ok=True)
        return raw_dir

    @property
    def raw_paths(self) -> list[Path]:
        return [self.raw_dir / f for f in self.raw_file_names]

    @property
    def processed_file_names(self) -> list[Path]:
        return [Path(f"data_{i}.pt") for i in range(self.shards)]

    @property
    def processed_dir(self) -> Path:
        processed_dir = self.cache_dir / Path("processed")
        processed_dir.mkdir(exist_ok=True)
        return processed_dir

    @property
    def processed_meta_path(self) -> Path:
        return self.processed_dir / Path("meta.json")

    @property
    def processed_paths(self) -> list[Path]:
        return [self.processed_dir / f for f in self.processed_file_names]

    @property
    def node_features(self):
        return rise.PyGraphData.num_node_types(self.var_names, self.ignore_unknown) + 1


@lru_cache(maxsize=100)
def _load_shard(chunk_path: Path):
    return torch.load(chunk_path, weights_only=False)


def pyrec_to_tensor(expr: rise.PyRecExpr, var_names: list[str], ignore_unknown: bool) -> tuple[Tensor, Tensor]:
    graph_data = rise.PyGraphData(expr, var_names, ignore_unknown)
    node_types = F.one_hot(
        torch.tensor(graph_data.nodes, dtype=torch.long),
        num_classes=rise.PyGraphData.num_node_types(var_names, ignore_unknown),
    )
    const_values = torch.tensor([[x] if x else [0] for x in graph_data.const_values])
    nodes = torch.hstack((node_types, const_values)).to_sparse()

    edges_indices = graph_data.edges
    n_edges = len(edges_indices[0])
    adjacency_matrix = torch.sparse_coo_tensor(
        torch.tensor(edges_indices),
        torch.full((n_edges,), True, dtype=torch.bool),
        dtype=torch.bool,
        size=torch.Size((150, 150)),
    )

    return nodes, adjacency_matrix


if __name__ == "__main__":
    # expr = rise.PyRecExpr("(app (app zip (var a)) 1)))")
    # x_s, edge_index_s = PairData._pyrec_to_feature_vec(expr, ["a"], True)
    # print(x_s)
    # print(edge_index_s)
    a = 2
