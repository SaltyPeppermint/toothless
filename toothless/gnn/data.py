from functools import lru_cache
from pathlib import Path
import json
import math

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

import polars as pl

from eggshell import rise  # type: ignore
from toothless.utils import loading


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    @staticmethod
    def from_tripple(a: rise.PyRecExpr, b: rise.PyRecExpr, distance: int, var_names: list[str], ignore_unknown: bool):
        "Alternative constructor. Cannot annotate return type because Python reasons"
        x_s, edge_index_s = PairData._pyrec_to_feature_vec(a, var_names, ignore_unknown)
        x_t, edge_index_t = PairData._pyrec_to_feature_vec(b, var_names, ignore_unknown)
        return PairData(x_s=x_s, edge_index_s=edge_index_s, x_t=x_t, edge_index_t=edge_index_t, y=distance)

    @staticmethod
    def _pyrec_to_feature_vec(
        expr: rise.PyRecExpr, var_names: list[str], ignore_unknown: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_data = rise.PyGraphData(expr, var_names, ignore_unknown)
        node_types = F.one_hot(
            torch.tensor(graph_data.nodes, dtype=torch.long),
            num_classes=rise.PyGraphData.num_node_types(var_names, ignore_unknown),
        )
        const_values = torch.tensor([[x] if x else [0] for x in graph_data.const_values])
        nodes = torch.hstack((node_types, const_values))

        return nodes, torch.tensor(graph_data.edges, dtype=torch.long)


torch.serialization.add_safe_globals([PairData])


class PairDataset(Dataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        var_names: list[str],
        ignore_unknown: bool,
        random_state: int = 0,
        chunks: int = 50,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl

        self.var_names = var_names
        self.ignore_unknown = ignore_unknown
        self.random_state = random_state
        self.chunks = chunks

        self.ds_size_memo = None
        self.len_memo = None
        self.data_cache = {}
        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])

        super().__init__(str(self.cache_dir), transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["df.parquet"]

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(self.chunks)] + ["meta.json"]
        # return ["data.pt"]

    def download(self):
        df = loading.load_df(self.json_root, self.var_names)
        df.write_parquet(self.raw_paths[0])

    def process(self):
        # Read data into huge `Data` list.

        expl_chains = pl.read_parquet(self.raw_paths[0]).get_column("explanation_chain")
        torch.manual_seed(self.random_state)

        expl_chains = [rise.PyRecExpr.batch_new(i) for i in expl_chains]
        print("Parallel processing done")

        picked_pairs = [self._pick_indices(len(chain)) for chain in expl_chains]

        self.len_memo = self.ds_size_memo = sum([len(chain_pairs) for chain_pairs in picked_pairs])

        print(f"Total pairs: {self.len_memo}")
        print(f"Chunk size: {self.max_chunk_size()}")

        save_buffer = []
        spill_buffer = []
        idx = 0

        for chain, picked_pairs in zip(expl_chains, picked_pairs):
            pairs = [
                PairData.from_tripple(chain[a], chain[b], a - b, self.var_names, self.ignore_unknown)
                for a, b in picked_pairs
            ]

            if self.pre_filter is not None:
                pairs = [data for data in pairs if self.pre_filter(data)]

            if self.pre_transform is not None:
                pairs = [self.pre_transform(data) for data in pairs]

            remaining_space = self.max_chunk_size() - len(save_buffer)
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
            json.dump({"len": len(self), "max_chunk_size": self.max_chunk_size()}, f)

    def max_chunk_size(self):
        # Get number of full chunks -> round up
        return -(-self.ds_size() // self.chunks)

    def len(self):
        if self.len_memo is None:
            with open(Path(self.processed_dir) / Path("meta.json")) as f:
                j = json.load(f)
                self.len_memo = j["len"]
                return j["len"]
        else:
            return self.len_memo

    def ds_size(self) -> int:
        if self.ds_size_memo is None:
            with open(Path(self.processed_dir) / Path("meta.json")) as f:
                j = json.load(f)
                self.len_memo = j["len"]
                return j["len"]
        else:
            return self.ds_size_memo

    def get(self, idx) -> PairData:
        max_chunk_size = self.max_chunk_size()
        chunk_path = Path(self.processed_paths[idx // max_chunk_size])
        chunk = _load_chunk(chunk_path)
        return chunk[idx % max_chunk_size]

    def to_dataloader(self, batch_size: int, shuffle: bool = False) -> tuple[DataLoader, DataLoader]:
        split_idx = math.floor(0.9 * len(self))

        train_loader = DataLoader(self[:split_idx], follow_batch=["x_s", "x_t"], batch_size=batch_size, shuffle=shuffle)  # type: ignore
        test_loader = DataLoader(self[split_idx:], follow_batch=["x_s", "x_t"], batch_size=batch_size, shuffle=shuffle)  # type: ignore
        return train_loader, test_loader

    def _pick_indices(self, max_index: int) -> set[tuple[int, int]]:
        xs = torch.randint(0, max_index, (self.pairs_per_expl,))
        ys = torch.randint(0, max_index, (self.pairs_per_expl,))
        return set((int(x), int(y)) if x < y else (int(y), int(x)) for x, y in zip(xs, ys))


@lru_cache(maxsize=100)
def _load_chunk(chunk_path: Path):
    return torch.load(chunk_path, weights_only=False)


if __name__ == "__main__":
    x_s = torch.randn(5, 16)  # 5 nodes.
    edge_index_s = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
        ]
    )

    x_t = torch.randn(4, 16)  # 4 nodes.
    edge_index_t = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 3],
        ]
    )
    y = torch.tensor([5], dtype=torch.long)

    pair_data = PairData(x_s=x_s, edge_index_s=edge_index_s, x_t=x_t, edge_index_t=edge_index_t, y=y)
    data_list = [pair_data, pair_data, pair_data, pair_data]
    loader = DataLoader(data_list, follow_batch=["x_s", "x_t"], batch_size=3)
    batch = next(iter(loader))

    print(batch.x_s)
    print(batch.edge_index_s)
    print(batch.x_t)
    print(batch.edge_index_t)
    print(batch.y)
    print(batch.x_s_batch)
    print(batch.x_t_batch)
    print(batch)

    expr = rise.PyRecExpr("(app (app zip (var a)) 1)))")
    x_s, edge_index_s = PairData._pyrec_to_feature_vec(expr, ["a"], True)
    print(x_s)
    print(edge_index_s)
