from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset

from sklearn.model_selection import train_test_split
import polars as pl
import numpy as np

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
        x_s, edge_index_s = _pyrec_to_feature_vec(a, var_names, ignore_unknown)
        x_t, edge_index_t = _pyrec_to_feature_vec(b, var_names, ignore_unknown)
        return PairData(x_s=x_s, edge_index_s=edge_index_s, x_t=x_t, edge_index_t=edge_index_t, y=distance)


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


def _pick_indices(max_index: int, n: int) -> set[tuple[int, int]]:
    rng = np.random.default_rng()
    indices = rng.integers(low=0, high=max_index, size=n * 2)
    first_half = indices[: len(indices) // 2]
    second_half = indices[len(indices) // 2 :]

    return set((int(x), int(y)) if x < y else (int(y), int(x)) for x, y in zip(first_half, second_half))


def load_df(data_path) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_parquet(data_path).select(["goal_expr", "middle_expr", "start_expr"])

    # print(df.head())

    test_size = 0.2
    random_state = 42
    train, eval = train_test_split(df, test_size=test_size, random_state=random_state)

    return train, eval


class PairDataset(InMemoryDataset):
    def __init__(
        self,
        json_root: Path,
        pairs_per_expl: int,
        var_names: list[str],
        ignore_unknown: bool,
        random_state: int = 0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.json_root = Path(json_root)
        self.pairs_per_expl = pairs_per_expl

        self.var_names = var_names
        self.ignore_unknown = ignore_unknown
        self.random_state = random_state

        self.cache_dir = Path("cache") / Path(*self.json_root.parts[2:])

        super().__init__(str(self.cache_dir), transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["df.parquet"]

    @property
    def processed_file_names(self):
        return ["processed_file_names.pt"]

    def download(self):
        df = loading.load_df(self.json_root, self.var_names)
        df.write_parquet(self.raw_paths[0])

    def process(self):
        # Read data into huge `Data` list.
        expl_chains = pl.read_parquet(self.raw_paths[0]).get_column("explanation_chain")
        np.random.seed(self.random_state)
        data_list = []
        for chain in expl_chains:
            index_pairs = _pick_indices(len(chain), self.pairs_per_expl)
            selected_pair_data = [
                PairData.from_tripple(
                    rise.PyRecExpr(chain[a]), rise.PyRecExpr(chain[b]), a - b, self.var_names, self.ignore_unknown
                )
                for a, b in index_pairs
            ]
            data_list.extend(selected_pair_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


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
    x_s, edge_index_s = _pyrec_to_feature_vec(expr, ["a"], True)
    print(x_s)
    print(edge_index_s)
