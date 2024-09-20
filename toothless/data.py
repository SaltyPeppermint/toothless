import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Self

from torch.nn import functional as F
import torch
from torch_geometric.data import Data
import tqdm
from torch.utils.data import Dataset

import eggshell
from eggshell import PyLang

from symbols import Symbol


@dataclass
class BaselineData:
    from_id: int
    to_id: int
    stop_reason: str
    total_time: float
    total_nodes: int
    total_iters: int

    @staticmethod
    def from_json(json_dict) -> Self:
        return BaselineData(
            json_dict["from"],
            json_dict["to"],
            json_dict["stop_reason"],
            json_dict["total_time"],
            json_dict["total_nodes"],
            json_dict["total_iters"],
        )


@dataclass
class ExplanationData:
    from_id: int
    to_id: int
    explanation: str

    @staticmethod
    def from_json(json_dict) -> Self:
        return ExplanationData(
            json_dict["from"], json_dict["to"], json_dict["explanation"]
        )


@dataclass
class EClassData:
    id: int
    generated: list[PyLang]
    baselines: Optional[list[BaselineData]]
    explanations: Optional[list[ExplanationData]]

    @staticmethod
    def from_json(json_dict) -> Self:
        id = json_dict["id"]
        generated = [PyLang.from_str(t) for t in json_dict["generated"]]
        baselines = (
            [BaselineData.from_json(t) for t in json_dict["baselines"]]
            if json_dict["baselines"] is not None
            else None
        )
        explanations = (
            [ExplanationData.from_json(t) for t in json_dict["explanations"]]
            if json_dict["explanations"] is not None
            else None
        )
        return EClassData(id, generated, baselines, explanations)


@dataclass
class EGraphData:
    seed_id: int
    seed_expr: PyLang
    truth_value: bool
    eclass_data: list[EClassData]

    @staticmethod
    def from_json(json_dict) -> Self:
        seed_id = json_dict["seed_id"]
        seed_expr = PyLang.from_str(json_dict["seed_expr"])
        truth_value = json_dict["truth_value"]
        eclass_data = [EClassData.from_json(t) for t in json_dict["eclass_data"]]
        return EGraphData(seed_id, seed_expr, truth_value, eclass_data)


def load_data(folder: Path) -> list[EGraphData]:
    data = []
    for file_path in folder.glob("*.json"):
        if "metadata" in file_path.name:
            continue
        with open(file_path) as f:
            json_data = json.load(f)
            for datapoint in json_data:
                data.append(EGraphData.from_json(datapoint))
    return data


def normalize_int(x: int, int_range: tuple[int, int]) -> float:
    return (x - int_range[0]) / (int_range[1] - int_range[0])


def check_int(s: str) -> bool:
    if s[0] == "-" and len(s) > 1:
        return s[1:].isdigit()
    return s.isdigit()


def expr2pt(
    expr: eggshell.PyLang | eggshell.PySketch, symbol_table: dict[str, Symbol]
) -> Data:
    flat_expr = expr.flat()

    tensor_len = len(symbol_table) + 1
    nodes = []
    for n in flat_expr.nodes:
        if check_int(n.name):
            t = torch.tensor(tensor_len - 1)
            one_hot = F.one_hot(t, num_classes=tensor_len).to(torch.float32)
            int_encode = normalize_int(int(n.name), int_range=(-1024, 1024))
            position = normalize_int(n.root_distance, int_range=(0, 32))

            nodes.append(
                torch.cat(
                    (one_hot, torch.tensor([int_encode]), torch.tensor([position]))
                )
            )
        else:
            t = torch.tensor(symbol_table[n.name].index)
            one_hot = F.one_hot(t, num_classes=tensor_len).to(torch.float32)

            position = normalize_int(n.root_distance, int_range=(0, 32))
            nodes.append(torch.cat((one_hot, torch.zeros(1), torch.tensor([position]))))

    x = torch.stack(nodes)
    edge_index = torch.transpose(torch.tensor(flat_expr.edges), 0, 1)

    return Data(x=x, edge_index=edge_index)


def pair2pt(
    lhs: eggshell.PyLang, rhs: eggshell.PyLang, symbol_table: dict[str, Symbol]
) -> tuple[Data, Data]:
    return (expr2pt(lhs, symbol_table)), (expr2pt(rhs, symbol_table))


def seed_pairs(
    data_dir: Path, symbol_table: dict[str, Symbol]
) -> list[tuple[Data, Data]]:
    egraph_data = load_data(data_dir)
    term_pairs = []
    for egraph in egraph_data:
        end_as_lang = eggshell.PyLang.from_str(str(egraph.truth_value).lower())
        pair = pair2pt(egraph.seed_expr, end_as_lang, symbol_table)
        term_pairs.append(pair)
    return term_pairs


def is_useful_baseline(baseline: BaselineData) -> bool:
    if baseline.total_iters < 2:
        return False
    if baseline.total_time < 1.0:
        return False
    return True


def baseline_pairs(
    data_dir: Path, symbol_table: dict[str, Symbol]
) -> list[tuple[Data, Data]]:
    egraph_data = load_data(data_dir)
    term_pairs = []
    for egraph in tqdm.tqdm(egraph_data):
        for eclass in egraph.eclass_data:
            for b in eclass.baselines:
                if not is_useful_baseline(b):
                    continue
                pair = pair2pt(
                    eclass.generated[b.from_id],
                    eclass.generated[b.to_id],
                    symbol_table,
                )
                term_pairs.append(pair)
    return term_pairs


def is_useful_generated(expr: eggshell.PySketch) -> bool:
    return expr.size() > 5


def generated_pairs(
    data_dir: Path, symbol_table: dict[str, Symbol]
) -> list[tuple[Data, Data]]:
    egraph_data = load_data(data_dir)
    term_pairs = []
    for egraph in tqdm.tqdm(egraph_data):
        for eclass in egraph.eclass_data:
            filtered_data = filter(is_useful_generated, eclass.generated)
            for lhs, rhs in [
                (lhs, rhs)
                for lhs in filtered_data
                for rhs in filtered_data
                if lhs != rhs
            ]:
                pair = pair2pt(lhs, rhs, symbol_table)
                term_pairs.append(pair)
    return term_pairs
