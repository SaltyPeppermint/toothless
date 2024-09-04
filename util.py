import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Self, Set
import pprint

import eggshell
from eggshell import PyLang


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
    generated: List[PyLang]
    baselines: List[BaselineData]
    explanations: List[ExplanationData]

    @staticmethod
    def from_json(json_dict) -> Self:
        id = json_dict["id"]
        generated = [PyLang.from_str(t) for t in json_dict["generated"]]
        baselines = (
            [BaselineData(t) for t in json_dict["baselines"]]
            if json_dict["baselines"] is not None
            else []
        )
        explanations = (
            [ExplanationData(t) for t in json_dict["explanations"]]
            if json_dict["explanations"] is not None
            else []
        )
        return EClassData(id, generated, baselines, explanations)

    # def __repr__(self) -> str:
    #     generated_str = [str(t) for t in self.generated]
    #     return f"EClassData(id={self.id}, generated={generated_str}, baselines={self.baselines}, explanations={self.explanations})"


@dataclass
class EGraphData:
    seed_id: int
    seed_expr: PyLang
    truth_value: bool
    eclass_data: List[EClassData]

    @staticmethod
    def from_json(json_dict) -> Self:
        seed_id = json_dict["seed_id"]
        seed_expr = PyLang.from_str(json_dict["seed_expr"])
        truth_value = json_dict["truth_value"]
        eclass_data = [EClassData.from_json(t) for t in json_dict["eclass_data"]]
        return EGraphData(seed_id, seed_expr, truth_value, eclass_data)

    # def __repr__(self) -> str:
    #     return f"EGraphData(seed_id={self.seed_id}, seed_expr='{str(self.seed_expr)}', truth_value={self.truth_value}, eclass_data={self.eclass_data})"


def load_data(folder: Path):
    data = []
    for file_path in folder.glob("*.json"):
        if "metadata" in file_path.name:
            continue
        with open(file_path) as f:
            json_data = json.load(f)
            for datapoint in json_data:
                data.append(EGraphData.from_json(datapoint))
    return data


def symbols_in_data(data: List[EGraphData]) -> Dict[str, int]:
    symbols = {}
    for graph_data in data:
        symbols.update(graph_data.seed_expr.symbols())

    def check_num(symbol: str) -> bool:
        if symbol.isnumeric():
            return True
        if len(symbol) >= 2:
            return symbol[0] == "-" and symbol[1:].isnumeric()
        return False

    return {k: v for k, v in symbols.items() if not check_num(k)}


def add_sketch_symbols(symbols: Dict[str, int]) -> Dict[str, int]:
    if "contains" in symbols.keys() or "or" in symbols.keys() or "?" in symbols.keys():
        raise ValueError("Symbol dictionary already contains the sketch symbols")
    symbols["contains"] = 1
    symbols["or"] = 2
    symbols["?"] = 0
    return symbols


if __name__ == "__main__":
    data = load_data(
        Path("data/5k_dataset_2024-09-04_12:15:28-50c2da92-5fdb-4bc2-9200-28494f940df3")
    )
    symbols = add_sketch_symbols(symbols_in_data(data))
    print(symbols)
