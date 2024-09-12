import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Self

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
