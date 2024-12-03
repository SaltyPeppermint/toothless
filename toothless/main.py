from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
from torch_geometric.data import Data
import json


import eggshell
# from eggshell.rise import PyAst

DATA = Path("data/formatted.json")

device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else None
)


if __name__ == "__main__":
    with open(DATA) as f:
        data = json.load(f)

    variable_names = [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "x0",
        "x1",
        "x2",
        "x3",
        "map",
        "mapSeq",
        "iterateStream",
        "split",
        "join",
        "transpose",
        "mfu22",
    ]

    start_expr = eggshell.rise.PyAst(data["start_expr"], variable_names)
    print(start_expr)
    print(start_expr.count_symbols(variable_names))

    for s in data["sample_data"][0:10]:
        sample_expr = eggshell.rise.PyAst(s["sample"], variable_names)
        print(sample_expr)
        print(sample_expr.feature_vec_simple(variable_names))
