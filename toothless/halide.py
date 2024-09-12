from torch.nn import functional as F
import torch
from torch_geometric.data import Data
import tqdm

import eggshell

from data import BaselineData, EGraphData
from symbols import LanguageSymbol, PartialSymbol, SketchSymbol, Symbol


def halide_symbols(vars: int, consts: int) -> list[Symbol]:
    halide_symbols = [
        LanguageSymbol(n, a, None) for (n, a) in eggshell.halide.symbols(vars, consts)
    ]
    sketch_symbols = [SketchSymbol(n, a, None) for (n, a) in eggshell.sketch_symbols()]
    active_symbol = PartialSymbol(*eggshell.active_symbol(), None)
    todo_symbol = PartialSymbol(*eggshell.open_symbol(), None)

    return sketch_symbols + [active_symbol] + [todo_symbol] + halide_symbols


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

    edges = torch.transpose(torch.tensor(flat_expr.edges), 0, 1)
    data = Data(x=nodes, edge_index=edges)

    return data


def seed2pt(
    data: list[EGraphData], symbol_table: dict[str, Symbol]
) -> list[(Data, Data)]:
    train_pairs = []
    for egraph in data:
        seed = expr2pt(egraph.seed_expr, symbol_table)
        end_as_lang = eggshell.PyLang.from_str(str(egraph.truth_value).lower())
        truth = expr2pt(end_as_lang, symbol_table)
        train_pairs.append((seed, truth))

    return train_pairs


def is_useful_baseline(baseline: BaselineData) -> bool:
    if baseline.total_iters < 2:
        return False
    if baseline.total_time < 1.0:
        return False
    return True


def baselines2pt(
    data: list[EGraphData], symbol_table: dict[str, Symbol]
) -> list[(Data, Data)]:
    train_pairs = []
    for egraph in tqdm.tqdm(data):
        for eclass in egraph.eclass_data:
            for b in eclass.baselines:
                if not is_useful_baseline(b):
                    continue
                lhs_expr = expr2pt(eclass.generated[b.from_id], symbol_table)
                rhs_expr = expr2pt(eclass.generated[b.to_id], symbol_table)
                train_pairs.append((lhs_expr, rhs_expr))

    return train_pairs


def is_useful(expr: eggshell.PySketch) -> bool:
    return expr.size() > 5


def generated2pt(
    data: list[EGraphData], symbol_table: dict[str, Symbol]
) -> list[(Data, Data)]:
    train_pairs = []
    for egraph in tqdm.tqdm(data):
        for eclass in egraph.eclass_data:
            filtered_data = filter(is_useful, eclass.generated)
            for pair in [
                (lhs, rhs)
                for lhs in filtered_data
                for rhs in filtered_data
                if lhs != rhs
            ]:
                lhs_expr = expr2pt(pair[0], symbol_table)
                rhs_expr = expr2pt(pair[1], symbol_table)
                train_pairs.append((lhs_expr, rhs_expr))

    return train_pairs
