from dataclasses import dataclass
from abc import ABC
from typing import Optional

import eggshell


@dataclass
class Symbol(ABC):
    name: str
    arity: int
    index: Optional[int]


@dataclass
class LanguageSymbol(Symbol):
    pass


@dataclass
class SketchSymbol(Symbol):
    pass


@dataclass
class PartialSymbol(Symbol):
    pass


def symbol_table(symbols: list[Symbol]) -> dict[str, Symbol]:
    table = {}
    for i, s in enumerate(symbols):
        s.index = i
        table[s.name] = s
    return table


def add_partial_symbols(symbols: list[Symbol]) -> list[Symbol]:
    active_symbol = PartialSymbol(*eggshell.active_symbol(), None)
    todo_symbol = PartialSymbol(*eggshell.open_symbol(), None)

    return symbols + [active_symbol] + [todo_symbol]


def halide_symbols(vars: int, consts: int) -> list[Symbol]:
    halide_symbols = [
        LanguageSymbol(n, a, None) for (n, a) in eggshell.halide.symbols(vars, consts)
    ]
    sketch_symbols = [SketchSymbol(n, a, None) for (n, a) in eggshell.sketch_symbols()]

    return halide_symbols + sketch_symbols
