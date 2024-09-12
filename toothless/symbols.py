from dataclasses import dataclass
from abc import ABC
from typing import Optional


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
