from typing import Callable
from gymnasium import Env
from gymnasium.spaces import Discrete

import eggshell

import data
from symbols import Symbol


class Sketching(Env):
    sketch: eggshell.PySketch
    lhs: eggshell.PyLang
    rhs: eggshell.PyLang
    max_size: int
    min_size: int
    max_fraction_sketch: float
    step_limit: int
    symbol_table: dict[str, Symbol]
    typechecker: Callable[[eggshell.PySketch], bool]
    symbol_table: list[Symbol]
    action_space: Discrete
    render_mode: str
    steps: int
    last_reward: int
    accumulated_reward: int

    def __init__(
        self,
        max_size: int,
        min_size: int,
        max_fraction_sketch: float,
        step_limit: int,
        symbol_table: dict[str, Symbol],
        typechecker: Callable[[eggshell.PySketch], bool],
        render_mode="human",
        **kwargs,
    ):
        self.max_size = max_size
        self.min_size = min_size
        self.max_fraction_sketch = max_fraction_sketch
        self.step_limit = step_limit
        self.symbol_table = symbol_table
        self.typechecker = typechecker
        self.symbol_list = list(self.symbol_table.values())
        self.action_space = Discrete(len(self.symbol_list))
        self.render_mode = render_mode

        super(Env, self).__init__(**kwargs)

    def step(self, action: int):
        # Translate index into symbol from the symbol_table
        chosen_symbol = self.symbol_list[action]
        # print(chosen_symbol)
        # Append Symbol at the current [active] node, aka taking the action
        terminated = self.sketch.append(
            eggshell.PySketch(chosen_symbol.name, chosen_symbol.arity)
        )
        # Check the time limit
        truncated = self.steps > self.step_limit

        typechecks = self.typechecker(self.sketch)
        depth = self.sketch.depth()
        size = self.sketch.size()
        sketch_symbols = self.sketch.sketch_symbols()
        fraction_sketch = sketch_symbols / size

        self.last_reward = 0

        # Add penalty if an agent goes over the size limit
        if size > self.max_size:
            self.last_reward -= size - self.max_size

        # Add penalty if an agent ends up with a sketch that is too big
        if terminated and size < self.max_size:
            self.last_reward -= 20

        # Add penalty if an agent uses too many sketch symbols
        if fraction_sketch >= self.max_fraction_sketch:
            self.last_reward -= size - self.max_fraction_sketch

        # Add penalty if an agent reaches a sketch that does not typecheck
        if typechecks:
            pass  # Here goes the usefulness calculation
        else:
            self.last_reward -= 50

        # # Add penalty if an agnet does not reach in time

        if truncated:
            self.last_reward -= 1000

        self.accumulated_reward += self.last_reward
        self.steps += 1

        info = {
            "reward": self.last_reward,
            "size": size,
            "depth": depth,
            "typechecks": typechecks,
            "native": self.sketch,
            "flattened": self.sketch.flat(),
            "accumulated_reward": self.accumulated_reward,
            "step": self.step,
        }

        if self.render_mode == "human":
            self.render()

        return (
            data.expr2pt(self.sketch, self.symbol_table),
            self.last_reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, lhs: eggshell.PyLang, rhs: eggshell.PyLang):
        self.sketch = eggshell.PySketch.new_root()
        self.lhs = lhs
        self.rhs = rhs
        self.steps = 0
        self.last_reward = 0
        self.accumulated_reward = 0

        info = {
            "size": self.sketch.size(),
            "depth": self.sketch.depth(),
            "typechecks": self.typechecker(self.sketch),
            "native": self.sketch,
            "flattened": self.sketch.flat(),
        }
        return data.expr2pt(self.sketch, self.symbol_table), info

    def render(self):
        print(
            f"Step {self.steps}:\n  Last Reward: {self.last_reward}\n  Accumulated Reward: {self.accumulated_reward}\n Sketch: {self.sketch}\n"
        )
