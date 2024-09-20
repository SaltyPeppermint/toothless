from dataclasses import dataclass
from typing import Any, Callable

from gymnasium import Env
from gymnasium.spaces import (
    Discrete,
    Graph,
    Box,
    Dict,
)
from gymnasium.envs.registration import register
import numpy as np

from torch import DeviceObjType, nn
from torch_geometric.data import Data, HeteroData


import eggshell

import data
from symbols import Symbol


register(
    id="custom/SketchEnv-v0",
    entry_point="rl_env:SketchEnv",
    max_episode_steps=100,
    nondeterministic=True,
)


class Observation(nn.Module):
    def __init__(self, lhs, rhs, sketch):
        super(Observation, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.sketch = sketch

    # Manual impl of to since nn doesnt provide it?
    def to(self, device: DeviceObjType):
        self.lhs.to(device)
        self.rhs.to(device)
        self.sketch.to(device)
        return self

    # Manual impl of detach since nn doesnt provide it?
    def detach(self):
        self.lhs.detach()
        self.rhs.detach()
        self.sketch.detach()
        return self


class SketchEnv(Env):
    sketch: eggshell.PySketch
    flat_term_pairs: list[tuple[Data, Data]]
    max_size: int
    min_size: int
    max_sketch_ratio: float
    symbol_table: dict[str, Symbol]
    typechecker: Callable[[eggshell.PySketch], bool]
    actions: list[Symbol]
    action_space: Discrete
    render_mode: str
    steps: int
    last_reward: int
    accumulated_reward: int

    def __init__(
        self,
        flat_term_pairs: list[tuple[Data, Data]],
        max_size: int,
        min_size: int,
        max_sketch_ratio: float,
        symbol_table: dict[str, Symbol],
        actions: list[Symbol],
        node_tensor_len: int,
        typechecker: Callable[[eggshell.PySketch], bool],
        render_mode: str,
        **kwargs,
    ):
        self.flat_term_pairs = flat_term_pairs
        self.max_size = max_size
        self.min_size = min_size
        self.max_sketch_ratio = max_sketch_ratio
        self.symbol_table = symbol_table
        self.symbol_list = actions
        self.typechecker = typechecker
        self.render_mode = render_mode if render_mode is not None else "human"

        self.action_space = Discrete(len(actions))
        # # We are essentially lying here because nested spaces and graphs are not supported
        # # This gives the relevant information of the node shape
        # self.observation_space = Box(
        #     low=0, high=1, shape=(node_tensor_len,), seed=42, dtype=np.float32
        # )
        self.observation_space = Dict(
            {
                "sketch": Graph(
                    node_space=Box(low=0, high=1, shape=(node_tensor_len,)),
                    edge_space=None,
                ),
                "lhs": Graph(
                    node_space=Box(low=0, high=1, shape=(node_tensor_len,)),
                    edge_space=None,
                ),
                "rhs": Graph(
                    node_space=Box(low=0, high=1, shape=(node_tensor_len,)),
                    edge_space=None,
                ),
            },
            seed=42,
        )

        self.action_space = Discrete(len(self.symbol_list))

        # Not forwarding the envs because of the unexpected frameskip argument
        super(Env, self).__init__(**kwargs)

    def step(self, action: int) -> tuple[Observation, int, bool, bool, dict[str, Any]]:
        # Translate index into symbol from the symbol_table
        chosen_symbol = self.symbol_list[action]
        # Append Symbol at the current [active] node, aka taking the action
        terminated = self.sketch.append(
            eggshell.PySketch(chosen_symbol.name, chosen_symbol.arity)
        )

        typechecks = self.typechecker(self.sketch)
        size = self.sketch.size()
        sketch_symbols = self.sketch.sketch_symbols()
        fraction_sketch = sketch_symbols / size

        self.last_reward = 0

        # Add penalty if an agent goes over the size limit
        if size > self.max_size:
            self.last_reward -= 5

        # Add penalty if an agent uses too many sketch symbols
        if fraction_sketch >= self.max_sketch_ratio:
            self.last_reward -= 5

        # Add a reward if an agent finsihes a sketch
        if terminated:
            # But only if it is big enough
            if size < self.min_size:
                self.last_reward -= 5
            else:
                self.last_reward += 20

        # Add penalty if an agent reaches a sketch that does not typecheck
        if typechecks:
            pass  # Here goes the usefulness calculation
        else:
            self.last_reward -= 50

        self.accumulated_reward += self.last_reward
        self.steps += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, self.last_reward, terminated, False, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        self.sketch = eggshell.PySketch.new_root()

        index = np.random.randint(0, len(self.flat_term_pairs))
        self.lhs, self.rhs = self.flat_term_pairs[index]

        self.steps = 0
        self.last_reward = 0
        self.accumulated_reward = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self):
        print(
            f"Step {self.steps}:\n  Last Reward: {self.last_reward}\n  Accumulated Reward: {self.accumulated_reward}\n  Sketch: {self.sketch}\n"
        )

    def _get_obs(self) -> HeteroData:
        sketch_data = data.expr2pt(self.sketch, self.symbol_table)

        return Observation(sketch_data, self.lhs, self.rhs)

    def _get_info(self):
        return {
            "reward": self.last_reward,
            "size": self.sketch.size(),
            "depth": self.sketch.depth(),
            "typechecks": self.typechecker(self.sketch),
            "native": self.sketch,
            "flattened": self.sketch.flat(),
            "accumulated_reward": self.accumulated_reward,
            "step": self.step,
        }
