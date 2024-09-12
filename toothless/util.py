from pathlib import Path

import eggshell
import gymnasium as gym

import symbols
import data
import halide
from rl_env import Sketching

if __name__ == "__main__":
    data = data.load_data(
        Path(
            "data/with_baseline/5k_dataset_2024-09-04_10:02:20-b3e59ffa-b9f7-4807-80e7-3f20ecc18c8f"
        )
    )
    # print(halide_symbols)
    # print(symbols.symbol_table(halide_symbols))

    # for term in data[4].eclass_data[4].generated:
    #     print(str(term))
    #     print(term.flat())

    symbol_table = symbols.symbol_table(halide.halide_symbols(10, 0))
    # seed_data = halide.seed2pt(data, symbol_table)
    # generated_data = halide.generated2pt(data, symbol_table)
    # print(f"Raw generated: {len(generated_data)}")
    # baseline_data = halide.baselines2pt(data, symbol_table)
    # print(f"With baseline: {len(baseline_data)}")

    lhs = eggshell.PyLang("0", [])
    rhs = eggshell.PyLang("1", [])
    env = Sketching(30, 10, 0.2, 50, symbol_table)

    observation, info = env.reset(lhs, rhs)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("TERMINATED OR TRUNCATED\n\n")
            observation, info = env.reset(lhs, rhs)

    env.close()

    # root = eggshell.PySketch.new_root()
    # node_1 = eggshell.PySketch("*", 2)
    # node_2 = eggshell.PySketch("5", 0)
    # node_3 = eggshell.PySketch("true", 0)

    # print(root)
    # print(f"Size: {root.size()}")
    # print(f"Depth: {root.depth()}")
    # print(f"Finished: {root.finished()}")
    # print(f"Typechecks: {eggshell.halide.typecheck_sketch(root)}")
    # print(f"Flattened: {root.flat()}")

    # root.append(node_1)
    # print(root)
    # print(f"Size: {root.size()}")
    # print(f"Depth: {root.depth()}")
    # print(f"Finished: {root.finished()}")
    # print(f"Typechecks: {eggshell.halide.typecheck_sketch(root)}")
    # print(f"Flattened: {root.flat()}")

    # root.append(node_2)
    # print(root)
    # print(f"Size: {root.size()}")
    # print(f"Depth: {root.depth()}")
    # print(f"Finished: {root.finished()}")
    # print(f"Typechecks: {eggshell.halide.typecheck_sketch(root)}")
    # print(f"Flattened: {root.flat()}")

    # root.append(node_3)
    # print(root)
    # print(f"Size: {root.size()}")
    # print(f"Depth: {root.depth()}")
    # print(f"Finished: {root.finished()}")
    # print(f"Typechecks: {eggshell.halide.typecheck_sketch(root)}")
    # print(f"Flattened: {root.flat()}")
