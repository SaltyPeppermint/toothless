from pathlib import Path
import eggshell
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


import symbols
from model import SketchNet
from rl_env import Sketching
from data import GeneratedDataset


def check_mps():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    else:
        print("MPS is available.")


def train(data, model, optimizer):
    writer = SummaryWriter()
    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

    writer.flush()


def eval(data, model):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f"Accuracy: {acc:.4f}")


# if __name__ == "__main__":
#     check_mps()
#     device = torch.device("mps")

#     dataset = Planetoid(root="/tmp/Cora", name="Cora")
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     data = dataset[0].to(device)

#     model = MyNet(32).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

#     train(data, model, optimizer)
#     eval(data, model)

DATA_PATH = Path(
    "data/with_baseline/5k_dataset_2024-09-04_10:02:20-b3e59ffa-b9f7-4807-80e7-3f20ecc18c8f"
)


if __name__ == "__main__":
    # print(halide_symbols)
    # print(symbols.symbol_table(halide_symbols))

    # for term in data[4].eclass_data[4].generated:
    #     print(str(term))
    #     print(term.flat())

    halide_symbols = symbols.halide_symbols(10, 0)
    symbol_table = symbols.symbol_table(symbols.add_partial_symbols(halide_symbols))
    # seed_data = SeedDataset(DATA_PATH, symbol_table)
    # print(f"Seed dataset: {len(seed_data)}")
    generated_data = GeneratedDataset(DATA_PATH, symbol_table)
    print(f"Raw dataset: {len(generated_data)}")
    # baseline_data = BaselineDataset(DATA_PATH, symbol_table)
    # print(f"Baseline dataset: {len(baseline_data)}")

    # loader = DataLoader(generated_data, batch_size=1, shuffle=True)

    # lhs = eggshell.PyLang("0", [])
    # rhs = eggshell.PyLang("1", [])
    env = Sketching(
        30,
        10,
        0.2,
        50,
        symbol_table,
        eggshell.halide.typecheck_sketch,
        render_mode=None,
    )

    n_input = generated_data[0][0].num_node_features
    model = SketchNet(n_input, 32, len(halide_symbols))

    for lhs, rhs in generated_data:
        observation, info = env.reset(lhs, rhs)
        print(lhs)
        print(rhs)
        for _ in range(1000):
            action = (
                env.action_space.sample()
            )  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("TERMINATED OR TRUNCATED\n")
                break

    env.close()
