import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


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


class MyNet(torch.nn.Module):
    def __init__(self, internal_size):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, internal_size)
        self.conv2 = GCNConv(internal_size, internal_size)
        self.conv3 = GCNConv(internal_size, internal_size)
        self.conv4 = GCNConv(internal_size, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


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


if __name__ == "__main__":
    check_mps()
    device = torch.device("mps")

    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = dataset[0].to(device)

    model = MyNet(32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-4)

    train(data, model, optimizer)
    eval(data, model)
