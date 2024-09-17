import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, Softmax

from torch_geometric.nn import GATv2Conv


class AstEmbedding(torch.nn.Module):
    def __init__(self, input_size, internal_size, output_size):
        super().__init__()
        self.conv1 = GATv2Conv(input_size, internal_size)
        self.conv2 = GATv2Conv(internal_size, internal_size)
        self.conv3 = GATv2Conv(internal_size, internal_size)
        self.conv4 = GATv2Conv(internal_size, output_size)

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

        return global_mean_pool(x)


class SketchNet(torch.nn.Module):
    def __init__(self, input_size, internal_size, output_classes):
        super().__init__()
        self.lhs_encoder = AstEmbedding(input_size, internal_size, internal_size)
        self.rhs_encoder = AstEmbedding(input_size, internal_size, internal_size)
        self.sketch_encoder = AstEmbedding(input_size, internal_size, internal_size)
        self.lin1 = Linear(internal_size * 3, internal_size * 3)
        self.lin2 = Linear(internal_size * 3, internal_size * 3)
        self.lin3 = Linear(internal_size * 3, output_classes)
        self.softmax = Softmax()

    def forward(self, lhs, rhs, sketch):
        lhs_emb = self.lhs_encoder(lhs)
        rhs_emb = self.rhs_encoder(rhs)
        sketch_emb = self.sketch_encoder(sketch)

        embs = torch.cat(sketch_emb, lhs_emb, rhs_emb)

        x = self.lin1(embs)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return F.log_softmax(x, dim=1)
