import torch
from torch import nn
from torch_geometric.nn import pool
import torch_geometric.nn as gnn
from gymnasium import spaces

from rl_env import Observation


class AstEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(AstEmbed, self).__init__()

        self.in_layer = gnn.GATv2Conv(input_dim, hidden_dim)

        self.hidden_1 = gnn.GATv2Conv(hidden_dim, hidden_dim)
        self.hidden_2 = gnn.GATv2Conv(hidden_dim, hidden_dim)
        self.hidden_3 = gnn.GATv2Conv(hidden_dim, hidden_dim)

        self.out_layer = gnn.GATv2Conv(hidden_dim, embed_dim)

    def forward(self, x, edge_index, root_index):
        x = self.in_layer(x, edge_index)
        x = x.relu()
        x = self.hidden_1(x, edge_index)
        x = x.relu()
        x = self.hidden_2(x, edge_index)
        x = x.relu()
        x = self.hidden_3(x, edge_index)
        x = x.relu()
        x = self.out_layer(x, edge_index)

        return x[root_index]


class SketchEmbed(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param internal_dim: (int) Size of the internal embedding layer
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        hidden_dim: int = 256,
        embed_dim: int = 256,
    ):
        super(SketchEmbed, self).__init__()
        lhs_in_shape = obs_space["lhs"].node_space.shape[0]
        rhs_in_shape = obs_space["lhs"].node_space.shape[0]
        sketch_in_shape = obs_space["lhs"].node_space.shape[0]

        self.lhs_encoder = AstEmbed(lhs_in_shape, hidden_dim, embed_dim)
        self.rhs_encoder = AstEmbed(rhs_in_shape, hidden_dim, embed_dim)
        self.sketch_encoder = AstEmbed(sketch_in_shape, hidden_dim, embed_dim)

        self.backbone = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, observation: Observation):
        lhs_emb = self.lhs_encoder(
            observation.x_lhs, observation.edge_index_lhs, observation.lhs_root
        )
        rhs_emb = self.rhs_encoder(
            observation.x_rhs, observation.edge_index_rhs, observation.rhs_root
        )
        sketch_emb = self.sketch_encoder(
            observation.x_sketch, observation.edge_index_sketch, observation.sketch_root
        )

        embs = torch.cat((sketch_emb, lhs_emb, rhs_emb), -1)

        return self.backbone(embs)
