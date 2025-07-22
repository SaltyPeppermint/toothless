import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feed_forward: int, dropout: float = 0.1, activation=F.gelu):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
