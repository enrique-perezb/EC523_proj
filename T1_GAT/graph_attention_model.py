import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=1):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True)
        self.relu = nn.ReLU()
        self.pool = global_mean_pool
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 1)
        )


    def forward(self, data):

        # Structure data for forward pass
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GAT layer 1
        x = self.gat1(x, edge_index)
        x = self.relu(x)

        # GAT layer 2
        x = self.gat2(x, edge_index)
        x = self.relu(x)

        # Mean pooling
        x = self.pool(x, batch)

        # MLP
        out = self.regressor(x)

        return out.view(-1)