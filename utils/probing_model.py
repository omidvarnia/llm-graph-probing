import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, Embedding, Dropout

def global_mean_pool_torch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch global mean pool without torch_scatter.
    x: [N, C], batch: [N] with graph ids in [0, G-1]
    Returns: [G, C]
    """
    x = torch.nan_to_num(x)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.empty((0, x.size(1)), device=x.device, dtype=x.dtype)
    sums = torch.zeros((num_graphs, x.size(1)), device=x.device, dtype=x.dtype)
    counts = torch.zeros((num_graphs, 1), device=x.device, dtype=x.dtype)
    sums.index_add_(0, batch, x)
    counts.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return sums / counts.clamp(min=1)

def global_max_pool_torch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch global max pool without torch_scatter.
    Uses per-graph loops which are acceptable for small graphs.
    """
    x = torch.nan_to_num(x)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.empty((0, x.size(1)), device=x.device, dtype=x.dtype)
    result = torch.full((num_graphs, x.size(1)), -float('inf'), device=x.device, dtype=x.dtype)
    for g in range(num_graphs):
        mask = batch == g
        if mask.any():
            vals = x[mask]
            result[g] = vals.max(dim=0).values
        else:
            result[g] = 0.0
    return result

class SimpleGCNConv(nn.Module):
    """Minimal GCN-like convolution without torch_scatter/torch_sparse.
    Aggregates neighbor features via index_add and applies a Linear.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None):
        # x: [N, C]; edge_index: [2, E]; edge_weight: [E] or None
        if edge_index.numel() == 0:
            return self.linear(x)
        N = x.size(0)
        src = edge_index[1]
        dst = edge_index[0]
        messages = torch.nan_to_num(x[src])
        if edge_weight is not None:
            ew = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
            ew = torch.abs(ew)
            messages = messages * ew.view(-1, 1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)
        # Mean aggregation to control magnitude
        deg = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
        ones = torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, ones)
        out = out / deg.clamp(min=1)
        out = torch.nan_to_num(out)
        return self.linear(out)


class MLPProbe(nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_layers, num_output=1):
        super(MLPProbe, self).__init__()
        self.mlp = ModuleList()
        for _ in range(num_layers):
            self.mlp.append(Linear(num_nodes, hidden_channels))
            self.mlp.append(ReLU())
            num_nodes = hidden_channels
        self.fc = Linear(num_nodes, num_output)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        output = self.fc(x)
        return output.squeeze(-1)


class GCNProbe(nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_layers, dropout=0.0, num_output=1, nonlinear_activation=True):
        super(GCNProbe, self).__init__()
        self.embedding = Embedding(num_nodes, hidden_channels)
        self.convs = nn.ModuleList([SimpleGCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.activation = nn.ReLU() if nonlinear_activation else nn.Identity()
        self.fc1 = Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_output)
        self.dropout = Dropout(dropout)

    def forward_graph_embedding(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout.p, training=self.training)
            x = torch.nan_to_num(x)
        mean_x = global_mean_pool_torch(x, batch)
        max_x = global_max_pool_torch(x, batch)
        x = torch.cat([mean_x, max_x], dim=1)
        return x

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.forward_graph_embedding(x, edge_index, edge_weight, batch)
        x = self.activation(self.fc1(x))
        output = self.fc2(x)
        return output.squeeze(-1)