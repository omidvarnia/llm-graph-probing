import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, Embedding, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


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
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Changed from batch_norms to layer_norms for stability
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
            self.layer_norms.append(nn.LayerNorm(hidden_channels))  # LayerNorm is more stable than BatchNorm for small batches
        
        if nonlinear_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
            
        self.fc1 = Linear(2*hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_output)
        self.dropout = Dropout(dropout)
        
        # Better initialization
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward_graph_embedding(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = self.layer_norms[i](x)  # Use LayerNorm instead of BatchNorm
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout.p, training=self.training)
        mean_x = global_mean_pool(x, batch)
        max_x = global_max_pool(x, batch)
        x = torch.cat([mean_x, max_x], dim=1)
        return x

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.forward_graph_embedding(x, edge_index, edge_weight, batch)
        x = self.activation(self.fc1(x))
        output = self.fc2(x)
        return output.squeeze(-1)