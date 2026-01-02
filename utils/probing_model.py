import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU, Embedding, Dropout
from absl import logging
import os

# Import PyTorch Geometric (required)
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool


def detect_pyg_device():
    """
    Detect optimal device for PyTorch Geometric operations.
    Priority: ROCm → CUDA → CPU
    
    Detection strategy:
    1. Check if ROCm/HIP is installed (torch.version.hip)
    2. If ROCm, try to initialize device even if torch.cuda.is_available() is False
    3. Check if CUDA is available (torch.cuda.is_available())
    4. Fall back to CPU
    
    Returns:
        tuple: (device, device_info_str)
    """
    import os
    
    # Check for ROCm/HIP in PyTorch version
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    if is_rocm:
        # ROCm/HIP is compiled into PyTorch, try to use it
        logging.info(f"ROCm/HIP detected in PyTorch: torch.version.hip={torch.version.hip}")
        try:
            device = torch.device("cuda:0")
            
            # Try to get device name
            try:
                device_name = torch.cuda.get_device_name(0)
            except:
                device_name = "ROCm GPU (device name unavailable)"
            
            # Test PyG operations on the device
            test_x = torch.randn(10, 16, device=device)
            test_batch = torch.zeros(10, dtype=torch.long, device=device)
            
            # Test pooling
            _ = global_mean_pool(test_x, test_batch)
            _ = global_max_pool(test_x, test_batch)
            
            device_info = f"ROCm/HIP - {device_name} (HIP {torch.version.hip})"
            logging.info(f"✓ Successfully initialized ROCm device")
            return device, device_info
            
        except RuntimeError as e:
            error_msg = str(e)
            if "No HIP GPUs" in error_msg or "no visible GPU" in error_msg.lower():
                logging.warning(f"ROCm installed but no GPU devices visible: {error_msg}")
                device = torch.device("cpu")
                return device, f"CPU (ROCm available but no GPU devices visible)"
            else:
                logging.warning(f"ROCm initialization error: {e}")
                device = torch.device("cpu")
                return device, f"CPU (ROCm initialization error)"
        except Exception as e:
            logging.warning(f"Unexpected error with ROCm: {e}")
            device = torch.device("cpu")
            return device, f"CPU (ROCm error: {type(e).__name__})"
    
    elif torch.cuda.is_available():
        # CUDA is available (NVIDIA)
        logging.info("CUDA detected (NVIDIA GPU)")
        try:
            device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)
            
            # Test PyG operations on the device
            test_x = torch.randn(10, 16, device=device)
            test_batch = torch.zeros(10, dtype=torch.long, device=device)
            
            # Test pooling
            _ = global_mean_pool(test_x, test_batch)
            _ = global_max_pool(test_x, test_batch)
            
            device_info = f"CUDA - {device_name}"
            logging.info(f"✓ Successfully initialized CUDA device")
            return device, device_info
            
        except Exception as e:
            logging.warning(f"CUDA initialization failed: {e}")
            device = torch.device("cpu")
            return device, f"CPU (CUDA initialization failed)"
    
    else:
        # No GPU available
        logging.info("No GPU detected (no ROCm, no CUDA)")
        device = torch.device("cpu")
        return device, "CPU (no GPU detected)"


# Detect PyG device at module load time
PYG_DEVICE, PYG_DEVICE_INFO = detect_pyg_device()
logging.info(f"PyTorch Geometric backend initialized: {PYG_DEVICE_INFO}")


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
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        
        if nonlinear_activation:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
            
        self.fc1 = Linear(2*hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_output)
        self.dropout = Dropout(dropout)

    def forward_graph_embedding(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
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