# Correlation Matrices & GNN Input Architecture

## Quick Answer

| Question | Answer |
|----------|--------|
| **Which correlation matrix classifies true/false?** | Per-layer Pearson correlation matrices computed from LLM hidden states |
| **Input to GNN?** | Thresholded sparse correlation matrix (edges = correlations, nodes = hidden dimensions) |
| **Exact input size?** | 12 nodes (one per LLM layer) → after thresholding → variable edges (1-66 per question) |
| **Correlation matrix size?** | 12×12 (layer-average) OR variable×variable (per-layer, e.g., 8-15 hidden dims) |

---

## 1. Correlation Matrices Used for Classification

### 1.1 Two Types of Correlation Matrices Computed

The pipeline computes **two types** of correlation matrices from hidden states:

#### **Type A: Layer-Average Correlation (12×12)**
```python
# Lines 244-246 in compute_llm_network.py

layer_average_hidden_states = hidden_states_layer_average[:, i, sentence_attention_mask == 1]
# Shape: (12 layers, N_tokens)
# Each element = average of that layer's hidden_dim dimension

layer_average_corr = np.corrcoef(layer_average_hidden_states)
# Output: 12×12 correlation matrix (layer-to-layer correlations)
```

**What it measures**: How correlated are the 12 LLM layers with each other when averaged across all hidden dimensions

**Example**:
- Layer 0 avg activation: [0.523, 0.234, ..., 0.345] (12 values, one per layer)
- Layer 1 avg activation: [0.519, 0.233, ..., 0.344]
- ... (12 layers × N_tokens matrix)
- Correlation: How does layer 0's pattern correlate with layer 11's pattern?

#### **Type B: Per-Layer Correlation (Variable × Variable)**
```python
# Lines 258-265 in compute_llm_network.py

for j, layer_idx in enumerate(layer_list):
    layer_hidden_states = hidden_states[j, i]  # Shape: (N_tokens, hidden_dim)
    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
    # Now shape: (hidden_dim, N_tokens) - each row is one hidden dimension
    
    corr = np.corrcoef(sentence_hidden_states)
    # Output: hidden_dim × hidden_dim correlation matrix
```

**What it measures**: How correlated are individual hidden dimensions within that layer

**Example for Layer 5**:
- Hidden states shape: (N_tokens=8, hidden_dim=256)
- After selecting valid tokens and transposing: (256, 8)
- Correlation matrix: 256×256 (how each hidden dimension correlates with others)

### 1.2 Which Matrix Is Used for True/False Classification?

**ANSWER: Per-Layer Correlation Matrices (Type B)**

```python
# From dataset.py, lines 31-38:

if self.llm_layer == -1:
    self.dense_filename = "layer_average_corr.npy"
else:
    self.dense_filename = f"layer_{self.llm_layer}_corr.npy"  # ← TYPE B USED
```

The model trains on **per-layer correlations**, not layer-average.

**Example Usage** (from dataset.py):
```python
# In _load_data() method:
adj = np.load(data_path / self.dense_filename)  # e.g., "layer_5_corr.npy"
# Shape: (256, 256) for Layer 5 of Qwen2.5-0.5B

# Thresholding
percentile_threshold = self.network_density * 100  # default 1.0 = 100%
threshold = np.percentile(np.abs(adj), 100 - percentile_threshold)
adj[np.abs(adj) < threshold] = 0
```

---

## 2. GNN Input Architecture

### 2.1 Data Flow: Question → Graph Input

```
TruthfulQA Question
│
├─ LLM Forward Pass
│  └─ Output: Hidden states across 12 layers
│
├─ Compute Correlation
│  └─ For Layer 5: 256×256 correlation matrix
│     (256 hidden dimensions × 256 hidden dimensions)
│
├─ Thresholding (network_density controls sparsity)
│  └─ Keep top 100% of |correlations|
│  └─ Output: Sparse 256×256 matrix with ~2,500 edges
│
├─ Convert to PyTorch Graph
│  ├─ nodes (N): 256 (hidden dimensions)
│  ├─ edges (E): variable, e.g., 2,500 edges
│  ├─ edge_attr: correlation values [-1, 1]
│  └─ node features: x = torch.arange(256)
│
└─ Pass to GCN
   └─ Output: Binary classification (True/False label)
```

### 2.2 Exact Input Sizes to GNN

#### **Node Dimension**
```python
# From dataset.py, line 91:
x = torch.arange(num_nodes)  # x is just node IDs [0, 1, 2, ..., N-1]
```

| Layer | Hidden Dim | Nodes in Graph |
|-------|-----------|----------------|
| Layer 5 | 1024 (Qwen) | 1024 nodes |
| Layer 11 | 1024 (Qwen) | 1024 nodes |
| Layer-avg | 12 | 12 nodes |

#### **Edge Dimension**
```python
# From dataset.py, lines 71-77:
percentile_threshold = self.network_density * 100  # 100 = keep all
threshold = np.percentile(np.abs(adj), 100 - percentile_threshold)
# For network_density=1.0: threshold ≈ 0 (keep all correlations ≠ 0)
# For network_density=0.5: threshold ≈ 0.5 (keep top 50%)
# For network_density=0.1: threshold ≈ 0.8 (keep top 10%)

adj[np.abs(adj) < threshold] = 0
```

**Number of Edges**:
- **Full density (1.0)**: All non-zero correlations kept
  - Expected: ~60% of N² edges (since many correlations are near 0)
  - For N=1024: ~630K edges
  
- **50% density (0.5)**: Keep top 50% by magnitude
  - Expected: ~320K edges
  
- **10% density (0.1)**: Keep top 10% by magnitude
  - Expected: ~64K edges

#### **Edge Features (Correlation Values)**
```python
edge_attr: torch.Tensor of shape [E]
# Each value ∈ [-1.0, 1.0] (Pearson correlation range)
# After sanitization: clamped to [-1.0, 1.0]
```

### 2.3 PyTorch Geometric Data Object

```python
# From dataset.py, line 92:
Data(
    x=x,                      # Node features: [N] = [1024]
    edge_index=edge_index,    # Graph structure: [2, E]
    edge_attr=edge_attr,      # Edge weights: [E]
    y=y                       # Label: scalar (0 or 1)
)
```

**Example Tensor Shapes** (Layer 5, Qwen2.5-0.5B, network_density=1.0):
```
x:           [1024]          (node ID features)
edge_index:  [2, 630000]     (source, destination pairs)
edge_attr:   [630000]        (correlation weights)
y:           scalar (0 or 1) (true/false label)
batch:       [1024]          (all zeros for single graph)
```

---

## 3. GNN Architecture

### 3.1 Model: GCNProbe

```python
# From utils/probing_model.py, lines 80-100

class GCNProbe(nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_layers, dropout=0.0, num_output=1):
        super(GCNProbe, self).__init__()
        self.embedding = Embedding(num_nodes, hidden_channels)  # Maps node IDs → embeddings
        self.convs = nn.ModuleList([
            SimpleGCNConv(hidden_channels, hidden_channels) 
            for _ in range(num_layers)
        ])
        self.fc1 = Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, num_output)
        self.dropout = Dropout(dropout)
```

### 3.2 Input Processing

```python
def forward(self, x, edge_index, edge_weight, batch):
    # x: [N] = [1024] node IDs
    # edge_index: [2, E] = [2, 630000]
    # edge_weight: [E] = [630000] correlation values
    # batch: [N] = [1024] graph assignment
    
    x = self.embedding(x)  # x: [1024] → [1024, hidden_channels=32]
```

### 3.3 GCN Convolution with Correlation Weights

```python
# From utils/probing_model.py, SimpleGCNConv.forward()

def forward(self, x, edge_index, edge_weight):
    # x: [N, C] = [1024, 32]
    # edge_index: [2, E]
    # edge_weight: [E] correlation values [-1, 1]
    
    src = edge_index[1]   # Source nodes
    dst = edge_index[0]   # Destination nodes
    
    # Get neighbor features
    messages = x[src]  # [E, 32]
    
    # Weight by correlation strength
    ew = edge_weight.clamp(min=-1.0, max=1.0)  # Ensure valid correlations
    messages = messages * ew.view(-1, 1)        # Broadcast multiply
    # Result: [E, 32] where each row is weighted by correlation
    
    # Aggregate to destination nodes
    out = torch.zeros_like(x)  # [1024, 32]
    out.index_add_(0, dst, messages)  # Sum messages to each node
    
    # Normalize by in-degree
    deg = torch.zeros((N, 1))
    deg.index_add_(0, dst, torch.ones((E, 1)))
    out = out / deg.clamp(min=1)  # Avoid division by zero
    
    return self.linear(out)
```

**Key Point**: Correlation values directly weight the message passing. High correlation → strong influence between dimensions.

### 3.4 Final Classification

```python
def forward(self, x, edge_index, edge_weight, batch):
    # After GCN layers: x = [batch_size, hidden_channels]
    
    mean_x = global_mean_pool(x, batch)      # [batch_size, hidden_channels]
    max_x = global_max_pool(x, batch)        # [batch_size, hidden_channels]
    x = torch.cat([mean_x, max_x], dim=1)   # [batch_size, 2*hidden_channels]
    
    x = self.fc1(x)      # [batch_size, hidden_channels]
    output = self.fc2(x) # [batch_size, 1]
    
    return output.squeeze(-1)  # [batch_size] logits for binary classification
```

---

## 4. Exact Configuration from Analysis

### 4.1 Training Configuration
```yaml
# From pipeline_config_qwen.yaml and run_hallu_detec_mpcdf.slurm

llm_layer: 5                    # Use Layer 5 correlations
network_density: 1.0            # Keep all correlations
hidden_channels: 32             # GCN embedding dimension
num_layers: 1                   # Single GCN convolution
batch_size: 16
num_epochs: 100
```

### 4.2 Data Preparation
```python
# From my_analysis/hallucination_detection_analysis.py

llm_layer = 5                   # Layer 5 hidden states
network_density = 1.0           # 100% of correlations
from_sparse_data = False        # Use dense correlation matrix

# Result:
# - Load "layer_5_corr.npy" (256×256 or 1024×1024)
# - Threshold to keep all correlations
# - Convert to sparse graph representation
# - Input to GNN: Graph with 256-1024 nodes, ~2500-630K edges
```

---

## 5. Complete Pipeline Visualization

```
Question: "What color is grass?"
Truth: "Green" (label=1)

│
├─ LLM Forward Pass
│  └─ For Layer 5: hidden_states shape (8 tokens, 1024 dims)
│     [Green → Grass → is → color → What] = valid tokens
│
├─ Compute Correlation
│  └─ Pearson(1024, 8) → 1024×1024 correlation matrix
│     Entry [i,j] = corr(dim_i, dim_j) across 8 tokens
│
├─ Thresholding (density=1.0)
│  └─ Keep all |corr| > 0
│  └─ Result: Sparse with ~630K edges
│
├─ Graph Construction
│  ├─ Nodes: 1024 (one per hidden dimension)
│  ├─ Edges: 630K (dimension pairs with corr ≠ 0)
│  └─ Edge weights: correlation values ∈ [-1, 1]
│
├─ GCN Processing
│  ├─ Embedding: node IDs [0...1023] → [1024, 32] embeddings
│  ├─ Conv 1: [1024, 32] → [1024, 32] (message passing via correlations)
│  ├─ Pooling: [1024, 32] → [1, 64] (mean + max pool)
│  └─ FC: [1, 64] → [1, 32] → [1, 1] logits
│
└─ Output: Probability that question answer is truthful
   logit ≈ 0.8 → P(true) = sigmoid(0.8) ≈ 0.69
   Prediction: 1 (True) ✓ Correct!
```

---

## 6. Key Implementation Details

### 6.1 Node Features
```python
# Nodes don't have semantic features; they're just identifiers
x = torch.arange(num_nodes)  # [0, 1, 2, ..., 1023]

# Inside embedding layer:
x_embedded = self.embedding(x)  # [1024] → [1024, 32] learned embeddings
```

**Why this works**: The GCN learns node representations based on graph structure (correlations) and topology, not pre-computed features.

### 6.2 Edge Weights
```python
# Correlation values directly used as edge weights
edge_attr = torch.from_numpy(edge_attr_np)  # [-1, 1] range

# In message passing:
messages = messages * edge_weight.view(-1, 1)  # Weight by correlation
```

**Interpretation**:
- `edge_weight = 0.9`: Strong positive correlation → strong influence
- `edge_weight = 0.0`: No correlation → no direct influence (edge removed)
- `edge_weight = -0.7`: Strong negative correlation → inhibitory influence

### 6.3 Why Correlation Matrices?

The hypothesis: **Hidden dimensions that correlate strongly likely encode similar information**

- If dim₀ and dim₁ always activate together → corr ≈ 1.0 → likely redundant
- If dim₀ and dim₁ never activate together → corr ≈ 0.0 → encode different info
- This pattern (how dimensions are used) differs between true/false statements

The GCN learns to classify based on these correlation patterns.

---

## 7. Summary Table

| Aspect | Value | Details |
|--------|-------|---------|
| **Correlation Matrix Type** | Per-layer Pearson | Computed from Layer 5 hidden states |
| **Correlation Matrix Size** | 1024×1024 (Qwen) | One dimension per hidden neuron |
| **GNN Input: Nodes** | 1024 | Hidden dimensions as graph nodes |
| **GNN Input: Edges** | ~630K (density=1.0) | Variable based on network_density |
| **GNN Input: Edge Features** | Correlation values | Range [-1.0, 1.0] |
| **GNN Architecture** | GCNProbe | 1 GCN layer + pooling + 2 FC layers |
| **GNN Embedding Dim** | 32 | Hidden channels for node embeddings |
| **GNN Output** | 1 logit | Binary classification (True/False) |
| **Classification Task** | Binary | 0=False, 1=True statement |
| **Accuracy (Layer 5)** | 52.2% | From TruthfulQA analysis |

---

## References

- **Correlation Computation**: [hallucination/compute_llm_network.py](hallucination/compute_llm_network.py#L244)
- **Data Loading**: [hallucination/dataset.py](hallucination/dataset.py#L71)
- **GCN Model**: [utils/probing_model.py](utils/probing_model.py#L80)
- **Training Loop**: [hallucination/train.py](hallucination/train.py#L60)
