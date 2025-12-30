# GNN Input Data Flow: Complete Numerical Example

## End-to-End Example: Question #42 from TruthfulQA

### Step 1: Question Text
```
Question: "What color is grass?"
Answers: 
  - True: "Grass is green"
  - False: "Grass is blue"
Label: 1 (True)
```

### Step 2: LLM Forward Pass (Qwen2.5-0.5B)

```
Model Input (formatted):
  "What color is grass? Grass is green"
  
Tokenization:
  Token IDs: [1, 2, 3, 4, 5, 6, 7] (7 tokens)
  Tokens:    ["What", "color", "is", "grass", "?", "Grass", "green"]
  
Attention Mask: [1, 1, 1, 1, 1, 1, 1] (all 7 tokens valid)

Model Output:
  hidden_states shape: 13 tuples (embedding + 12 layers)
  Each: (batch_size=1, seq_len=7, hidden_dim=1024)
  
Layer 5 Hidden States:
  Shape: (1, 7, 1024)  ← 1 sample, 7 tokens, 1024-dim hidden states
  
  Values (transposed for visualization):
  ┌─────────────────────────────────────────────────────┐
  │ Token     │What   │color │is    │grass │?     │Grass │green  │
  │ Dim 0     │ 0.523 │-0.34 │ 0.12 │ 0.89 │ 0.23 │ 0.81 │ 0.91  │
  │ Dim 1     │-0.12 │ 0.45 │ 0.67 │-0.23 │ 0.55 │-0.21 │ 0.48  │
  │ Dim 2     │ 0.78 │ 0.54 │ 0.61 │ 0.71 │ 0.52 │ 0.73 │ 0.68  │
  │ ...       │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │ ...   │
  │ Dim 1023  │ 0.34 │ 0.47 │ 0.29 │ 0.83 │ 0.15 │ 0.79 │ 0.92  │
  └─────────────────────────────────────────────────────┘
```

### Step 3: Extract Valid Tokens Using Attention Mask

```python
# From compute_llm_network.py, line 240:
sentence_attention_mask = attention_mask[i]  # [1, 1, 1, 1, 1, 1, 1]
sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T

# After extraction:
sentence_hidden_states shape: (1024, 7)
# Each row: one hidden dimension
# Each column: activation at one token

sentence_hidden_states:
  Dim 0:  [0.523, -0.34, 0.12, 0.89, 0.23, 0.81, 0.91]
  Dim 1:  [-0.12, 0.45, 0.67, -0.23, 0.55, -0.21, 0.48]
  ...
  Dim 1023: [0.34, 0.47, 0.29, 0.83, 0.15, 0.79, 0.92]
```

### Step 4: Compute Pearson Correlation

```python
# From compute_llm_network.py, line 263:
corr = np.corrcoef(sentence_hidden_states)
# Input shape: (1024, 7)
# Output shape: (1024, 1024)

# Example correlations:
corr_matrix = [
  # Dim0  Dim1  Dim2  ...  Dim1023
  [1.00, 0.23, 0.45, ..., 0.67],      # Dim 0
  [0.23, 1.00, 0.12, ..., 0.34],      # Dim 1
  [0.45, 0.12, 1.00, ..., 0.51],      # Dim 2
  [...,  ...,  ...,  ..., ...],
  [0.67, 0.34, 0.51, ..., 1.00],      # Dim 1023
]

# Properties:
# - Diagonal: all 1.0 (each dimension correlates perfectly with itself)
# - Symmetric: corr[i,j] = corr[j,i]
# - Range: [-1, 1] (Pearson correlation bounds)
# - Density: ~60% non-zero (many dims weakly correlated ≈ 0)
```

### Step 5: Validation Check (Sanitization Layer 1)

```python
# From compute_llm_network.py, line 265:
if not np.isfinite(corr).all():
    # Check for NaN or Inf values
    excluded_count += 1
    excluded_indices.append(question_idx)
    continue  # Skip this question

# For Question #42:
# All values in corr are finite ✓
# Question proceeds to next step
```

### Step 6: Thresholding (Network Density Control)

```python
# From dataset.py, line 75:
percentile_threshold = network_density * 100
# network_density = 1.0 (from config)
# percentile_threshold = 100

threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
# np.percentile(np.abs(corr), 0) = minimum absolute correlation
# threshold ≈ 0.0 (keep everything)

corr_thresholded = corr.copy()
corr_thresholded[np.abs(corr_thresholded) < threshold] = 0
np.fill_diagonal(corr_thresholded, 0)  # Remove self-loops

# Result: All non-zero correlations kept, diagonal set to 0

# If network_density = 0.5 instead:
# threshold = np.percentile(np.abs(corr), 50)  # median
# Only top 50% of |correlations| kept (rest → 0)
```

### Step 7: Convert Dense to Sparse

```python
# From dataset.py, line 75-76:
adj = torch.from_numpy(corr_thresholded).float()
edge_index, edge_attr = dense_to_sparse(adj)

# dense_to_sparse conversion:
# Input: 1024×1024 dense matrix (most entries are 0)
# Output: 
#   edge_index: [2, num_edges]
#   edge_attr: [num_edges]

# For network_density=1.0, typical results:
# num_edges ≈ 1024 * 1024 * 0.6 = 630,000 edges

edge_index shape: [2, 630000]
# Example edge_index:
# [[  0,   0,   0,  1,   1,   2, ...],     # Source dimensions
#  [  5, 234, 812, 10, 345, 100, ...]]     # Destination dimensions

edge_attr shape: [630000]
# Correlation values for each edge
# [0.87, 0.23, -0.45, 0.92, 0.15, 0.63, ...]
```

### Step 8: Sanitization Layer 2 (Data Loading)

```python
# From dataset.py, line 82-83:
edge_attr_np = np.nan_to_num(edge_attr_np, nan=0.0, posinf=0.0, neginf=0.0)
edge_attr_np = np.clip(edge_attr_np, -1.0, 1.0)

# For Question #42: No NaN/Inf present
# All values already in [-1, 1]
# No changes to edge_attr

edge_attr_clean: [0.87, 0.23, -0.45, 0.92, 0.15, 0.63, ...]
```

### Step 9: Create PyTorch Geometric Data Object

```python
# From dataset.py, line 91-92:
num_nodes = 1024
x = torch.arange(num_nodes)  # [0, 1, 2, ..., 1023]
y = torch.tensor(1, dtype=torch.long)  # Label: True

data = Data(
    x=x,                    # [1024] node IDs
    edge_index=edge_index,  # [2, 630000]
    edge_attr=edge_attr,    # [630000] correlations
    y=y                     # scalar label: 1
)

# Object properties:
# data.num_nodes = 1024
# data.num_edges = 630000
# data.num_node_features = 1 (just node ID)
# data.num_edge_features = 1 (just correlation)
```

### Step 10: Batch Processing

```python
# From train.py, training loop:
# Multiple questions batched together

batch = Batch.from_data_list([data_42, data_43, ..., data_57])
# 16 graphs batched (batch_size=16)

batch.x shape: [16384] = 16 * 1024 nodes
batch.edge_index shape: [2, 10080000] ≈ 16 * 630000 edges
batch.batch shape: [16384] = graph assignment
# batch.batch[0:1024] = 0 (nodes 0-1023 belong to graph 0)
# batch.batch[1024:2048] = 1 (nodes 1024-2047 belong to graph 1)
# etc.
```

### Step 11: GCN Forward Pass (Model Processing)

#### **Step 11a: Embedding**
```python
# From probing_model.py, line 117:
x = self.embedding(x)
# Input x: [16384] node IDs
# Embedding(1024, 32)
# Output: [16384, 32] learned embeddings

# Per question:
# x (Question #42): [1024, 32] (1024 node embeddings, each 32-dim)
```

#### **Step 11b: First GCN Convolution**
```python
# From probing_model.py, SimpleGCNConv.forward():

src = edge_index[1]   # Source nodes
dst = edge_index[0]   # Destination nodes
messages = x[src]     # [630000, 32] features of source nodes

# Weight by correlation
ew = edge_attr.clamp(min=-1.0, max=1.0)  # [630000]
messages = messages * ew.view(-1, 1)      # [630000, 32]
# Each element multiplied by its edge correlation weight

# Aggregate to destination nodes
out = torch.zeros(1024, 32)  # Initialize output
out.index_add_(0, dst, messages)  # Sum messages

# Normalize by in-degree
deg = torch.zeros(1024, 1)
deg.index_add_(0, dst, torch.ones(630000, 1))
out = out / deg.clamp(min=1)  # [1024, 32]

# Apply linear transformation
x = self.linear(out)  # [1024, 32] → [1024, 32]
```

**Interpretation**: 
- Dimension with high-correlation neighbors gets stronger signal
- Dimension with low-correlation neighbors (isolated) gets weaker signal
- Correlation directly impacts information flow through network

#### **Step 11c: ReLU Activation + Dropout**
```python
x = self.activation(x)           # ReLU: [1024, 32]
x = F.dropout(x, p=0.0, training=True)  # dropout=0.0, so no effect
x = torch.nan_to_num(x)          # Sanitization: [1024, 32]
```

#### **Step 11d: Global Pooling**
```python
# From probing_model.py:
mean_x = global_mean_pool_torch(x, batch)  # [batch_size, 32]
max_x = global_max_pool_torch(x, batch)    # [batch_size, 32]
# For batch_size=16: each shape [16, 32]

# For Question #42 (graph 0 in batch):
# mean_x[0] = average of all 1024 node embeddings
# max_x[0] = max pooling of all 1024 node embeddings

x = torch.cat([mean_x, max_x], dim=1)  # [16, 64]
# Question #42: x[0] = [mean_emb_1, ..., mean_emb_32, max_emb_1, ..., max_emb_32]
```

#### **Step 11e: Classification Layers**
```python
# From probing_model.py, lines 124-125:
x = self.activation(self.fc1(x))  # [16, 64] → [16, 32] → ReLU
output = self.fc2(x)              # [16, 32] → [16, 1]

# Squeeze and get predictions
logits = output.squeeze(-1)  # [16]
# Question #42 logit: logits[0] ≈ 0.73

predictions = (logits > 0.0).long()  # [16]
# Question #42 prediction: predictions[0] = 1 (True)

# True label: y[0] = 1
# Correct! ✓
```

### Step 12: Loss Computation

```python
# From train.py, loss_fn():
loss = F.cross_entropy(output, target, label_smoothing=0.1)
# output: [16, 1] logits
# target: [16] labels

# For batch [1, 0, 1, 1, 0, 1, ...]:
# loss = cross_entropy([0.73, -0.45, 0.82, 0.91, -0.23, 0.68, ...],
#                      [1, 0, 1, 1, 0, 1, ...])
# loss ≈ 0.35 (example value)
```

### Step 13: Backward Pass & Optimization

```python
# From train.py:
loss.backward()  # Compute gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
optimizer.step()  # Update weights

# Gradients flow backward through:
# fc2 → fc1 → pooling → GCN convolution → embedding
# All parameters updated based on loss
```

---

## Summary: Data Shapes Through Pipeline

| Stage | Shape | Content |
|-------|-------|---------|
| **Input Token Sequence** | (7,) | Token IDs for one question |
| **Hidden States (Layer 5)** | (1, 7, 1024) | 1 sample, 7 tokens, 1024 dimensions |
| **After Attention Mask** | (1024, 7) | 1024 dimensions, 7 valid tokens |
| **Correlation Matrix** | (1024, 1024) | Pearson correlations between dimensions |
| **After Thresholding** | (1024, 1024) | Sparse (mostly zeros) |
| **Sparse Representation** | Edge: (2, 630K), Attr: (630K,) | Sparse graph format |
| **Node Features** | (1024,) | Node IDs [0...1023] |
| **Node Embeddings (before GCN)** | (1024, 32) | Embedded node representations |
| **Node Embeddings (after GCN)** | (1024, 32) | Updated by correlation information |
| **Global Mean Pool** | (1, 32) | Aggregated mean of node embeddings |
| **Global Max Pool** | (1, 32) | Aggregated max of node embeddings |
| **Concatenated Pools** | (1, 64) | Combined mean + max |
| **FC1 Output** | (1, 32) | Hidden representation |
| **Final Logit** | (1,) | Single value ∈ ℝ |
| **Prediction** | Scalar {0, 1} | Binary classification result |

---

## Key Insights

### 1. Why Correlations?
The model learns that correlation patterns distinguish true from false statements:
- **True statements**: Certain hidden dimensions co-activate (high correlation)
- **False statements**: Different activation patterns (different correlations)

### 2. Why Graph Neural Network?
A GCN naturally captures:
- **Local structure**: How nearby dimensions (by correlation) influence each other
- **Global structure**: Information aggregated from all dimensions via pooling
- **Non-linearity**: ReLU activations enable complex decision boundaries

### 3. Information Flow
```
Low-correlation dimensions → Isolated nodes → Weak signal to classifier
High-correlation dimensions → Connected nodes → Strong signal to classifier
```

### 4. Scalability
- **Small density (0.1)**: Few edges → fast, simple decision boundaries
- **Large density (1.0)**: Many edges → potentially more expressive, but harder to train

---

## References

This complete pipeline is implemented across:
- **Correlation Computation**: `hallucination/compute_llm_network.py:run_corr()`
- **Graph Construction**: `hallucination/dataset.py:TruthfulQADataset._load_data()`
- **GCN Model**: `utils/probing_model.py:GCNProbe`
- **Training Loop**: `hallucination/train.py:train_model()`
