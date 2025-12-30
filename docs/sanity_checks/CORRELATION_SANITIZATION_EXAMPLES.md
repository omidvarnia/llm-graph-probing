# Correlation Matrix Sanitization - Concrete Numerical Examples

## Example 1: Zero Variance Issue (Most Common NaN Cause)

### Scenario: Question with Identical Token Embeddings in Layer 5

**Question**: "What color is grass?"

**Candidate Answers**:
- True (label=1): "Grass is green"
- False (label=0): "Grass is blue"

**Hidden States at Layer 5** (after attention masking, 4 valid tokens):
```
Question tokens: [What, color, is, grass]

Layer-average hidden state matrix (4 tokens × 12 layer-averages):
Layer_0: [0.523, 0.519, 0.521, 0.520]  ← Very similar values
Layer_1: [0.234, 0.233, 0.235, 0.234]
Layer_2: [0.891, 0.889, 0.890, 0.888]
...
Layer_11: [0.345, 0.344, 0.346, 0.345]  ← Again, very similar

[This could happen if tokens are semantically similar and model
 produces nearly identical representations across the network]
```

### Computing Correlation

```python
# Layer-average states: shape (12, 4) - 12 layer-averages, 4 tokens
import numpy as np

layer_avg_states = np.array([
    [0.523, 0.519, 0.521, 0.520],  # Layer 0
    [0.234, 0.233, 0.235, 0.234],  # Layer 1
    [0.891, 0.889, 0.890, 0.888],  # Layer 2
    # ... 9 more layers ...
    [0.345, 0.344, 0.346, 0.345]   # Layer 11
])

# Calculate std for each layer
stds = np.std(layer_avg_states, axis=1)
print(stds)
# Output: [0.0013, 0.0008, 0.0009, ..., 0.0007]
# All EXTREMELY small (< 0.002)

# When std is very close to 0, Pearson correlation becomes undefined
corr_matrix = np.corrcoef(layer_avg_states)
print(corr_matrix)

# Output:
# [[  1.  , nan  , nan  , nan  , ... , nan  ]
#  [ nan  ,  1.  , nan  , nan  , ... , nan  ]
#  [ nan  , nan  ,  1.  , nan  , ... , nan  ]
#  ...
#  [ nan  , nan  , nan  , nan  , ... ,  1.  ]]
```

### Sanitization Response

```python
# COMPUTATION-TIME CHECK (Line 248)
if not np.isfinite(corr_matrix).all():
    # This condition triggers because NaN values exist
    excluded_count += 1
    excluded_indices.append(question_idx)  # Question marked for exclusion
    continue  # Skip to next question, write NO files
    
# RESULT: Question excluded from training/test
# FILES WRITTEN: NONE
# LOG: exclusions_worker_0.txt contains this question_idx
```

### Why This Happens

**Mathematical Reason**: 
- Pearson correlation: ρ = Cov(X, Y) / (σ_X × σ_Y)
- When σ_X ≈ 0 (zero variance), we get 0/0 = NaN
- Numerically: log(0) or division by very small number

**LLM Reason**:
- Query "What color is grass?" is simple/redundant
- LLM may produce nearly identical activations across layers
- All tokens have minimal variation in their representations

---

## Example 2: Extreme Value (Inf) Issue

### Scenario: Numerical Overflow in Nested Correlations

**Question**: Extremely complex multi-part question with 150 tokens

**What Happens**:
```python
# Very large hidden states (e.g., if normalization fails upstream)
layer_hidden_states = np.array([
    [1e6, 2e6, 1.5e6, ...],   # Token 1 activations
    [0.9e6, 1.9e6, 1.4e6, ...], # Token 2 activations
    # ... 148 more tokens ...
])

# Computing variance on such large numbers
variance = np.var(layer_hidden_states, axis=0)
# variance ≈ [1e12, 1e12, ...]  ← Extremely large

# Covariance computation
covariance = np.cov(layer_hidden_states)
# Can overflow during intermediate calculations

# Results in correlation matrix with Inf values
correlation_matrix = np.corrcoef(layer_hidden_states)
print(correlation_matrix)

# Output:
# [[ 1.0 ,  0.95,  inf , -inf, ...]
#  [ 0.95,  1.0 , -inf ,  inf, ...]
#  [ inf , -inf ,  1.0 , 0.85, ...]
#  ...
```

### Sanitization Response

**COMPUTATION-TIME**:
```python
# Line 246-248: Post-computation validation
if not np.isfinite(correlation_matrix).all():
    # Detects inf/-inf values
    excluded_count += 1
    excluded_indices.append(question_idx)
    continue
```

**DATA-LOADING-TIME** (if somehow it passed computation):
```python
# Line 71 in dataset.py
adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

# Converts any lingering Inf values to 0.0
before = np.array([[1.0, 0.95, inf],
                   [0.95, 1.0, -inf],
                   [inf, -inf, 1.0]])

after = np.nan_to_num(before, nan=0.0, posinf=0.0, neginf=0.0)
# Result:
# [[1.0, 0.95, 0.0],
#  [0.95, 1.0, 0.0],
#  [0.0, 0.0, 1.0]]
```

---

## Example 3: Sparse Data Edge Cases

### Scenario: Pre-saved Sparse Correlation with Artifacts

**Files on Disk**:
```
data/hallucination/truthfulqa/Qwen_Qwen2_5_0_5B/
  └─ 350/
     ├─ label.npy
     ├─ layer_5_corr.npy
     ├─ layer_5_sparse_05_edge_index.npy
     └─ layer_5_sparse_05_edge_attr.npy  ← May have NaN/Inf from save errors
```

**Disk Content** (simulated):
```python
# edge_index: [2, 48] (48 edges)
edge_index = np.array([[0, 0, 1, 2, 3, ...],  # source nodes
                       [1, 2, 3, 4, 5, ...]])  # dest nodes

# edge_attr: [48] (correlation weights)
edge_attr = np.array([0.85, nan, -0.52, inf, 0.23, -inf, 0.17, ...])
#                           ↑ potential issues from earlier save
```

### Sanitization at Load Time

**DATASET.PY - Lines 82-83**:
```python
# Load from disk
edge_attr_np = np.load(data_path / f"layer_{layer_idx}_sparse_{density_tag}_edge_attr.npy")
# edge_attr_np: [0.85, nan, -0.52, inf, 0.23, -inf, 0.17, ...]

# STEP 1: Clean NaN/Inf
edge_attr_np = np.nan_to_num(edge_attr_np, nan=0.0, posinf=0.0, neginf=0.0)
# Result: [0.85, 0.0, -0.52, 0.0, 0.23, 0.0, 0.17, ...]

# STEP 2: Clamp to [-1, 1]
edge_attr_np = np.clip(edge_attr_np, -1.0, 1.0)
# Result: [0.85, 0.0, -0.52, 0.0, 0.23, 0.0, 0.17, ...]
# (no changes this time since all in range)

# Convert to tensor
edge_attr = torch.from_numpy(edge_attr_np).float()
# Safe for GPU
```

**Why Two Steps?**
1. **np.nan_to_num**: Handles NaN/Inf from numerical errors
2. **np.clip**: Handles values outside [-1, 1] from rounding errors or data corruption

---

## Example 4: GNN Forward-Pass Sanitization

### Scenario: Edge Weights Become NaN During Computation

**Input to GCN Layer**:
```python
x = torch.tensor([[0.5, 0.3, nan],   # Node 0 features
                  [0.8, nan, 0.2],   # Node 1 features
                  [nan, 0.4, 0.6],   # Node 2 features
                  [0.1, 0.2, 0.3]])  # Node 3 features

edge_index = torch.tensor([[0, 1, 2],  # source
                          [1, 2, 3]])  # dest

edge_attr = torch.tensor([0.85, -0.52, 0.23])  # Correlations
```

### Step 1: Input Cleaning

```python
# Line 50: Clean node features
messages = torch.nan_to_num(x[src])
# src indices: [0, 1, 2]
# x[src] before: [[0.5, 0.3, nan],
#                 [0.8, nan, 0.2],
#                 [nan, 0.4, 0.6]]

messages = torch.nan_to_num(x[src])
# messages after: [[0.5, 0.3, 0.0],
#                  [0.8, 0.0, 0.2],
#                  [0.0, 0.4, 0.6]]
```

### Step 2: Edge Weight Cleaning & Clamping

```python
# Line 51-55: Process edge weights
ew = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
# Result: [0.85, -0.52, 0.23] (no change - already clean)

ew = ew.clamp(min=-1.0, max=1.0)
# Result: [0.85, -0.52, 0.23] (no change - in range)

# Apply edge weights to messages
messages = messages * ew.view(-1, 1)  # Broadcast multiply
# Result:
# [0.5, 0.3, 0.0] * 0.85  = [0.425, 0.255, 0.0]
# [0.8, 0.0, 0.2] * -0.52 = [-0.416, 0.0, -0.104]
# [0.0, 0.4, 0.6] * 0.23  = [0.0, 0.092, 0.138]
```

### Step 3: Aggregation & Normalization

```python
# Index add to destination nodes
# dst indices: [1, 2, 3]
out = torch.zeros_like(x)  # [4, 3]
out.index_add_(0, dst, messages)

# out after aggregation:
# Node 0: [0, 0, 0]
# Node 1: [0.425, 0.255, 0.0]        # Received from node 0
# Node 2: [-0.416, 0.0, -0.104]      # Received from node 1
# Node 3: [0.0, 0.092, 0.138]        # Received from node 2

# Compute in-degree (how many messages each node received)
deg = torch.tensor([[0], [1], [1], [1]])  # In-degrees

# Normalize by degree
out = out / deg.clamp(min=1)
# Result: out remains same (all degrees ≥ 1)
```

### Step 4: Final Cleaning

```python
# Line 60: Clean output
out = torch.nan_to_num(out)
# Catches any NaN produced by degree normalization
# (unlikely here, but defensive)
```

---

## Example 5: Complete Question Lifecycle

### Question #42: Passes All Checks ✓

```
INPUT: Hidden states from LLM forward pass
│
├─ Attention masking → 12 valid tokens (PASS: ≥2)
│
├─ COMPUTATION-TIME SANITIZATION
│  ├─ Layer-avg corr: 12×12 matrix, all finite ✓
│  ├─ Layer-5 corr: 12×12 matrix, all finite ✓
│  ├─ Layer-6 corr: 12×12 matrix, all finite ✓
│  ├─ ... (layers 7-11 all pass)
│  └─ Combined corr: 84×84 matrix (12*7 layers), all finite ✓
│
├─ FILES WRITTEN:
│  └─ data/hallucination/truthfulqa/Qwen_Qwen2_5_0_5B/42/
│     ├─ label.npy (class label)
│     ├─ layer_average_corr.npy
│     ├─ layer_5_corr.npy
│     ├─ layer_5_corr_thresh_05.npy
│     ├─ layer_5_sparse_05_edge_index.npy
│     ├─ layer_5_sparse_05_edge_attr.npy
│     ├─ ... (all other layers)
│     └─ ... (all other files)
│
├─ DATA LOADING: Training/Evaluation
│  ├─ Load from disk → np.nan_to_num() + clamp
│  └─ Create PyTorch Data object
│
├─ GNN FORWARD PASS
│  ├─ torch.nan_to_num(node_features)
│  ├─ torch.nan_to_num(edge_weights) + clamp
│  ├─ Aggregation
│  └─ torch.nan_to_num(output)
│
└─ RESULT: Question #42 in train/test split, successfully processed
```

### Question #500: Fails Computation ✗

```
INPUT: Hidden states from LLM forward pass
│
├─ Attention masking → 8 valid tokens (PASS: ≥2)
│
├─ COMPUTATION-TIME SANITIZATION
│  ├─ Layer-avg corr: Compute...
│     └─ FOUND NaN (zero variance in some layers)
│        VALIDATION FAILS (line 248)
│        excluded_count++
│        excluded_indices.append(500)
│        CONTINUE (skip to next question)
│
├─ FILES WRITTEN: NONE
│     The question output directory is not created
│
├─ TRACKING:
│  └─ exclusions_worker_0.txt contains "500"
│
├─ DATA LOADING: SKIPPED (no files to load)
│
├─ GNN FORWARD PASS: SKIPPED
│
└─ RESULT: Question #500 not in any dataset, excluded from analysis
```

---

## Summary Table: Sanitization Coverage

| Issue Type | Example | Detection | Fix | Result |
|-----------|---------|-----------|-----|--------|
| Zero variance | Token embeddings identical | `np.isfinite()` check | Exclude question | No output files |
| NaN in computation | Pearson(zero_var) | Post-corr validation | Exclude question | No output files |
| Inf from overflow | Large values in covariance | Post-corr validation | Exclude question | No output files |
| NaN in saved data | Disk corruption | `np.nan_to_num()` load | Replace with 0.0 | Safe tensor |
| Inf in saved data | Rounding error | `np.clip()` | Bound to [-1, 1] | Safe tensor |
| NaN during GNN | Aggregation error | `torch.nan_to_num()` | Replace with 0.0 | Stable forward pass |
| Extreme edge weights | Values > 1.0 | `clamp()` after load | Clamp to [-1, 1] | Valid correlations |
| Division by zero | deg=0 in normalize | `clamp(min=1)` | Safe division | Finite output |

---

## Key Insights

1. **Computation is Strictest**: Questions with ANY non-finite correlation are completely excluded before any files are written

2. **Data Loading is Defensive**: Assumes disk data might have artifacts; cleans proactively

3. **GNN is Paranoid**: Every operation that could produce NaN gets sanitized

4. **Exclusions are Tracked**: For auditing, we know exactly which questions failed and where

5. **Triple Redundancy**: If something slips through earlier stages, it will be caught later

This ensures numerically stable training and prevents downstream NaN propagation that would crash the GNN.
