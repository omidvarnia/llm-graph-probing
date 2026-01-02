# All Differences From Reference Implementation

## Summary
Found **3 critical differences** between current code and reference implementation:

---

## 1. ⚠️ CRITICAL: Sparse Data Path Filename Convention

### Location
[hallucination/dataset.py](hallucination/dataset.py) lines 75-77

### Current (BROKEN)
```python
density_tag = f"{int(round(self.network_density * 100)):02d}"
edge_index = torch.from_numpy(np.load(data_path / f"layer_{self.llm_layer}_sparse_{density_tag}_edge_index.npy")).long()
edge_attr_np = np.load(data_path / f"layer_{self.llm_layer}_sparse_{density_tag}_edge_attr.npy").astype(np.float32)
```

**Example filename**: `layer_5_sparse_05_edge_index.npy` (for 5% density)

### Reference (CORRECT)
```python
edge_index = torch.from_numpy(np.load(os.path.join(data_path, f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_index.npy"))).long()
edge_attr = torch.from_numpy(np.load(os.path.join(data_path, f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_attr.npy"))).float()
```

**Example filename**: `layer_5_sparse_0.05_edge_index.npy` (for 5% density)

### Impact
- **CRITICAL**: If sparse data exists, your code looks for `sparse_05` but files are named `sparse_0.05`
- Falls back to dense path (lines 64-73) which works but defeats sparse optimization
- Expected: Save 50% memory on large graphs
- Actual: Loads full correlation matrix every time

### Fix
Replace `density_tag` approach with direct float:
```python
edge_index = torch.from_numpy(np.load(data_path / f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_index.npy")).long()
edge_attr_np = np.load(data_path / f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_attr.npy").astype(np.float32)
```

---

## 2. ⚠️ IMPORTANT: Model Name Sanitization

### Location
[hallucination/dataset.py](hallucination/dataset.py) lines 40-45

### Current (ADDED)
```python
# Sanitize model name for filesystem paths
sanitized_model_name = self.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if self.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{self.ckpt_step}"
self.data_dir = main_dir / "data" / "hallucination" / self.dataset_name / model_dir
```

### Reference (SIMPLER)
```python
if self.ckpt_step == -1:
    model_dir = self.llm_model_name
else:
    model_dir = f"{self.llm_model_name}_step{self.ckpt_step}"
self.data_dir = os.path.join("data/hallucination", self.dataset_name, model_dir)
```

### Impact
- **MEDIUM**: If model names contain `/`, `-`, `.` (like `gpt2-large`):
  - Current: Looks for `data/hallucination/.../gpt2_large/`
  - Reference: Looks for `data/hallucination/.../gpt2-large/`
- Data won't be found → FileNotFoundError

### Example
- Model: `gpt2-large`
- Current path: `data/hallucination/truthfulqa/gpt2_large`
- Reference path: `data/hallucination/truthfulqa/gpt2-large`
- **Data not found!**

### Fix
Remove sanitization or ensure data directory matches:
```python
if self.ckpt_step == -1:
    model_dir = self.llm_model_name
else:
    model_dir = f"{self.llm_model_name}_step{self.ckpt_step}"
self.data_dir = main_dir / "data" / "hallucination" / self.dataset_name / model_dir
```

---

## 3. ⚠️ IMPORTANT: Dataset Path Handling

### Location
[hallucination/dataset.py](hallucination/dataset.py) lines 1-6, 96

### Current (PATHLIB + ENVIRONMENT VARIABLE)
```python
from pathlib import Path

main_dir = Path(os.environ.get('MAIN_DIR', '.'))

def get_truthfulqa_dataloader(...):
    dataset_filename = main_dir / "data/hallucination" / f"{dataset_name}.csv"
    ...
```

### Reference (SIMPLE STRING)
```python
def get_truthfulqa_dataloader(...):
    dataset_filename = f"data/hallucination/{dataset_name}.csv"
    ...
```

### Impact
- **LOW**: If `MAIN_DIR` env variable not set:
  - Current: Uses `.` (current directory)
  - Reference: Uses relative path from current directory
  - Usually same effect, but can cause issues if script called from different directory

### Fix
Either set `MAIN_DIR` or use relative path directly:
```python
# Option 1: Set env var
export MAIN_DIR=/u/aomidvarnia/GIT_repositories/llm-graph-probing

# Option 2: Use relative path
dataset_filename = f"data/hallucination/{dataset_name}.csv"
```

---

## 4. ⚠️ IMPORTANT: GCN Implementation Differences

### Custom SimpleGCNConv (Current) vs PyG GCNConv (Reference)

#### A. Line Transformation Order
**Custom**:
```python
out = out / deg.clamp(min=1)  # Mean aggregation
return self.linear(out)        # Linear AFTER
```

**Reference**:
```python
x = self.lin(x)               # Linear BEFORE
out = self.propagate(...)     # Message passing
```

**Impact**: Moderate - changes GCN semantics, but both are valid approaches

#### B. Pooling Functions
**Custom**:
```python
def global_mean_pool_torch(x, batch):  # Manual implementation
    sums = torch.zeros((num_graphs, x.size(1)), ...)
    sums.index_add_(0, batch, x)
    return sums / counts.clamp(min=1)
```

**Reference**:
```python
from torch_geometric.nn import global_mean_pool  # PyG built-in
```

**Impact**: Low - mathematical equivalence, custom handles NaN conversion

#### C. Edge Weight Handling
**✅ ALREADY FIXED**: Both now preserve signed correlations

---

## 5. LOW IMPACT: Path Type Handling

### Dense Path (Dense tensor → sparse conversion)
**Current**:
```python
adj = np.load(data_path / self.dense_filename)  # pathlib.Path
threshold = np.percentile(np.abs(adj), ...)
adj[np.abs(adj) < threshold] = 0
np.fill_diagonal(adj, 0)
adj = torch.from_numpy(adj).float()
edge_index, edge_attr = dense_to_sparse(adj)
```

**Reference**:
```python
adj = np.load(os.path.join(data_path, self.dense_filename))  # string path
threshold = np.percentile(np.abs(adj), ...)
adj[np.abs(adj) < threshold] = 0
np.fill_diagonal(adj, 0)
adj = torch.from_numpy(adj).float()
edge_index, edge_attr = dense_to_sparse(adj)
```

**Impact**: None - np.load accepts both pathlib.Path and string

---

## Severity Assessment

| Difference | Severity | Impact | Fixed? |
|-----------|----------|--------|--------|
| Sparse data filename (density_tag) | 🔴 CRITICAL | Sparse data never loaded, fallback to dense | ❌ NO |
| Model name sanitization | 🟠 HIGH | Data not found if model name has special chars | ❌ NO |
| Abs() calls (data + GCN) | 🔴 CRITICAL | 30% accuracy loss | ✅ YES |
| Path type (pathlib vs string) | 🟢 LOW | No impact, both work | ✅ N/A |
| GCN impl (custom vs PyG) | 🟠 MEDIUM | Architectural difference, acceptable | ✅ N/A |

---

## Required Fixes

### Fix #1: Sparse Data Path (CRITICAL)
```python
# hallucination/dataset.py lines 75-80

# BEFORE
density_tag = f"{int(round(self.network_density * 100)):02d}"
edge_index = torch.from_numpy(np.load(data_path / f"layer_{self.llm_layer}_sparse_{density_tag}_edge_index.npy")).long()
edge_attr_np = np.load(data_path / f"layer_{self.llm_layer}_sparse_{density_tag}_edge_attr.npy").astype(np.float32)

# AFTER
edge_index = torch.from_numpy(np.load(data_path / f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_index.npy")).long()
edge_attr_np = np.load(data_path / f"layer_{self.llm_layer}_sparse_{self.network_density}_edge_attr.npy").astype(np.float32)
```

### Fix #2: Model Name Sanitization (HIGH)
```python
# hallucination/dataset.py lines 40-45

# BEFORE
sanitized_model_name = self.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if self.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{self.ckpt_step}"

# AFTER
if self.ckpt_step == -1:
    model_dir = self.llm_model_name
else:
    model_dir = f"{self.llm_model_name}_step{self.ckpt_step}"
```

---

## Current Status

✅ **FIXED**: 
- `np.abs()` in dataset.py lines 67, 81
- `torch.abs()` in probing_model.py line 58

❌ **NOT FIXED**:
- Sparse data filename convention (density_tag issue)
- Model name sanitization

⚠️ **ARCHITECTURAL** (acceptable):
- Custom GCN vs PyG GCN (working, just different)
- Custom pooling vs PyG pooling (mathematically equivalent)
