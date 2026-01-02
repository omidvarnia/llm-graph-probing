# GCN Implementation Comparison: Custom vs PyTorch Geometric

## Summary

Yes, I reviewed both implementations in detail. Here's the critical finding:

**Your custom SimpleGCNConv has a SECOND bug beyond the data loading issue**: It takes `torch.abs(ew)` on edge weights in the forward pass, which **AGAIN destroys negative correlation information**.

## Detailed Comparison

### Reference Implementation (PyTorch Geometric GCNConv)

**File**: `llm-graph-probing_reference_do_not_change/utils/probing_model.py`

```python
self.convs.append(GCNConv(hidden_channels, hidden_channels, 
                         add_self_loops=False, normalize=False))

# Forward pass:
x = conv(x, edge_index, edge_weight)
```

**What it does**:
1. **Linear transformation**: `x = W @ x`  
2. **Message passing**: `out = propagate(edge_index, x=x, edge_weight=edge_weight)`
3. **Edge weights used directly**: No abs(), no modification, **SIGNED weights preserved**
4. **No normalization**: `normalize=False` means no D^{-1/2} A D^{-1/2} transform
5. **No self-loops**: `add_self_loops=False`

**Mathematical formula**:
```
out[i] = Linear(Σ_{j ∈ N(i)} edge_weight[i,j] * x[j])
```

Where `edge_weight[i,j]` can be **positive or negative** (signed correlations).

---

### Your Custom Implementation (SimpleGCNConv)

**File**: `utils/probing_model.py` lines 47-67

```python
def forward(self, x, edge_index, edge_weight):
    src = edge_index[1]
    dst = edge_index[0]
    messages = torch.nan_to_num(x[src])
    if edge_weight is not None:
        ew = torch.nan_to_num(edge_weight, nan=0.0, posinf=0.0, neginf=0.0)
        ew = torch.abs(ew)  # ← SECOND BUG: Takes absolute value!
        messages = messages * ew.view(-1, 1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, messages)
    deg = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
    ones = torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, ones)
    out = out / deg.clamp(min=1)  # Mean aggregation
    out = torch.nan_to_num(out)
    return self.linear(out)
```

**What it does**:
1. **Takes absolute value**: `ew = torch.abs(ew)` ← **BUG!**
2. **Message passing**: Aggregates with `index_add`
3. **Mean normalization**: Divides by degree
4. **Linear transformation**: Applied after aggregation

**Mathematical formula**:
```
out[i] = Linear((1/deg[i]) * Σ_{j ∈ N(i)} |edge_weight[i,j]| * x[j])
```

Where `|edge_weight[i,j]|` is **always positive** (unsigned).

---

## Critical Differences

| Aspect | Reference (PyG GCNConv) | Your Custom (SimpleGCNConv) | Impact |
|--------|-------------------------|------------------------------|--------|
| **Edge weight sign** | Preserved (can be ±) | **Taken abs** (`torch.abs(ew)`) | **CATASTROPHIC** |
| **Aggregation** | Sum (unnormalized) | Mean (normalized by degree) | Moderate |
| **Linear timing** | Before aggregation | After aggregation | Mathematical difference |
| **Self-loops** | Controlled by flag | Never added | Minor |

---

## The Double Bug Problem

You have **TWO places** where edge signs are destroyed:

### Bug #1: Data Loading (PRIMARY)
**File**: `hallucination/dataset.py`
- Line 67: `adj = np.abs(adj)`
- Line 81: `edge_attr_np = np.abs(edge_attr_np)`

**Impact**: Edge attributes loaded into memory are already unsigned.

### Bug #2: GCN Forward Pass (REDUNDANT BUT STILL WRONG)
**File**: `utils/probing_model.py`  
- Line 58: `ew = torch.abs(ew)`

**Impact**: Even if Bug #1 is fixed, Bug #2 would still destroy the signs.

---

## Why Your Custom GCN Has torch.abs()

Looking at your code, it appears you added `torch.abs()` because:

1. **Good intention**: Handle potentially negative weights robustly
2. **Misunderstanding**: Thought negative weights were "errors" to be cleaned
3. **Defensive coding**: Added NaN/Inf sanitization + abs to be "safe"

However, **negative correlations are NOT errors** - they are critical signal!

---

## Mathematical Impact

### Reference GCN (Correct)
```python
# Node gets messages from neighbors weighted by SIGNED correlations
out[i] = W @ mean([
    +0.8 * x[j1],  # Positively correlated neighbor
    -0.6 * x[j2],  # Negatively correlated neighbor (opposition)
    +0.3 * x[j3]   # Weakly positively correlated
])
```

The GCN learns:
- Positive weights → neurons activate together
- Negative weights → neurons activate oppositely
- Different patterns for truthful vs hallucinated

### Your Custom GCN (Broken)
```python
# Node gets messages from neighbors weighted by UNSIGNED magnitudes
out[i] = W @ mean([
    +0.8 * x[j1],  # Positive correlation (correct)
    +0.6 * x[j2],  # WRONG: Was -0.6, now +0.6 (looks like positive!)
    +0.3 * x[j3]   # Positive correlation (correct)
])
```

The GCN sees:
- All weights are positive magnitudes
- Cannot distinguish "fire together" from "fire oppositely"
- Topological signature destroyed → accuracy ~50%

---

## Additional Difference: Normalization

### Reference: Sum Aggregation (Unnormalized)
```python
out[i] = Σ_{j ∈ N(i)} edge_weight[i,j] * x[j]
```

### Your Implementation: Mean Aggregation (Normalized)
```python
out[i] = (1/deg[i]) * Σ_{j ∈ N(i)} edge_weight[i,j] * x[j]
```

**Impact**: 
- **Moderate**: Mean aggregation is actually reasonable (controls magnitude)
- **Not the main issue**: The `torch.abs()` is far more destructive
- Reference uses unnormalized sum, which can have larger magnitudes

---

## The Fix (Two-Part)

### Part 1: Fix Data Loading (CRITICAL)
**File**: `hallucination/dataset.py`

Remove absolute value calls:
```python
# Line 67: BEFORE
adj = np.abs(adj)

# Line 67: AFTER  
# adj = np.abs(adj)  # REMOVED - keep signed correlations!

# Line 81: BEFORE
edge_attr_np = np.abs(edge_attr_np)

# Line 81: AFTER
# edge_attr_np = np.abs(edge_attr_np)  # REMOVED - keep signed correlations!
```

### Part 2: Fix GCN Forward (CRITICAL)
**File**: `utils/probing_model.py`

Remove absolute value from edge weight processing:
```python
# Line 58: BEFORE
ew = torch.abs(ew)

# Line 58: AFTER
# ew = torch.abs(ew)  # REMOVED - keep signed correlations!
# OR just: ew = ew  # Keep as-is
```

### Optional: Match Reference Normalization (OPTIONAL)
If you want to exactly match the reference, you could also remove mean normalization:

```python
# Lines 62-66: BEFORE
deg = torch.zeros((N, 1), device=x.device, dtype=x.dtype)
ones = torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype)
deg.index_add_(0, dst, ones)
out = out / deg.clamp(min=1)

# AFTER (to match reference unnormalized sum)
# out = out  # No normalization - keep sum aggregation
```

**Note**: This is optional. Mean aggregation is not wrong, just different. The `torch.abs()` is the critical bug.

---

## Why Custom GCN Was Needed

You implemented custom GCN because:
1. **ROCm compatibility**: `torch_scatter` may not compile easily on AMD GPUs
2. **Pure PyTorch**: More portable, no C++ dependencies
3. **Defensive**: Added NaN/Inf handling everywhere

**The custom implementation is fine** - just remove the `torch.abs()` call!

---

## Expected Results After Fix

**Before fix** (current):
- Accuracy: 50-59% (near random)
- Layers 8-10: 46-49% (below chance)
- GCN cannot learn signed correlation patterns

**After fix** (expected):
- Accuracy: 75-85% (matching reference paper Figure 5b)
- All layers should improve significantly
- GCN can now distinguish positive/negative correlations

---

## Validation Test

To verify the fix works, you can add a debug print:

```python
# In SimpleGCNConv.forward(), after loading edge_weight
if edge_weight is not None and training:
    print(f"Edge weight stats: min={edge_weight.min():.3f}, max={edge_weight.max():.3f}, "
          f"num_negative={(edge_weight < 0).sum()}/{edge_weight.numel()}")
```

**Before fix**: `num_negative=0` (all positive due to abs)  
**After fix**: `num_negative>0` (should be ~50% negative from correlations)

---

## Summary

| Component | Issue | Status | Fix Required |
|-----------|-------|--------|--------------|
| **Data loading abs()** | Line 67, 81 in dataset.py | ❌ BROKEN | Remove `np.abs()` |
| **GCN forward abs()** | Line 58 in probing_model.py | ❌ BROKEN | Remove `torch.abs()` |
| **Mean vs sum aggregation** | Lines 62-66 in probing_model.py | ⚠️ Different but OK | Optional to match |
| **Linear before/after** | Architectural difference | ✅ OK | No change needed |
| **NaN handling** | Throughout custom GCN | ✅ GOOD | Keep as-is |

**Both abs() calls must be removed** to restore the 30% lost accuracy.
