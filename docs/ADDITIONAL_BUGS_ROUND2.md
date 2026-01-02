# ADDITIONAL CRITICAL BUGS FOUND (Round 2)

## Bug #1: 🔴 CRITICAL - ConstantBaseline Return Type Mismatch REVERTED

**Location**: [hallucination/eval.py](hallucination/eval.py) lines 42-47

**Issue**: I previously changed ConstantBaseline to return predictions:
```python
# MY CHANGE (WRONG!)
return logits.argmax(dim=-1)  
```

**Reference (CORRECT)**:
```python
return logits  # Returns [batch_size, 2] logits
```

**Problem**: 
- I changed it based on assumption that test_fn expects predictions
- But test_fn calls `.argmax()` on output ITSELF (line 96, 104 in utils.py)
- My change breaks everything by double-applying argmax
- ConstantBaseline returns shape `[batch_size]` but test_fn expects `[batch_size, 2]`
- Will crash with shape mismatch when calling argmax on 1D tensor

**Status**: ❌ **MUST REVERT THIS CHANGE**

---

## Bug #2: 🔴 CRITICAL - eval.py Uses Wrong Model Save Path Format

**Location**: [hallucination/eval.py](hallucination/eval.py) lines 132-133

**Issue**:
```python
density_tag = f"{int(round(FLAGS.density * 100)):02d}"
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

**Reference (train.py)** - uses direct density value in model save path:
```python
density_tag = f"{int(round(FLAGS.density * 100)):02d}"
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

**Wait - they're the SAME!** Let me check what reference does...

**Reference eval.py**:
```python
model_save_path = os.path.join(
    f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
    f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
)
```

**THE BUG**: eval.py uses `density_tag` (formatted as `05`) but reference uses `FLAGS.density` (float `0.05`)

- **train.py saves**: `best_model_density-05_dim-32_hop-3_input-activation.pth`
- **eval.py loads**: `best_model_density-05_dim-32_hop-3_input-activation.pth` ✓ MATCHES (both use density_tag)
- **reference saves**: `best_model_density-0.05_dim-32_hop-3_input-activation.pth`
- **reference loads**: `best_model_density-0.05_dim-32_hop-3_input-activation.pth` ✓ MATCHES

**BUT WAIT** - Check train.py line 54:

train.py HAS:
```python
density_tag = f"{int(round(FLAGS.density * 100)):02d}"
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

**ACTUAL BUG**: train.py uses `density_tag` format BUT this is WRONG!

Should use `FLAGS.density` like reference does:
```python
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

**Impact**: Saved models have wrong filenames. If you run with `density=0.05`:
- Saves to: `best_model_density-05_dim-32_...pth`
- Should save to: `best_model_density-0.05_dim-32_...pth`
- Can't resume training, model paths inconsistent

---

## Bug #3: 🔴 CRITICAL - eval.py Missing scheduler Argument

**Location**: [hallucination/eval.py](hallucination/eval.py) - entire eval pipeline

**Issue**: eval.py creates model but doesn't create scheduler for warm-up or learning rate scheduling

**Problem**: Unlike train.py which uses scheduler in train_model(), eval.py doesn't need it for evaluation - but this is INCONSISTENT with train.py expectations

**Status**: Not directly critical for eval, but architecture inconsistency

---

## Bug #4: 🟠 IMPORTANT - eval.py Path Type Inconsistency

**Current eval.py (Line 133)**:
```python
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}..."
```

**Reference eval.py**:
```python
model_save_path = os.path.join(
    f"saves/{save_model_name}/layer_{FLAGS.llm_layer}",
    f"best_model_density-{FLAGS.density}..."
)
```

**Issue**:
- Current uses pathlib (`/` operator with `main_dir`)
- Reference uses os.path.join (string-based)
- Both work but inconsistent with rest of eval.py structure

---

## Summary of New Bugs Found

| Bug | Severity | File | Line(s) | Issue | Status |
|-----|----------|------|---------|-------|--------|
| #1 | 🔴 CRITICAL | eval.py | 47 | ConstantBaseline return argmax (WRONG) | ❌ NEED TO REVERT |
| #2 | 🔴 CRITICAL | train.py | 54 | density_tag format instead of float | ❌ NEED TO FIX |
| #3 | 🔴 CRITICAL | eval.py | 132 | density_tag format instead of float | ❌ NEED TO FIX |
| #4 | 🟠 IMPORTANT | eval.py | 133 | Path type inconsistency | ⚠️ MINOR |

---

## Required Fixes

### Fix #1: REVERT ConstantBaseline return to logits

```python
# eval.py lines 42-47
# CURRENT (WRONG)
def forward(self, x, *args):
    batch_size = x.shape[0]
    logits = torch.zeros(batch_size, 2, device=x.device)
    logits[:, self.label] = 1.0
    return logits.argmax(dim=-1)  # ❌ WRONG!

# SHOULD BE
def forward(self, x, *args):
    batch_size = x.shape[0]
    logits = torch.zeros(batch_size, 2, device=x.device)
    logits[:, self.label] = 1.0
    return logits  # ✓ Return logits, let test_fn handle argmax
```

### Fix #2: Change train.py save path format

```python
# train.py line 54
# CURRENT (WRONG)
density_tag = f"{int(round(FLAGS.density * 100)):02d}"
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"

# SHOULD BE
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

### Fix #3: Change eval.py load path format

```python
# eval.py lines 132-133
# CURRENT (WRONG)
density_tag = f"{int(round(FLAGS.density * 100)):02d}"
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"

# SHOULD BE
model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{FLAGS.density}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
```

---

## Why I Missed These Earlier

1. **ConstantBaseline bug**: I made an assumption about what test_fn expected without fully verifying against reference
2. **density_tag bugs**: Didn't carefully compare train.py vs reference eval.py model save path format
3. **Path inconsistency**: Didn't check if train.py also had same issue

These are subtle but critical for model persistence and evaluation!
