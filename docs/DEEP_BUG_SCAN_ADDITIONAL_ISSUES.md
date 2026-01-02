# Deep Bug Scan Results - Additional Critical Issues Found

## Overview
Found **7 additional bugs** beyond the correlation sign destruction issue.

---

## BUG #1: 🔴 CRITICAL - Model Name Sanitization in train.py (STILL PRESENT)

**File**: [hallucination/train.py](hallucination/train.py)
**Lines**: 200-205

**Issue**:
```python
sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if FLAGS.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"
```

**Problem**: 
- Still sanitizing model names in training
- But dataset loading **no longer sanitizes** (we removed it)
- **MISMATCH**: Data dir is `gpt2-large`, but saves to `gpt2_large`
- Can't load best model to resume training!

**Reference (correct)**:
```python
if FLAGS.ckpt_step == -1:
    model_dir = FLAGS.llm_model_name
else:
    model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"
```

**Impact**: 
- Resume training fails (line 229-231 can't find saved models)
- Saved models go to wrong directory

---

## BUG #2: 🔴 CRITICAL - Model Name Sanitization in eval.py (STILL PRESENT)

**File**: [hallucination/eval.py](hallucination/eval.py)
**Lines**: 62-65

**Issue**:
```python
sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if FLAGS.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
```

**Problem**: 
- Evaluation looks for saved models in wrong directory
- Data loaded from `gpt2-large` directory
- But tries to load checkpoints from `gpt2_large` directory
- **FileNotFoundError** on eval!

**Impact**: Cannot evaluate saved models

---

## BUG #3: 🔴 CRITICAL - Device Selection in train.py vs eval.py

**File 1**: [hallucination/train.py](hallucination/train.py)
**Line**: 196
```python
device = select_device(FLAGS.gpu_id)  # ✅ GOOD: Safe device selection
```

**File 2**: [hallucination/eval.py](hallucination/eval.py)
**Line**: 56
```python
device = torch.device(f"cuda:{FLAGS.gpu_id}")  # ❌ BAD: Unsafe!
```

**Reference (train.py)**:
```python
device = torch.device(f"cuda:{FLAGS.gpu_id}")  # Uses simple device creation
```

**Problem**:
- train.py uses `select_device()` with GPU sanity checks
- eval.py uses raw `torch.device()` with no error handling
- If GPU unavailable, eval.py crashes with cryptic error
- Inconsistent error handling

**Impact**: eval.py will fail silently with confusing CUDA errors

---

## BUG #4: 🟠 IMPORTANT - Missing Warmup Scheduler in train.py

**File**: [hallucination/train.py](hallucination/train.py)
**Line**: 46 (function signature)
```python
def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device):
```

**Line**: 130 (usage)
```python
if epoch < FLAGS.warmup_epochs:
    warmup_scheduler.step()
```

**Problem**: 
- Function expects `warmup_scheduler` parameter
- But in main(), it's **never created**!

**Reference (train.py main())**:
```python
# Line ~250 in current - doesn't create warmup_scheduler
optimizer = torch.optim.Adam(...)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)
writer = SummaryWriter(...)
train_model(model, train_loader, test_loader, optimizer, scheduler, writer, ...)
#                                                                ❌ Missing warmup_scheduler!
```

**Impact**: 
- Runtime error: `NameError: warmup_scheduler is not defined`
- Warmup learning rate schedule never applied
- Training fails immediately

---

## BUG #5: 🟠 IMPORTANT - Wrong train_model() Signature

**Current (broken)**:
```python
def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device):
```

**Reference (correct)**:
```python
def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
```

**Problem**:
- Current train.py defines function with `warmup_scheduler` parameter
- But main() doesn't pass it
- Parameter never created

**Fix**:
Either:
1. Remove `warmup_scheduler` from function and usage
2. Create warmup_scheduler in main() and pass it

---

## BUG #6: 🟠 IMPORTANT - Extra seldom function signature (utilities in eval.py)

**File**: [hallucination/eval.py](hallucination/eval.py)
**Lines**: 37-45

```python
class ConstantBaseline(torch.nn.Module):
    def __init__(self, label):
        super().__init__()
        if label not in (0, 1):
            raise ValueError("baseline_label must be 0 or 1.")
        self.label = label

    def forward(self, x, *args):
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 2, device=x.device)
        logits[:, self.label] = 1.0
        return logits
```

**Problem**: This class expects 2D output `logits` with shape `[batch_size, 2]`, but:
- test_fn() expects output shape `[batch_size]` (after argmax)
- ConstantBaseline returns `[batch_size, 2]` logits
- This causes shape mismatch in test_fn()

**Impact**: Baseline evaluation will crash

---

## BUG #7: 🟡 MEDIUM - Unused select_device import

**File**: [hallucination/eval.py](hallucination/eval.py)
**Line**: 7

```python
from hallucination.utils import test_fn, select_device
```

**Problem**:
- Imports `select_device` but doesn't use it
- Uses raw `torch.device()` instead
- Dead code

**Impact**: Low - just unused import

---

## BUG #8: 🟡 MEDIUM - Inconsistent utils between train and eval

**File 1**: [hallucination/train.py](hallucination/train.py)
**Line**: 11
```python
from hallucination.utils import test_fn, select_device
```

**File 2**: [hallucination/eval.py](hallucination/eval.py)
**Line**: 7
```python
from hallucination.utils import test_fn, select_device
```

**Problem**: 
- Reference doesn't have `select_device()` in utils.py
- Current code added it to utils.py
- Creates dependency that reference doesn't have
- Makes reference comparison harder

---

## Summary Table

| Bug # | Severity | File | Issue | Status |
|-------|----------|------|-------|--------|
| 1 | 🔴 CRITICAL | train.py | Model name still sanitized | ❌ NOT FIXED |
| 2 | 🔴 CRITICAL | eval.py | Model name still sanitized | ❌ NOT FIXED |
| 3 | 🔴 CRITICAL | eval.py | Wrong device initialization | ❌ NOT FIXED |
| 4 | 🟠 IMPORTANT | train.py | Missing warmup_scheduler | ❌ NOT FIXED |
| 5 | 🟠 IMPORTANT | train.py | Wrong function signature | ❌ NOT FIXED |
| 6 | 🟠 IMPORTANT | eval.py | ConstantBaseline shape mismatch | ❌ NOT FIXED |
| 7 | 🟡 MEDIUM | eval.py | Unused import | ❌ NOT FIXED |
| 8 | 🟡 MEDIUM | Both | Added select_device to utils | ❌ NOT FIXED |

---

## Required Fixes

### Fix 1: Remove sanitization from train.py (lines 200-205)
```python
# BEFORE
sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if FLAGS.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"

# AFTER
if FLAGS.ckpt_step == -1:
    model_dir = FLAGS.llm_model_name
else:
    model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
```

### Fix 2: Remove sanitization from eval.py (lines 62-65)
```python
# BEFORE
sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if FLAGS.ckpt_step == -1:
    model_dir = sanitized_model_name
else:
    model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"

# AFTER
if FLAGS.ckpt_step == -1:
    model_dir = FLAGS.llm_model_name
else:
    model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
```

### Fix 3: Fix device initialization in eval.py (line 56)
```python
# BEFORE
device = torch.device(f"cuda:{FLAGS.gpu_id}")

# AFTER
device = select_device(FLAGS.gpu_id)
```

### Fix 4: Remove warmup_scheduler from train.py
**Line 46**: Remove parameter from function signature
```python
# BEFORE
def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device):

# AFTER
def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
```

**Line 130**: Remove warmup step
```python
# BEFORE
if epoch < FLAGS.warmup_epochs:
    warmup_scheduler.step()

# AFTER
# Remove or comment out this line
```

### Fix 5: Update call to train_model()
**Find where train_model is called**:
```python
# BEFORE
train_model(model, train_loader, test_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device)

# AFTER
train_model(model, train_loader, test_loader, optimizer, scheduler, writer, save_model_name, device)
```

### Fix 6: Fix ConstantBaseline forward method in eval.py
```python
# BEFORE
def forward(self, x, *args):
    batch_size = x.shape[0]
    logits = torch.zeros(batch_size, 2, device=x.device)
    logits[:, self.label] = 1.0
    return logits

# AFTER
def forward(self, x, *args):
    batch_size = x.shape[0]
    logits = torch.zeros(batch_size, 2, device=x.device)
    logits[:, self.label] = 1.0
    return logits.argmax(dim=-1)  # Return predictions not logits
```

---

## Root Cause Analysis

**Why these bugs appeared**:
1. Code evolved independently from reference
2. Added `select_device()` function for better error handling
3. Added warmup learning rate schedule (then forgot to use it)
4. Inconsistently applied model name sanitization fixes
5. eval.py not updated to match train.py conventions

**Why they weren't caught**:
- Not enough testing on full pipeline
- eval.py not run after recent changes
- Function signatures changed without updating all call sites
