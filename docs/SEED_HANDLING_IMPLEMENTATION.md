# Seed Handling Implementation for Reproducibility

## Problem Statement

When `seed: null` is specified in the config file, each layer's training could potentially get a **different random train/test split**, making cross-layer comparisons invalid and results non-reproducible.

## Solution Implemented

### 1. **Auto-Generate Seed at Pipeline Start** (Line 315-329)

```python
# If seed is None/null in config, generate a random seed once at the beginning
# This ensures ALL layers use the SAME train/test split for fair comparison
seed = hallu_cfg.get('seed', None)  # None means random (no seed)
if seed is not None:
    seed = int(seed)
else:
    # Generate a random seed based on current time
    seed = int(time.time() * 1000000) % (2**32)  # 32-bit seed for PyTorch compatibility
```

**Key Points:**
- Seed is generated **once** at the beginning of the pipeline
- Uses current time in microseconds for uniqueness
- Modulo 2^32 ensures PyTorch compatibility (32-bit seeds)
- All subsequent steps use this **same seed value**

### 2. **Log and Save Seed** (Line 458-492)

```python
# Report the seed being used
seed_source = "from config file" if hallu_cfg.get('seed', None) is not None else "AUTO-GENERATED"
logging.info(f"Random Seed: {seed} ({seed_source})")
logging.info(f"  This seed will be used for ALL layers to ensure identical train/test splits")

# Save seed to file
seed_file = reports_root / "random_seed.txt"
with open(seed_file, 'w') as f:
    f.write(f"seed={seed}\n")
    f.write(f"# To reproduce this exact analysis, add this to your config YAML:\n")
    f.write(f"# hallucination:\n")
    f.write(f"#   seed: {seed}\n")
```

**Output File:** `reports/hallucination_analysis/{model_tag}/random_seed.txt`

**Example Content:**
```
# Random Seed Configuration
# Generated: 2026-01-02 14:30:45
# Source: AUTO-GENERATED
# Model: gpt2
# Dataset: truthfulqa
# Layers: [5, 6, 7, 8, 9, 10, 11]

seed=1735831845234567

# To reproduce this exact analysis, add this to your config YAML:
# hallucination:
#   seed: 1735831845234567
```

### 3. **Pass Seed to All Training Steps** (Line 1048)

```python
result = run([
    python_exe, "-m", "hallucination.train",
    f"--seed={seed}",  # Always pass seed (auto-generated if not in config)
    ...
])
```

**Changed from:** `] + ([f"--seed={seed}"] if seed is not None else []) + ...`
**Changed to:** `f"--seed={seed}",` (always passed)

## Verification Flow

### Data Flow with Seed

```
Config (seed: null)
    ↓
Pipeline Start: Generate seed=1735831845234567 (auto-generated)
    ↓
Save to: random_seed.txt
    ↓
Layer 5: train.py --seed=1735831845234567
    ↓ Uses prepare_data(seed=1735831845234567)
    ↓ Creates train indices [0-4731], test indices [4732-5914]
    ↓
Layer 6: train.py --seed=1735831845234567
    ↓ Uses prepare_data(seed=1735831845234567)
    ↓ Creates train indices [0-4731], test indices [4732-5914]  ← IDENTICAL
    ↓
Layer 7: train.py --seed=1735831845234567
    ↓ Uses prepare_data(seed=1735831845234567)
    ↓ Creates train indices [0-4731], test indices [4732-5914]  ← IDENTICAL
    ↓
...all layers use the SAME train/test split
```

### With Fixed Seed (seed: 42)

```
Config (seed: 42)
    ↓
Pipeline Start: Use seed=42 (from config file)
    ↓
Save to: random_seed.txt (Source: from config file)
    ↓
Layer 5: train.py --seed=42
Layer 6: train.py --seed=42
Layer 7: train.py --seed=42
...all layers use the SAME train/test split
```

## Benefits

### ✅ Reproducibility
- Every run saves the exact seed used
- Can reproduce exact results by copying seed to config

### ✅ Cross-Layer Comparability
- All layers use **identical train/test splits**
- Differences in performance come from layer properties, not data differences
- Fair statistical comparison across layers

### ✅ Transparency
- Seed source clearly logged (config vs auto-generated)
- Seed value prominently displayed in console output
- Saved to file for future reference

### ✅ Flexibility
- `seed: null` → auto-generate random seed (new split each run)
- `seed: 42` → use fixed seed (reproducible split across runs)
- Best of both worlds: random when exploring, fixed when reproducing

## Testing

### Test Case 1: Auto-Generated Seed
```yaml
# config.yaml
hallucination:
  seed: null
```

**Expected Behavior:**
1. Pipeline generates unique seed (e.g., 1735831845234567)
2. Logs: "Random Seed: 1735831845234567 (AUTO-GENERATED)"
3. Saves to: `random_seed.txt`
4. All layers receive: `--seed=1735831845234567`
5. All layers get identical train/test split

### Test Case 2: Fixed Seed
```yaml
# config.yaml
hallucination:
  seed: 42
```

**Expected Behavior:**
1. Pipeline uses seed=42
2. Logs: "Random Seed: 42 (from config file)"
3. Saves to: `random_seed.txt`
4. All layers receive: `--seed=42`
5. All layers get identical train/test split (reproducible across runs)

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Seed generation | `my_analysis/hallucination_detection_analysis.py` | 315-329 |
| Seed logging | `my_analysis/hallucination_detection_analysis.py` | 458-492 |
| Seed passing | `my_analysis/hallucination_detection_analysis.py` | 1048 |
| Seed usage | `hallucination/dataset.py` | 18-27 |
| Train/test split | `hallucination/dataset.py` | 25 |

## Implementation Date
January 2, 2026
