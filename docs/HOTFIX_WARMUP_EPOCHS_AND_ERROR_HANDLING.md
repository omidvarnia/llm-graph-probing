# HOTFIX: warmup_epochs Removal and Fatal Error Handling

## Issue Description

The pipeline was encountering a fatal error during execution:
```
FATAL Flags parsing error: Unknown command line flag 'warmup_epochs'. Did you mean: num_epochs ?
```

This error repeated 7 times, indicating multiple parallel processes were encountering the same issue. The problem was that `warmup_epochs` was still being passed as a command-line argument to training scripts, but the flag had been removed from the main `train.py` module.

## Root Cause Analysis

The `warmup_epochs` parameter was:
1. ✅ Removed from main training script (`hallucination/train.py`) - correct
2. ❌ BUT still being passed in the orchestrator script (`hallucination_detection_analysis.py`)
3. ❌ Still defined in alternative training script (`hallucination/train_activation_probe.py`)
4. ❌ Still referenced in parameter mapping (`utils/load_params.py`)

This created a mismatch where the orchestrator tried to pass a flag that no longer existed.

## Changes Made

### 1. Removed warmup_epochs from Python Files

#### `my_analysis/hallucination_detection_analysis.py`
- **Line 272**: Removed `warmup_epochs = int(hallu_cfg.get('warmup_epochs', 5))`
- **Line 683**: Removed logging of warmup_epochs
- **Line 730**: Removed `f"--warmup_epochs={warmup_epochs}"` from command args

#### `hallucination/train_activation_probe.py`
- **Line 47**: Removed flag definition `flags.DEFINE_integer("warmup_epochs", 5, ...)`
- **Line 129**: Removed warmup check `if epoch < FLAGS.warmup_epochs`
- **Line 159**: Removed warmup condition before scheduler
- **Line 171**: Removed warmup condition before early stopping  
- **Lines 246-247**: Removed LinearLR warmup scheduler definition

#### `utils/load_params.py`
- **Line 72**: Removed `'warmup_epochs': 'WARMUP_EPOCHS'` from parameter mapping

### 2. Added Comprehensive Fatal Error Detection and Handling

#### Enhanced `run()` Function in `hallucination_detection_analysis.py`

**Before**: Subprocess output was just printed/logged without checking for fatal errors

**After**: 
```python
# Detect FATAL errors - these should immediately stop the pipeline
if "FATAL" in line:
    fatal_error_lines.append(line.rstrip())
    print(f"🔴 FATAL ERROR DETECTED: {line.rstrip()}")
    if log_handle:
        log_handle.write(f"\n🔴 FATAL ERROR DETECTED: {line}\n")

# ... at end of function ...

# If fatal errors were detected, raise exception immediately
if fatal_error_lines:
    error_msg = "\n".join(fatal_error_lines[:5])
    raise RuntimeError(f"Pipeline stopped due to fatal error:\n{error_msg}")
```

#### Added try-except Blocks in All Pipeline Steps

Each step now catches fatal errors and immediately exits:

**Step 1 (Dataset Construction)**:
```python
try:
    result = run([...], cwd=project_dir, env=env, log_file=step1_log)
except RuntimeError as e:
    logging.error(f"✗ FATAL ERROR in Step 1: {e}")
    logging.error("Pipeline stopping due to fatal error")
    sys.exit(1)
```

**Step 2 (Neural Network Computation)**:
```python
try:
    result = run([...], cwd=project_dir, env=env, log_file=step2_log)
except RuntimeError as e:
    logging.error(f"✗ FATAL ERROR in Step 2: {e}")
    logging.error("Pipeline stopping due to fatal error")
    sys.exit(1)
```

**Step 3 (Probe Training)** - per layer:
```python
try:
    result = run([...], cwd=project_dir, env=env, log_file=step3_log)
except RuntimeError as e:
    logging.error(f"✗ FATAL ERROR in Step 3 (Layer {lid}): {e}")
    logging.error("Pipeline stopping due to fatal error")
    sys.exit(1)
```

**Step 4 (Probe Evaluation)** - per layer:
```python
try:
    result = run([...], cwd=project_dir, env=env, log_file=step4_log)
except RuntimeError as e:
    logging.error(f"✗ FATAL ERROR in Step 4 (Layer {lid}): {e}")
    logging.error("Pipeline stopping due to fatal error")
    sys.exit(1)
```

**Step 5 (Graph Analysis)** - per layer:
```python
try:
    result = run([...], cwd=project_dir, env=env, log_file=step5_log)
except RuntimeError as e:
    logging.error(f"✗ FATAL ERROR in Step 5 (Layer {lid}): {e}")
    logging.error("Pipeline stopping due to fatal error")
    sys.exit(1)
```

## Verification Checklist

- [x] ✅ `warmup_epochs` removed from `hallucination_detection_analysis.py`
- [x] ✅ `warmup_epochs` removed from `train_activation_probe.py`
- [x] ✅ `warmup_epochs` removed from `utils/load_params.py`
- [x] ✅ No references to `warmup_epochs` in `train.py`
- [x] ✅ No references to `warmup_epochs` in `eval.py`
- [x] ✅ Fatal error detection implemented in `run()` function
- [x] ✅ Error handling added to all 5 pipeline steps
- [x] ✅ Each step calls `sys.exit(1)` on fatal error (no continue)
- [x] ✅ Clear error messages logged for each fatal error

## Files Modified

1. **my_analysis/hallucination_detection_analysis.py**
   - Removed warmup_epochs configuration and CLI args
   - Enhanced run() function with FATAL error detection
   - Added try-except blocks to all 5 pipeline steps
   - Status: ✅ Complete

2. **hallucination/train_activation_probe.py**
   - Removed warmup_epochs flag definition
   - Removed all warmup_epochs checks and scheduler
   - Status: ✅ Complete

3. **utils/load_params.py**
   - Removed warmup_epochs from parameter mapping
   - Status: ✅ Complete

## Expected Behavior After Fix

### Before Fix:
```
FATAL Flags parsing error: Unknown command line flag 'warmup_epochs'. Did you mean: num_epochs ?
(repeated 7 times)
```
Pipeline continues after error (wrong behavior)

### After Fix:
1. ✅ No more `warmup_epochs` errors
2. ✅ Pipeline proceeds normally
3. ✅ If any FATAL error occurs (flag parsing, out of memory, etc.), it is:
   - Immediately detected
   - Clearly highlighted with 🔴 marker in logs
   - Reported with full error message
   - Pipeline exits with `sys.exit(1)` (no continue)

## How Error Detection Works

The enhanced `run()` function now:
1. Streams all subprocess output line-by-line
2. **Scans each line for "FATAL" keyword**
3. Collects all fatal error messages
4. After process completes, checks if any fatal errors occurred
5. **If fatal errors found: raises RuntimeError immediately**
6. Orchestrator catches exception and calls `sys.exit(1)`
7. Pipeline stops completely (no partial execution)

## Testing

To verify the fix works:

```bash
# Run the pipeline normally
sbatch run_hallu_detec_mpcdf.slurm

# Expected results:
# 1. No "warmup_epochs" flag errors
# 2. If any FATAL errors occur, pipeline exits cleanly
# 3. Log files show clear error messages with 🔴 markers
```

## Log Output Examples

### Success Case (After Fix):
```
STEP 1: Constructing Dataset
Executing construct_dataset.py...
✓ Dataset constructed successfully
✓ SANITY CHECK: Dataset file exists at ...

STEP 2: Generating Neural Topology (Network Graph)
Executing compute_llm_network.py...
✓ Network computation completed

STEP 3: Training Hallucination Detection Probes
Executing train.py...
✓ Probe trained successfully for layer 5

STEP 4: Evaluating Probes
Executing eval.py...
✓ Evaluation completed

STEP 5: Graph Analysis
Executing graph_analysis.py...
✓ Analysis completed
```

### Fatal Error Case (After Fix):
```
STEP 3: Training Hallucination Detection Probes
Executing train.py...
🔴 FATAL ERROR DETECTED: FATAL Flags parsing error: Unknown command line flag 'xyz'...
✗ FATAL ERROR in Step 3 (Layer 5): Pipeline stopped due to fatal error...
Pipeline stopping due to fatal error
[Pipeline exits with sys.exit(1)]
```

## Rollback Instructions

If needed to rollback:
```bash
git revert <commit-hash>
```

Or manually restore by:
1. Re-add `warmup_epochs` flag to appropriate files
2. Remove try-except blocks from orchestrator
3. Revert run() function to original version

## Impact Assessment

- **Backward Compatibility**: ✅ Full - removed deprecated parameter
- **Performance Impact**: ✅ None - no performance changes
- **User Experience**: ✅ Improved - clearer error messages and proper exit behavior
- **Testing Required**: ✅ Yes - verify pipeline runs successfully

## Summary

This hotfix addresses two critical issues:
1. **Removed deprecated `warmup_epochs` parameter** that was causing pipeline failures
2. **Added comprehensive fatal error detection** to ensure pipeline stops immediately on errors instead of continuing with partial execution

The pipeline will now:
- Execute without `warmup_epochs` errors
- Detect and report all fatal errors clearly
- Exit gracefully and immediately on any fatal error
- Provide clear diagnostic information in logs for troubleshooting
