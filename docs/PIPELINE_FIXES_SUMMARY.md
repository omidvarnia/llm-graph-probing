# Pipeline Fixes Summary

## Date: December 30, 2025

## Issues Identified

### 1. **Only Layer 5 Was Being Processed**
   - **Problem**: The analysis pipeline only trained and evaluated probes for the first layer in `layer_list` (layer 5), ignoring layers 6-11
   - **Root Cause**: Steps 3 and 4 used `layer_id = layer_ids[0]` instead of looping through all layers
   - **Impact**: Missing evaluation metrics for 6 out of 7 layers

### 2. **Early Stopping Triggered Prematurely**
   - **Problem**: Training stopped at epoch 21 despite `num_epochs=200` and `early_stop_patience=200`
   - **Root Cause**: Early stopping counter (`epochs_no_improve`) was incremented during the warmup period (first 5 epochs)
   - **Impact**: Training terminated after only 21 epochs instead of running for up to 200 epochs

### 3. **No Comprehensive Metrics Summary**
   - **Problem**: Classification metrics (accuracy, precision, recall, F1, confusion matrix) were not reported in a consolidated summary
   - **Root Cause**: Pipeline only saved logs individually, no aggregated metrics table
   - **Impact**: Difficult to compare performance across layers

## Fixes Applied

### Fix 1: Process All Layers in Steps 3 & 4

**File**: `my_analysis/hallucination_detection_analysis.py`

**Changes**:
- Modified Step 3 (Training) to loop through all `layer_ids`:
  ```python
  for lid in layer_ids:
      # Train probe for each layer
  ```
- Modified Step 4 (Evaluation) to loop through all `layer_ids`:
  ```python
  for lid in layer_ids:
      # Evaluate probe for each layer
  ```
- Updated all references from `layer_id` to `lid` within the loops
- Added `"layer": lid` to all step results for proper tracking

**Result**: All layers (5-11) will now be trained and evaluated independently

### Fix 2: Correct Early Stopping Logic

**File**: `hallucination/train.py`

**Changes**:
- Modified early stopping counter to only increment **after warmup period**:
  ```python
  else:
      # Only count epochs_no_improve after warmup period
      if epoch >= FLAGS.warmup_epochs:
          epochs_no_improve += 1
          if epochs_no_improve >= FLAGS.early_stop_patience:
              logging.info(f"Early stopping at epoch {epoch + 1} (no improvement for {epochs_no_improve} epochs)")
              break
  ```

**Before**: Counter incremented from epoch 1, causing early stop at epoch 21 (5 warmup + 16 no-improve = 21)

**After**: Counter starts at epoch 6 (after warmup), allowing full 200 epochs of patience

### Fix 3: Add Classification Metrics Summary

**File**: `my_analysis/hallucination_detection_analysis.py`

**Changes**:
1. Extract confusion matrix from evaluation logs using regex
2. Calculate accuracy, precision, recall, F1 from confusion matrix
3. Log comprehensive metrics table:
   ```
   Layer    Accuracy   Precision  Recall     F1 Score   Above Chance
   ------------------------------------------------------------------------
   5        0.4869     0.4569     0.4249     0.4403     ✗ NO
   6        ...        ...        ...        ...        ...
   ```
4. Save metrics to CSV file: `classification_metrics_summary.csv`
5. Include confusion matrix breakdown (TN, FP, FN, TP) for each layer

**Result**: Easy-to-read summary showing which layers perform above chance (>50% accuracy)

## Expected Behavior After Fixes

### Training Duration
- **Before**: ~21 epochs per layer (~12 minutes total)
- **After**: Up to 200 epochs per layer or early stop after 200 epochs of no F1 improvement (~2-3 hours per layer)

### Layers Processed
- **Before**: Only layer 5
- **After**: All layers 5, 6, 7, 8, 9, 10, 11

### Output Files Structure
```
results/hallucination_analysis/Qwen_Qwen2_5_0_5B/
├── classification_metrics_summary.csv    # NEW: Consolidated metrics table
├── summary.json                          # Updated with all layers
├── layer_5/
│   ├── step3_train.log
│   ├── step4_eval.log
│   ├── train_loss.png
│   └── test_metrics.png
├── layer_6/
│   ├── step3_train.log                   # NEW
│   ├── step4_eval.log                    # NEW
│   └── ...
├── layer_7/
│   └── ...
... (layers 8-11)
```

### Metrics Summary Format (CSV)
```csv
layer,accuracy,precision,recall,f1_score,above_chance,tn,fp,fn,tp
5,0.4869,0.4569,0.4249,0.4403,False,194,454,153,382
6,...,...,...,...,...,...,...,...,...
...
```

## Verification Steps

After re-running the pipeline:

1. **Check all layers were processed**:
   ```bash
   find results/hallucination_analysis/Qwen_Qwen2_5_0_5B -name "step4_eval.log" | wc -l
   # Should return 7 (layers 5-11)
   ```

2. **Check training duration**:
   ```bash
   grep "Best Epoch" results/hallucination_analysis/Qwen_Qwen2_5_0_5B/layer_*/step3_train.log
   # Should show epochs > 21 for most layers
   ```

3. **Check metrics summary exists**:
   ```bash
   cat results/hallucination_analysis/Qwen_Qwen2_5_0_5B/classification_metrics_summary.csv
   # Should show 7 rows (one per layer)
   ```

4. **Check which layers are above chance**:
   ```bash
   grep "Above chance" results/hallucination_analysis/Qwen_Qwen2_5_0_5B/classification_metrics_summary.csv
   # Should show True/False for each layer
   ```

## Configuration Used

From `config_files/pipeline_config_qwen.yaml`:
- `num_epochs: 200`
- `early_stop_patience: 200`
- `warmup_epochs: 5`
- `layer_list: "5,6,7,8,9,10,11"`

## Notes

- The early stopping issue was subtle: it counted epochs from the start, including warmup epochs where the model hasn't stabilized yet
- With warmup=5 and patience=200, the old code would stop at epoch 205 at earliest, but the counter was being incremented during warmup, causing premature termination
- The fix ensures early stopping only activates **after** the warmup period, giving the model a fair chance to improve

## Next Steps

1. Re-run the analysis pipeline with these fixes:
   ```bash
   sbatch run_hallu_detec_mpcdf.slurm
   ```

2. Monitor progress:
   ```bash
   tail -f /ptmp/aomidvarnia/analysis_results/llm_graph/slurm_logs/<job_id>.out
   ```

3. After completion, review `classification_metrics_summary.csv` to identify which layers detect hallucinations above chance

4. Expected runtime: ~14-21 hours (2-3 hours per layer × 7 layers)
