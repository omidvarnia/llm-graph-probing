# Pipeline Step Selection Guide

## Overview

The hallucination detection pipeline now supports selective step execution. You can run any combination of steps 1-5, and the pipeline will automatically validate that previous step results exist.

## Configuration

In your config file (`pipeline_config_gpt2.yaml`, etc.), add the `steps` parameter:

```yaml
common:
  dataset_name: truthfulqa
  model_name: gpt2
  steps: "1,2,3,4,5"  # Specify which steps to run
  ...
```

## Step Definitions

- **Step 1**: Dataset Construction (Extract LLM activations)
- **Step 2**: Neural Topology Computation (Compute correlation networks)
- **Step 3**: Probe Training (Train hallucination detection models)
- **Step 4**: Probe Evaluation (Evaluate trained models on test set)
- **Step 5**: Graph Analysis (Compute intra vs inter-group correlation metrics)

## Usage Examples

### Run All Steps (Default)
```yaml
steps: "1,2,3,4,5"
```

### Run Only Step 3 (Training) - Reuse Data & Networks
```yaml
steps: "3"
```
Requires:
- Step 1 results: `{MAIN_DIR}/data/hallucination/{dataset_name}.csv`
- Step 2 results: `{MAIN_DIR}/data/hallucination/{dataset_name}/{model_dir}/`

### Run Steps 1-2 Only (Data Preparation)
```yaml
steps: "1,2"
```
Prepares all data without training models.

### Run Steps 3-4 (Train & Evaluate)
```yaml
steps: "3,4"
```
Requires:
- Step 1 results: `{MAIN_DIR}/data/hallucination/{dataset_name}.csv`
- Step 2 results: `{MAIN_DIR}/data/hallucination/{dataset_name}/{model_dir}/`

### Run Steps 1 and 3 (Skip Network Computation)
```yaml
steps: "1,3"
```
Skips Step 2. Useful when you already have networks but want to retrain with different parameters.

### Run Only Step 5 (Analysis Only)
```yaml
steps: "5"
```
Requires:
- Step 1 results
- Step 2 results
- Step 3 results (trained models)

## Validation & Error Handling

The pipeline includes automatic validation:

1. **If `steps` includes any step > 1**: Previous step results are checked
2. **Missing results error**: If previous results don't exist, the pipeline exits with detailed error message
3. **Timestamps**: All logging includes date/time stamps (`YYYY-MM-DD HH:MM:SS`)

### Example Error Message
```
✗ VALIDATION FAILED: Step 2 outputs not found
  Expected network directory: /ptmp/.../data/hallucination/truthfulqa/gpt2/
  Cannot start from step 3 without Step 2 results
  Pipeline stopping due to error
```

## Output Structure

Results are organized by step:

```
{MAIN_DIR}/
├── data/hallucination/              (Steps 1-2: Raw data & networks)
│   └── {dataset_name}/{model_dir}/
├── reports/hallucination_analysis/  (Steps 3-5: Results with probe type)
│   └── {model}_{probe_type}/        (e.g., gpt2_mlp, gpt2_gcn)
│       └── layer_{id}/
│           ├── step3_train.log
│           ├── step4_eval.log
│           └── step5_graph_analysis.log
└── saves/hallucination/             (Step 3: Trained models)
    └── {dataset_name}/{model_dir}/
```

## Logging

All logs now include timestamps in format `YYYY-MM-DD HH:MM:SS`:

```
2026-01-03 14:30:15 hallucination_detection_analysis.py:391 - VALIDATION CHECK: Checking previous step results...
2026-01-03 14:30:16 hallucination_detection_analysis.py:402 - ✓ Step 1 outputs found: /path/to/dataset
2026-01-03 14:30:17 hallucination_detection_analysis.py:1055 - ⊘ SKIPPED Step 2: Neural Topology Computation (not in requested steps)
```

## Common Workflows

### Workflow 1: One-Shot Execution (New Analysis)
```yaml
steps: "1,2,3,4,5"  # Run everything
```

### Workflow 2: Fast Iteration (Model Development)
```
1. First run: steps: "1,2"            # Prepare data (takes ~30-60 min)
2. Training: steps: "3"               # Train with config A (takes ~10-20 min/layer)
3. Evaluation: steps: "4,5"           # Evaluate results (takes ~5 min/layer)
4. Modify model config, repeat step 2-3
```

### Workflow 3: Different Probe Types (GCN vs MLP)
```
1. First run: steps: "1,2,3,4,5" with num_layers: 3 (GCN)
2. Second run: steps: "3,4,5" with num_layers: -2 (MLP)  
   → Reuses same data & networks, only retrains probe
```

### Workflow 4: Analysis on Existing Results
```yaml
steps: "5"  # Run only graph analysis
```
Requires all previous steps completed. Useful for:
- Changing visualization parameters
- Recomputing statistics with different thresholds
- Generating additional plots

## Tips & Best Practices

1. **Always start with Step 1 for new datasets**
   - Ensures data validation and preprocessing

2. **Validate your networks before training**
   - Run steps "1,2" first to check network generation
   - Inspect logs and saved correlation matrices

3. **Use step selection for hyperparameter tuning**
   - Fix steps "1,2", vary config for steps "3,4,5"
   - Saves time on data preparation

4. **Combine with model selection**
   - GCN (num_layers=3): Better if ROCm/CUDA available
   - MLP (num_layers=-2): Fallback or comparison baseline

5. **Check log timestamps for performance**
   ```bash
   # See duration of each step
   grep "COMPLETE\|SKIPPED" output.log
   ```

## Troubleshooting

### Error: "Step X outputs not found"
**Solution**: Run the missing step first
```bash
# Missing Step 2? Run Step 1-2 first
steps: "1,2"
```

### Error: "invalid literal for int()"
**Solution**: Check step format
```yaml
# Correct formats:
steps: "1"           # Single step
steps: "1,2,3"       # Multiple steps
steps: "3,4"         # Non-consecutive steps

# Incorrect:
steps: "1, 2, 3"     # Don't use spaces
steps: "[1,2,3]"     # Don't use brackets
```

### Pipeline runs longer than expected
**Solution**: Check what steps are executing
```bash
grep "⊘ SKIPPED\|VALIDATION" logfile.log
```

## Implementation Details

### Step Validation Logic
- **Step 1 required**: Always safe to run (prepares data from scratch)
- **Step 2+ requires Step 1**: Checks for `{MAIN_DIR}/data/hallucination/{dataset_name}.csv`
- **Step 3+ requires Steps 1-2**: Checks for network directory with valid files
- **Step 4-5+ require Step 3**: Checks for trained model files in `{MAIN_DIR}/saves/`

### Probe Type Aware
- Results folders include probe type: `{model}_mlp` or `{model}_gcn`
- Data folders (Step 1-2) are probe-agnostic (reusable)
- Training folders (Step 3-5) are probe-specific

