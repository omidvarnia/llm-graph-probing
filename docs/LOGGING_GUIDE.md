# Production Logging Guide

This document describes the comprehensive logging system implemented across all pipeline steps for production visibility and debugging.

## Overview

The hallucination detection pipeline now includes **production-grade logging** across all 4 steps with:
- ✅ Clear step identification headers
- ✅ Device configuration details (GPU/CPU type and specs)
- ✅ Layer-specific analysis markers
- ✅ Structured metrics reporting with 80-character separators
- ✅ Processing phase transitions clearly marked

## Pipeline Logging Structure

### STEP 1: Dataset Construction & Activation Extraction
**File**: `hallucination/construct_dataset.py`

```
════════════════════════════════════════════════════════════════════════════════
STEP 1: DATASET CONSTRUCTION & ACTIVATION EXTRACTION
════════════════════════════════════════════════════════════════════════════════
Device: CPU (Data processing)
Dataset: truthfulqa
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
Dataset Statistics:
  Original size: 5915
  After deduplication: 5879
  Samples removed: 36
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
STEP 1 COMPLETE: Dataset Construction
════════════════════════════════════════════════════════════════════════════════
✓ Saved dataset to /path/to/output/truthfulqa.csv
════════════════════════════════════════════════════════════════════════════════
```

**Key Metrics Logged**:
- Dataset name
- Original size
- Deduplicated size
- Samples removed

---

### STEP 2: Neural Network Computation (FC Matrices)
**File**: `hallucination/compute_llm_network.py`

```
════════════════════════════════════════════════════════════════════════════════
STEP 2: NEURAL NETWORK COMPUTATION (FC MATRICES)
════════════════════════════════════════════════════════════════════════════════

GPU ALLOCATION & MULTIPROCESSING CONFIGURATION
────────────────────────────────────────────────────────────────────────────────
  Network Density: 5%
  GPU IDs: [0, 1]
  Batch Size: 16
  Worker Processes: 40
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
Producer Process Allocation (GPU Forward Pass)
════════════════════════════════════════════════════════════════════════════════
  Producer 0 → GPU 0
  Producer 1 → GPU 1

════════════════════════════════════════════════════════════════════════════════
Consumer Processes (CPU Correlation Computation)
════════════════════════════════════════════════════════════════════════════════
  40 CPU workers for correlation computation

════════════════════════════════════════════════════════════════════════════════
STEP 2 COMPLETE: Neural Topology Computation
════════════════════════════════════════════════════════════════════════════════
✓ Processed 5879 samples across layers [12]
════════════════════════════════════════════════════════════════════════════════
```

**Key Metrics Logged**:
- Network density setting
- GPU allocation (producer processes)
- Worker allocation (consumer processes)
- Dataset fraction (if sampling)
- Layers being processed

---

### STEP 3: Probe Training & Evaluation
**File**: `hallucination/train.py`

```
════════════════════════════════════════════════════════════════════════════════
STEP 3: HALLUCINATION DETECTION PROBE TRAINING & EVALUATION
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
Device Configuration:
  Device Type: ROCM
  GPU Name: AMD Radeon (MI250X)
  GPU Memory: 110 GB
────────────────────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
Layer Analysis Configuration:
  Layer ID: 5
  Probe Type: GCN (PyTorch Geometric)
  Network Density: 5.0%
  Hidden Channels: 32
  Number of Layers: 3
  Dropout: 0.1
  Batch Size: 32
  Learning Rate: 0.001
  Early Stop Patience: 20 epochs
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
LAYER ANALYSIS: TRAINING PHASE
════════════════════════════════════════════════════════════════════════════════
Layer 5 - Training probe on 4700 samples

[Training Progress...]
Epoch 1/100, Train Loss: 0.6823, Accuracy: 0.5432
Epoch 2/100, Train Loss: 0.5432, Accuracy: 0.6821
...

════════════════════════════════════════════════════════════════════════════════
LAYER ANALYSIS COMPLETE: Layer 5
════════════════════════════════════════════════════════════════════════════════
✓ Training completed successfully on device: ROCM
════════════════════════════════════════════════════════════════════════════════
```

**Key Metrics Logged**:
- Device type (ROCm/CUDA/CPU)
- GPU name and memory
- Layer ID
- Probe architecture and settings
- Training hyperparameters
- Training progress per epoch

---

### STEP 4: Probe Evaluation
**File**: `hallucination/eval.py`

```
════════════════════════════════════════════════════════════════════════════════
STEP 4: HALLUCINATION DETECTION PROBE EVALUATION
════════════════════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────────────
Device Configuration:
  Device Type: ROCM
  GPU Name: AMD Radeon (MI250X)
  GPU Memory: 110 GB
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
LAYER ANALYSIS: EVALUATION PHASE
════════════════════════════════════════════════════════════════════════════════
Layer 5 - Evaluating on 1179 samples

════════════════════════════════════════════════════════════════════════════════
LAYER 5 - EVALUATION RESULTS
════════════════════════════════════════════════════════════════════════════════

Classification Metrics:
  Test Accuracy:  0.8234
  Precision:      0.8156
  Recall:         0.8312
  F1 Score:       0.8233

Confusion Matrix:
  True Negatives:  512
  False Positives: 98
  False Negatives: 87
  True Positives:  523

Confusion Matrix (raw):
[[512  98]
 [ 87 523]]

════════════════════════════════════════════════════════════════════════════════
✓ Evaluation completed on device: ROCM
════════════════════════════════════════════════════════════════════════════════
```

**Key Metrics Logged**:
- Device type
- Layer ID
- Number of test samples
- Classification metrics (Accuracy, Precision, Recall, F1)
- Confusion matrix with clear labels
- Completion confirmation

---

## Device Type Indicators

The logging system automatically detects and reports device types:

| Device Type | Indicator | Example |
|---|---|---|
| AMD ROCm | `ROCM` | `Device Type: ROCM` |
| NVIDIA CUDA | `CUDA` | `Device Type: CUDA` |
| CPU Only | `CPU` | `Device Type: CPU` |

## Separator Patterns

All major sections use consistent separator patterns for visual clarity:

- **80-character separators**: `════════════════════════════════════════════════════════════════════════════════`
- **Sub-section separators**: `────────────────────────────────────────────────────────────────────────────────`
- **Metrics section headers**: `Layer Analysis Configuration:`, `Classification Metrics:`, etc.

## Logging Best Practices

1. **Always check device logs at Step 3 start**: Confirms correct GPU/CPU selection
2. **Monitor layer analysis markers**: Clearly indicates which layer is being processed
3. **Review final metrics in Step 4**: Layer-specific accuracy tells if probe is effective
4. **Check completion messages**: Confirms successful device execution for each step
5. **Use confusion matrix for debugging**: Helps understand false positive vs false negative patterns

## Troubleshooting via Logs

### Device-Related Issues
- **CPU fallback unexpectedly used**: Check if GPU is available (`nvidia-smi` or `rocm-smi`)
- **Memory errors**: Review GPU memory in device configuration block
- **Process allocation mismatch**: Check producer/consumer allocation in Step 2

### Training-Related Issues
- **Accuracy plateauing**: Check Layer Analysis Configuration in Step 3 for hyperparameters
- **Loss not decreasing**: Monitor epoch-by-epoch loss in training phase logs
- **Early stopping**: Check if patience (default: 20 epochs) was exceeded

### Evaluation-Related Issues
- **Low accuracy despite training**: Review confusion matrix for systematic errors
- **High false positives/negatives**: May indicate data quality or model tuning needs

## Example Full Pipeline Output

When running the complete pipeline, logs will flow as:
```
STEP 1 COMPLETE → STEP 2 HEADER → GPU/Multiprocessing Config → STEP 2 COMPLETE
  ↓
STEP 3 HEADER → Device Config → Layer Analysis Config → Training Progress → STEP 3 COMPLETE
  ↓
STEP 4 HEADER → Device Config → Evaluation Phase → Metrics & Confusion Matrix → STEP 4 COMPLETE
```

## Configuration Parameters Visibility

All key configuration parameters are now logged at step entry points:
- Dataset settings (name, fraction, size)
- Model architecture (layer ID, GCN/MLP choice, hidden dimensions)
- Training settings (batch size, learning rate, early stopping)
- Device allocation (GPU selection, multiprocessing workers)

This ensures every run is fully traceable and reproducible.
