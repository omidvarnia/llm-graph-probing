# Hallucination Detection & Graph Statistics Pipeline Guide

## Overview

This guide provides step-by-step instructions for running the complete hallucination detection and graph statistics analysis pipeline using the `hallucination_pipeline.py` script.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Steps Explained](#pipeline-steps-explained)
3. [Usage Examples](#usage-examples)
4. [Advanced Configuration](#advanced-configuration)
5. [Output Files](#output-files)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify GPU Access** (optional but recommended)
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Run Complete Pipeline (Default Settings)

```bash
# Run the entire pipeline with default parameters
python hallucination_pipeline.py
```

This will:
- Download TruthfulQA dataset
- Compute neural topology for qwen2.5-0.5b model (layer 12)
- Train a GNN probe for hallucination detection
- Evaluate the trained probe
- Perform graph statistics analysis

---

## Pipeline Steps Explained

### Step 1: Data Preparation

**What it does:**
- Downloads the TruthfulQA dataset from HuggingFace
- Creates a CSV file with questions, answers, and labels
- Label 1 = truthful answer, Label 0 = hallucinated/false answer

**Output:**
- `data/hallucination/truthfulqa-validation.csv`

**Run only this step:**
```bash
python hallucination_pipeline.py --step 1
```

**Expected structure of the CSV:**
```
question_id,question,answer,label
0,"What is the capital of France?","Paris",1
0,"What is the capital of France?","London",0
...
```

---

### Step 2: Compute Neural Topology

**What it does:**
- Processes each answer through the LLM
- Extracts hidden-state activations at specified layer
- Computes correlation matrices between neurons
- Saves neural topology graphs for each sample

**Key Parameters:**
- `--llm_model_name`: Model to analyze (e.g., "qwen2.5-0.5b", "pythia-70m")
- `--llm_layer`: Layer to extract features from (e.g., 12 for middle layer)
- `--network_density`: Sparsity level (1.0 = full graph, 0.1 = 10% of edges)
- `--sparse`: Enable sparse graph storage (saves memory)

**Output:**
- `data/hallucination/<model_name>/[sample_id]/layer_<layer>_corr.npy`
- `data/hallucination/<model_name>/[sample_id]/layer_<layer>_activation.npy`

**Run only this step:**
```bash
python hallucination_pipeline.py --step 2 --llm_layer 12 --network_density 1.0
```

**Example with sparse graphs:**
```bash
python hallucination_pipeline.py --step 2 --sparse --network_density 0.3
```

---

### Step 3: Train Hallucination Detection Probes

**What it does:**
- Loads the neural topology data
- Trains a probe (GNN/MLP/Linear) to classify truthful vs. hallucinated answers
- Monitors performance on validation set
- Saves best model checkpoint

**Key Parameters:**
- `--probe_input`: Input type
  - `"corr"`: Use correlation matrices (recommended for graph structure)
  - `"activation"`: Use raw activations
- `--num_layers`: Architecture choice
  - `> 0`: GNN with specified number of layers
  - `= 0`: Simple linear probe
  - `< 0`: MLP with |num_layers| hidden layers
- `--hidden_channels`: Dimensionality of hidden layers
- `--lr`: Learning rate (0.001 for GNN/MLP, 0.00001 for linear on correlations)
- `--dropout`: Dropout rate for regularization

**Output:**
- `saves/hallucination/<model_name>/layer_<layer>/best_model_*.pth`
- TensorBoard logs in `runs/`

**Run only this step:**
```bash
python hallucination_pipeline.py --step 3 \
  --probe_input corr \
  --num_layers 2 \
  --hidden_channels 64 \
  --lr 0.001 \
  --dropout 0.1
```

**Different probe architectures:**

1. **Graph Neural Network (GNN) - Recommended for correlation graphs**
   ```bash
   python hallucination_pipeline.py --step 3 \
     --probe_input corr \
     --num_layers 2 \
     --hidden_channels 32 \
     --lr 0.001
   ```

2. **Linear Probe - Fast baseline**
   ```bash
   python hallucination_pipeline.py --step 3 \
     --probe_input corr \
     --num_layers 0 \
     --lr 0.00001
   ```

3. **MLP - For activation features**
   ```bash
   python hallucination_pipeline.py --step 3 \
     --probe_input activation \
     --num_layers -2 \
     --hidden_channels 128 \
     --lr 0.001
   ```

---

### Step 4: Evaluate Trained Probes

**What it does:**
- Loads the best trained model
- Evaluates on test set
- Reports comprehensive metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

**Output:**
- Printed metrics to console and log file
- Evaluation results saved to model directory

**Run only this step:**
```bash
python hallucination_pipeline.py --step 4
```

**Expected output format:**
```
Test Accuracy: 0.8542
Test Precision: 0.8734
Test Recall: 0.8312
Test F1 Score: 0.8518

Confusion Matrix:
[[245  32]
 [ 41 203]]
```

---

### Step 5: Graph Statistics Analysis

**What it does:**
- Analyzes topology differences between truthful and hallucinated answers
- Computes intra-answer correlations (within true, within false)
- Computes inter-answer correlations (between true and false)
- Identifies topological signatures of hallucination

**Key Metrics:**
- **Intra-True Correlation**: Similarity among truthful answers for the same question
- **Intra-False Correlation**: Similarity among hallucinated answers for the same question
- **Inter-Correlation**: Similarity between truthful and hallucinated answers

**Run only this step:**
```bash
python hallucination_pipeline.py --step 5 --llm_layer 12
```

**Analyze different features:**
```bash
# Analyze correlation matrices
python -m hallucination.graph_analysis \
  --llm_model_name qwen2.5-0.5b \
  --layer 12 \
  --feature corr

# Analyze activations
python -m hallucination.graph_analysis \
  --llm_model_name qwen2.5-0.5b \
  --layer 12 \
  --feature activation
```

**Expected insights:**
- Higher intra-true correlation suggests consistent neural patterns for truthful answers
- Lower inter-correlation suggests distinct neural signatures for true vs. false

---

## Usage Examples

### Example 1: Quick Test Run (Small Model)

```bash
# Run complete pipeline on a small model
python hallucination_pipeline.py \
  --llm_model_name pythia-70m \
  --llm_layer 5 \
  --batch_size 8 \
  --num_layers 1 \
  --hidden_channels 16
```

### Example 2: High-Performance Setup (Large Model)

```bash
# Run on larger model with optimized settings
python hallucination_pipeline.py \
  --llm_model_name pythia-1.4b \
  --llm_layer 20 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --num_layers 3 \
  --hidden_channels 64 \
  --lr 0.0005 \
  --dropout 0.2 \
  --gpu_id 0
```

### Example 3: Sparse Graph Analysis

```bash
# Run with sparse graphs to save memory
python hallucination_pipeline.py \
  --llm_model_name qwen2.5-0.5b \
  --llm_layer 12 \
  --network_density 0.2 \
  --sparse \
  --probe_input corr \
  --num_layers 2
```

### Example 4: Resume Interrupted Pipeline

```bash
# Skip already completed steps
python hallucination_pipeline.py \
  --skip_data_prep \
  --skip_topology \
  --llm_model_name qwen2.5-0.5b
```

### Example 5: Multi-Layer Analysis

```bash
# Analyze different layers sequentially
for layer in 6 12 18 24; do
  echo "Processing layer $layer..."
  python hallucination_pipeline.py \
    --llm_layer $layer \
    --skip_data_prep \
    --llm_model_name pythia-1b
done
```

### Example 6: Comparison Across Models

```bash
# Compare different models
for model in pythia-70m pythia-160m pythia-410m; do
  echo "Processing $model..."
  python hallucination_pipeline.py \
    --llm_model_name $model \
    --llm_layer 10 \
    --skip_data_prep
done
```

---

## Advanced Configuration

### Multi-GPU Processing

For Step 2 (neural topology computation), you can use multiple GPUs:

```bash
# Edit the pipeline script or run the module directly
python -m hallucination.compute_llm_network \
  --dataset_filename data/hallucination/truthfulqa-validation.csv \
  --llm_model_name qwen2.5-0.5b \
  --llm_layer 12 \
  --batch_size 16 \
  --gpu_id 0 1 2 3 \
  --num_workers 40
```

### Custom Learning Rate Schedules

For different probe architectures, adjust learning rates:

```python
# For linear probes on correlation graphs
--lr 0.00001

# For GNN probes
--lr 0.001

# For MLP probes
--lr 0.001 or 0.0005
```

### Memory Optimization

If running out of memory:

```bash
# Reduce batch size
python hallucination_pipeline.py --batch_size 4 --eval_batch_size 8

# Use sparse graphs
python hallucination_pipeline.py --sparse --network_density 0.1

# Process fewer samples at a time
python hallucination_pipeline.py --num_workers 2
```

---

## Output Files

### Directory Structure After Running Pipeline

```
llm-graph-probing/
├── data/
│   └── hallucination/
│       ├── truthfulqa-validation.csv          # Step 1 output
│       └── qwen2.5-0.5b/                       # Step 2 output
│           ├── 0/
│           │   ├── layer_12_corr.npy
│           │   ├── layer_12_activation.npy
│           │   └── ...
│           ├── 1/
│           └── ...
├── saves/
│   └── hallucination/
│       └── qwen2.5-0.5b/
│           └── layer_12/
│               └── best_model_density-1.0_dim-32_hop-1_input-corr.pth  # Step 3 output
├── runs/                                       # TensorBoard logs
└── hallucination_pipeline.log                 # Execution log
```

### Understanding Output Files

1. **Neural Topology Files** (`layer_X_corr.npy`)
   - NumPy array of shape `[num_neurons, num_neurons]`
   - Correlation matrix between neuron activations
   - Can be loaded with: `np.load('layer_12_corr.npy')`

2. **Model Checkpoints** (`best_model_*.pth`)
   - PyTorch state dict
   - Contains trained probe weights
   - Load with: `torch.load('best_model_*.pth')`

3. **TensorBoard Logs**
   - Training/validation curves
   - View with: `tensorboard --logdir runs/`

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
python hallucination_pipeline.py --batch_size 4

# Use CPU instead of GPU
python hallucination_pipeline.py --gpu_id -1

# Use sparse graphs
python hallucination_pipeline.py --sparse --network_density 0.2
```

#### 2. Dataset Download Fails

**Error:**
```
ConnectionError: Failed to download dataset
```

**Solutions:**
```bash
# Check internet connection
# Try manual download from HuggingFace
# Use VPN if behind firewall

# Or manually run:
python -m hallucination.construct_dataset
```

#### 3. Model Not Found

**Error:**
```
KeyError: 'model_name' not in hf_model_name_map
```

**Solution:**
Check available models in `utils/constants.py` or use standard HuggingFace names.

#### 4. Missing Topology Files

**Error:**
```
FileNotFoundError: Neural topology file not found
```

**Solution:**
```bash
# Re-run Step 2
python hallucination_pipeline.py --step 2

# Or check if it completed successfully
ls -la data/hallucination/qwen2.5-0.5b/
```

#### 5. Poor Probe Performance

**Symptoms:**
- Accuracy ~50% (random guessing)
- High loss, not decreasing

**Solutions:**
```bash
# Try different learning rate
python hallucination_pipeline.py --step 3 --lr 0.0001

# Use correlation instead of activation
python hallucination_pipeline.py --step 3 --probe_input corr

# Add more GNN layers
python hallucination_pipeline.py --step 3 --num_layers 3

# Reduce regularization
python hallucination_pipeline.py --step 3 --dropout 0.0
```

---

## Performance Benchmarks

### Expected Runtime (on single GPU)

| Step | Small Model (70M) | Medium Model (0.5B) | Large Model (1.4B) |
|------|-------------------|---------------------|---------------------|
| Step 1: Data Prep | ~2 min | ~2 min | ~2 min |
| Step 2: Topology | ~15 min | ~45 min | ~2 hours |
| Step 3: Training | ~10 min | ~20 min | ~30 min |
| Step 4: Evaluation | ~2 min | ~5 min | ~10 min |
| Step 5: Analysis | ~5 min | ~10 min | ~15 min |
| **Total** | **~34 min** | **~82 min** | **~3.5 hours** |

*Note: Times are approximate and depend on hardware*

### Expected Memory Usage

| Component | Small Model | Medium Model | Large Model |
|-----------|-------------|--------------|-------------|
| Model Loading | ~500 MB | ~2 GB | ~6 GB |
| Topology Data | ~1 GB | ~3 GB | ~8 GB |
| Training | ~2 GB | ~5 GB | ~12 GB |

---

## Next Steps

After running the pipeline:

1. **Visualize Results**
   ```bash
   tensorboard --logdir runs/
   ```

2. **Analyze Multiple Layers**
   Compare which layers are best for hallucination detection

3. **Try Different Architectures**
   Experiment with GNN vs. MLP vs. Linear probes

4. **Cross-Model Analysis**
   Compare hallucination patterns across different LLMs

5. **Publication-Ready Plots**
   Use the graph analysis output to create visualizations

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{zheng2025probing,
  title={Probing Neural Topology of Large Language Models}, 
  author={Zheng, Yu and Yuan, Yuan and Li, Yong and Santi, Paolo},
  journal={arXiv preprint arXiv:2506.01042},
  year={2025}
}
```

---

## Support

For issues or questions:
- Check the main README.md
- Review the troubleshooting section above
- Examine the log file: `hallucination_pipeline.log`
- Refer to individual module scripts in `hallucination/` directory
