# Quick Reference: Hallucination Detection Pipeline

## One-Page Summary

This pipeline detects hallucinations in language models by analyzing functional connectivity patterns in their neural networks.

### The Basic Idea

1. **Extract Neural Activations**: Run question-answer pairs through GPT-2 layer 5, getting 768-dimensional activations per token
2. **Compute Connectivity**: Calculate correlations between all pairs of neurons → 768×768 correlation matrix
3. **Sparsify Network**: Keep only strongest 5% of connections to reduce complexity
4. **Train Graph Network**: Train a GNN to predict "truthful" vs "hallucinated" from the connectivity graph
5. **Evaluate**: Measure accuracy on held-out test set

## Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| **Dataset** | TruthfulQA |
| **Total Samples** | 5,915 question-answer pairs |
| **Train/Test Split** | 80% / 20% (4,732 / 1,183) |
| **Class Balance** | 50.3% truthful, 49.7% hallucinated |
| **Layer Used** | Layer 5 (middle of 12-layer GPT-2) |
| **Neurons per Layer** | 768 |
| **Network Density** | 5% (29,491 edges out of 589,824) |
| **GNN Hidden Dim** | 32 |
| **GNN Layers** | 3 |
| **Training Epochs** | 10 (with early stopping) |
| **Total Runtime** | ~12.8 minutes |

## Architecture Diagram

```
Input: "Is this statement true?" + "Model response"
    ↓
GPT-2 Tokenization & Forward Pass
    ↓
Extract Hidden States from Layer 5 (768 dimensions)
    ↓
Compute Correlation Matrix (768 × 768)
    ↓
Threshold to 5% Density (top 29,491 edges)
    ↓
Build Graph: 768 nodes, correlated pairs as edges
    ↓
GCN Processing:
    - Node Embedding (768 → 32)
    - 3× [GCN + ReLU]
    - Global Pooling [Mean; Max]
    - 2× FC layers
    ↓
Binary Classification Output
    ↓
Prediction: Hallucinated (0) or Truthful (1)
```

## Five-Step Pipeline

### Step 1: Dataset Construction (~3.5 sec)
- Load TruthfulQA from Hugging Face
- 5,915 question-answer pairs with binary labels
- Save to CSV with columns: [question_id, question, answer, label]
- **Module**: `hallucination.construct_dataset`
- **Function**: `_build_truthfulqa()`

### Step 2: Network Topology (~307.6 sec)
- For each Q-A pair, extract GPT-2 layer 5 activations
- Compute 768×768 Pearson correlation matrix
- Apply percentile thresholding (keep top 5%)
- Save sparse edge lists and dense correlation matrices
- **Module**: `hallucination.compute_llm_network`
- **Function**: `run_llm()` (producer), `compute_networks()` (coordinator)

### Step 3: Probe Training (~289.7 sec)
- Load graphs from disk
- Initialize GCN with 32 hidden channels, 3 layers
- Train with Adam optimizer (lr=0.001, weight_decay=1e-5)
- Early stopping on test F1-score (patience=20)
- Save best model checkpoint
- **Module**: `hallucination.train`
- **Function**: `train_model()`

### Step 4: Evaluation (~168.6 sec)
- Load best trained model
- Evaluate on 1,183 test samples
- Compute: accuracy, precision, recall, F1, confusion matrix
- Log metrics to TensorBoard and console
- **Module**: `hallucination.eval`
- **Function**: `evaluate()`

### Step 5: Graph Analysis (~0.7 sec)
- Load all correlation matrices
- Compute intra-class and inter-class connectivity statistics
- Generate summary statistics JSON
- **Module**: `hallucination.graph_analysis`
- **Function**: `intra_inter_analysis()`

## Configuration Parameters

Edit these in `my_analysis/hallucination_detection_analysis.py` (lines ~130-160):

```python
# Dataset & Model Selection
dataset_name = "openwebtext"    # "truthfulqa", "halueval", "medhallu", "helm"
model_name = "gpt2"             # "gpt2", "gpt2-medium", "gpt2-large", "pythia-160m"
ckpt_step = -1                  # -1 for main, or specific step for finetuned models

# Network Topology
layer_id = 5                    # Layer 0-11 for GPT-2 (12 total layers)
network_density = 0.05          # Keep top 5% of edges (0.01-0.2 typical range)
batch_size = 16                 # For network computation

# GNN Probe
probe_input = "corr"            # "corr", "activation", "activation_avg"
num_channels = 32               # Hidden dimension
num_layers = 3                  # GCN layers

# Training
learning_rate = 0.001
num_epochs = 10
from_sparse_data = True         # Use pre-thresholded sparse format
eval_batch_size = 32
```

## Output Files Generated

All outputs saved to: `/ptmp/aomidvarnia/analysis_results/llm_graph/reports/hallucination_analysis/`

| File | Purpose |
|------|---------|
| `dataset_head.csv` | First few dataset rows |
| `dataset_label_distribution.png` | Histogram of label distribution |
| `fc_before_threshold.png` | Full correlation matrix visualization |
| `fc_after_threshold.png` | Thresholded (5% density) matrix |
| `train_loss.png` | Training loss curve across epochs |
| `test_metrics.png` | Test accuracy, precision, recall, F1 curves |
| `step_durations.png` | Computational cost pie chart |
| `summary.json` | Complete pipeline summary with all metrics |
| `step1_construct_dataset.log` | Dataset construction log |
| `step2_compute_network.log` | Network computation log |
| `step3_train.log` | Training log with progress bars |
| `step4_eval.log` | Evaluation results |
| `step5_graph_analysis.log` | Analysis summary |

## Running the Pipeline

```bash
# Set up environment
export MAIN_DIR=/path/to/output
export PYTHONPATH=/path/to/llm-graph-probing

# Run analysis
cd /u/aomidvarnia/GIT_repositories/llm-graph-probing
python my_analysis/hallucination_detection_analysis.py

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir=/path/to/output/runs/ --port=6006
# Then visit http://localhost:6006
```

## GPT-2 Architecture Details

| Component | Value |
|-----------|-------|
| Model | GPT-2 Base |
| Total Parameters | 124 million |
| Hidden Dimension | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Context Window | 1,024 tokens |
| Vocab Size | 50,257 tokens |

**Layer 5 Selection**: Middle layer (of 12), expected to capture semantic relationships relevant to truthfulness prediction.

## Understanding the Numbers in Output

### Confusion Matrix Example
```
[[TN, FP],
 [FN, TP]]
 
[[815,  30],      # True Negatives: 815 hallucinations correctly detected
 [ 45, 293]]      # True Positives: 293 truths correctly identified
```

- **TN**: Correctly identified hallucinations (target: high)
- **FP**: False alarms (hallucinations mislabeled as truthful)
- **FN**: Misses (truthful labeled as hallucinated)
- **TP**: Correctly identified truthful statements

### Metrics Explained
- **Accuracy**: (TP + TN) / Total → Overall correctness
- **Precision**: TP / (TP + FP) → When we say "truthful", how often correct?
- **Recall**: TP / (TP + FN) → What fraction of truthful statements do we catch?
- **F1-Score**: Harmonic mean of precision and recall → Balanced metric

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "MAIN_DIR not set" | `export MAIN_DIR=/path/to/output` |
| "No module named hallucination" | `export PYTHONPATH=/path/to/llm-graph-probing` |
| Out of memory (OOM) | Reduce `batch_size`, `num_channels`, or `num_epochs` |
| Slow computation | Ensure GPU is being used; check `nvidia-smi` |
| TensorBoard events missing | Check `runs/hallucination/truthfulqa/{model}/layer_{id}/` |

## Key Functions Reference

```python
# Dataset
from hallucination.dataset import get_truthfulqa_dataloader
train_loader, test_loader = get_truthfulqa_dataloader(...)

# Training
from hallucination.train import train_model
train_model(model, train_loader, test_loader, optimizer, scheduler, writer, ...)

# Evaluation
from hallucination.utils import test_fn
accuracy, precision, recall, f1, cm = test_fn(model, test_loader, device)

# Network Computation
from hallucination.compute_llm_network import compute_networks
compute_networks(...)

# Analysis
from hallucination.graph_analysis import intra_inter_analysis
stats = intra_inter_analysis(correlation_matrices)
```

## Technical Insights

### Why Functional Connectivity?
- **Neural Signature of Hallucination**: Different connectivity patterns emerge when models generate hallucinated vs. truthful content
- **Layer Specificity**: Layer 5 (middle) balances low-level patterns vs. high-level output preparation
- **Interpretability**: Neuron pairs with high predictive power are identifiable and analyzable

### Why Graph Neural Networks?
- **Topology Preservation**: GNNs naturally process graph structure
- **Efficiency**: 5% density reduction keeps computation tractable
- **Aggregation**: Global pooling combines local neuron information into graph-level predictions
- **Generalization**: Learned patterns may transfer across models

### Network Sparsity (5%)
- **Full Graph**: 589,824 edges (768² - 768)
- **Thresholded**: ~29,491 edges (top 5% by absolute correlation)
- **Benefit**: Focuses on strongest, most reliable connections
- **Trade-off**: May lose weak but collectively meaningful patterns

## Paper References

Key concepts implemented:
- **Graph Neural Networks**: Kipf & Welling (2017)
- **Transformer Architecture**: Vaswani et al. (2017)
- **Hallucination in LLMs**: Zhang et al. (2023)
- **TruthfulQA**: Lin et al. (2022)

## Next Steps

1. **Modify Configuration**: Change layer, density, or model
2. **Test on Other Datasets**: Try HaluEval, MedHallu, HELM
3. **Analyze Learned Patterns**: Which neuron pairs are hallucination-predictive?
4. **Transfer Learning**: Can models trained on one dataset work on another?
5. **Scale to Larger Models**: Test on GPT-2-Medium, GPT-2-Large, or modern LLMs

## Document Index

- **Detailed Report**: `hallucination_detection_technical_report.tex` (full technical documentation with figures and references)
- **This File**: Quick reference for configuration and running
- **Code**: `my_analysis/hallucination_detection_analysis.py` (main orchestration script)
