# Hallucination Detection in LLMs: Technical Documentation

This directory contains comprehensive technical documentation for the hallucination detection analysis pipeline.

## Contents

### Main Document

- **`hallucination_detection_technical_report.tex`** - Complete technical report in LaTeX format

This report documents a comprehensive analysis pipeline for detecting hallucinations in large language models (LLMs) by leveraging their internal neural network structure.

## Report Structure

The technical report covers:

1. **Introduction** - Motivation and approach overview
2. **Experimental Setup** - Configuration parameters and environment
3. **GPT-2 Model Architecture** - Detailed specifications (124M parameters, 12 layers, 768 hidden dimension)
4. **Step 1: Dataset Construction** - TruthfulQA dataset with 5,915 question-answer pairs
5. **Step 2: Functional Connectivity Networks** - Computing correlation matrices from layer 5 activations
6. **Step 3: GNN Probe Training** - Training a 3-layer Graph Convolutional Network
7. **Step 4: Model Evaluation** - Binary classification metrics (accuracy, precision, recall, F1)
8. **Step 5: Graph Analysis** - Topology analysis and connectivity statistics
9. **Pipeline Timing** - Complete computational cost breakdown
10. **Key Definitions** - Clear variable definitions and explanations
11. **Discussion** - Methodological insights and future work
12. **Reproducibility** - Step-by-step instructions to reproduce the analysis
13. **Appendices** - Module references, file organization, and function documentation

## Key Highlights

### Data Statistics

- **Total Samples**: 5,915 question-answer pairs
- **Training Set**: 4,732 samples (80%)
- **Test Set**: 1,183 samples (20%)
- **Label Balance**: 50.3% truthful, 49.7% hallucinated

### Model Architecture

- **Layer**: GPT-2 layer 5 (middle transformer block)
- **Neurons per Token**: 768
- **Correlation Matrix Size**: 768 × 768
- **Network Sparsity**: 5% (29,491 out of 589,824 edges retained)

### GNN Probe

- **Architecture**: 3-layer Graph Convolutional Network
- **Hidden Dimension**: 32
- **Global Pooling**: Mean + Max aggregation
- **Output**: Binary classification (hallucinated vs. truthful)

### Computational Cost

- **Total Pipeline Time**: ~12.8 minutes (769.9 seconds)
  - Step 1 (Dataset): 3.5s (0.4%)
  - Step 2 (Network): 307.6s (41.3%)
  - Step 3 (Training): 289.7s (39.0%)
  - Step 4 (Evaluation): 168.6s (22.7%)
  - Step 5 (Analysis): 0.7s (0.1%)

## Running the Analysis

To reproduce this analysis:

```bash
# Clone repository
git clone https://github.com/omidvarnia/llm-graph-probing.git
cd llm-graph-probing

# Set up environment
export MAIN_DIR=/path/to/output/directory
export PYTHONPATH=/path/to/llm-graph-probing

# Run the pipeline
python my_analysis/hallucination_detection_analysis.py
```

Outputs are saved to: `{MAIN_DIR}/reports/hallucination_analysis/`

## Configuration Parameters

All parameters are configurable in `my_analysis/hallucination_detection_analysis.py`:

```python
# Dataset and Model
dataset_name = "openwebtext"      # Options: truthfulqa, halueval, medhallu, helm
model_name = "gpt2"               # Options: gpt2, gpt2-medium, gpt2-large, pythia-160m
layer_id = 5                      # Select layer 0-11

# Network Configuration
network_density = 0.05            # Sparsity: keep top 5% of edges
probe_input = "corr"              # Input type: correlation matrix

# GNN Probe
num_channels = 32                 # Hidden dimension
num_layers = 3                    # Number of GCN layers

# Training
learning_rate = 0.001
num_epochs = 10
batch_size = 16
```

## Module and Function Reference

### Core Modules

- **`hallucination.construct_dataset`** - TruthfulQA dataset loading and construction
- **`hallucination.compute_llm_network`** - Functional connectivity computation via multiprocessing
- **`hallucination.dataset`** - PyTorch Dataset and DataLoader for graphs
- **`hallucination.train`** - GCN probe training with TensorBoard logging
- **`hallucination.eval`** - Model evaluation on test set
- **`hallucination.graph_analysis`** - Topology analysis and statistics
- **`hallucination.utils`** - Utility functions (test_fn, format_prompt, etc.)
- **`utils.probing_model`** - GCN and MLP architectures
- **`utils.model_utils`** - Model loading and inference utilities

### Key Functions

#### Dataset Functions
- `hallucination.construct_dataset._build_truthfulqa()` - Load dataset from Hugging Face
- `hallucination.dataset.TruthfulQADataset` - PyTorch Dataset class for graphs
- `hallucination.dataset.get_truthfulqa_dataloader()` - Create train/test DataLoaders

#### Network Computation
- `hallucination.compute_llm_network.run_llm()` - Producer process for LLM inference
- `hallucination.compute_llm_network.compute_networks()` - Main computation coordinator

#### Training & Evaluation
- `hallucination.train.train_model()` - Main training loop with early stopping
- `hallucination.eval.evaluate()` - Evaluation coordinator
- `hallucination.utils.test_fn()` - Compute metrics (accuracy, precision, recall, F1, confusion matrix)
- `utils.probing_model.GCNProbe` - Graph Convolutional Network implementation
- `utils.probing_model.MLPProbe` - Multi-layer Perceptron implementation

#### Analysis Functions
- `hallucination.graph_analysis.load_correlations()` - Load correlation matrices
- `hallucination.graph_analysis.intra_inter_analysis()` - Compare connectivity patterns across classes

## Generated Artifacts

The pipeline generates several output files:

- **`dataset_head.csv`** - Sample of first few dataset records
- **`dataset_label_distribution.png`** - Visualization of label distribution
- **`fc_before_threshold.png`** - Correlation matrix before thresholding
- **`fc_after_threshold.png`** - Correlation matrix after 5% thresholding
- **`train_loss.png`** - Training loss curve
- **`test_metrics.png`** - Test-set metrics (accuracy, precision, recall, F1)
- **`step_durations.png`** - Computational cost breakdown
- **`summary.json`** - Complete analysis summary with paths and metrics
- **`step*.log`** - Detailed logs for each pipeline step

## Technical Details

### Functional Connectivity Computation

For each question-answer pair:
1. Tokenize input text
2. Forward pass through GPT-2 to get layer 5 hidden states: $\mathbf{H}^{(5)} \in \mathbb{R}^{T \times 768}$
3. Compute Pearson correlation: $\mathbf{C}_{ij} = \text{corr}(\mathbf{H}^{(5)}_{:,i}, \mathbf{H}^{(5)}_{:,j})$
4. Apply percentile threshold to keep top 5% of correlations
5. Convert to sparse graph representation (edge lists)

### GNN Architecture

```
Input: Graph with 768 nodes (neurons) and ~29,491 edges (correlations)
  ↓
Embedding (768 → 32)
  ↓
GCN Layer 1 (32 → 32) + ReLU
  ↓
GCN Layer 2 (32 → 32) + ReLU
  ↓
GCN Layer 3 (32 → 32) + ReLU
  ↓
Global Pooling: [Mean; Max] → 64-dim
  ↓
FC Layer (64 → 32) + ReLU
  ↓
Output Layer (32 → 1) + Sigmoid
  ↓
Binary Classification (hallucinated vs. truthful)
```

### Training Details

- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Batch Size**: 16
- **Max Epochs**: 10
- **Early Stopping**: 20 patience on F1 score

## Environment Requirements

See `requirements.txt` for full dependencies:

```
torch
torch_geometric
transformers
datasets
numpy
pandas
scikit-learn
matplotlib
tensorboard
tqdm
absl-py
```

## Reproducing the Report

To generate a PDF from the LaTeX source:

```bash
cd docs/
pdflatex hallucination_detection_technical_report.tex
pdflatex hallucination_detection_technical_report.tex  # Run twice for TOC
```

The compiled PDF includes:
- Detailed explanations of all pipeline steps
- Figures from the analysis (connectivity matrices, training curves, etc.)
- Complete function reference and module documentation
- Step-by-step reproducibility instructions

## Citation

If using this analysis, please cite:

```bibtex
@misc{hallucination_detection_gnn,
  title={Detecting Hallucinations in Large Language Models Using Graph Neural Networks},
  author={Graph Probing Analysis},
  year={2024}
}
```

## Questions?

For questions about the analysis pipeline, see:
- Main orchestration: `my_analysis/hallucination_detection_analysis.py`
- Detailed module code: `hallucination/` directory
- Utilities: `utils/` directory

## License

See LICENSE file in repository root.
