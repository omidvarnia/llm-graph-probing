# Quick Reference: Hallucination Detection Pipeline

## 🚀 Quick Start Commands

### Run Complete Pipeline (Default Settings)
```bash
python hallucination_pipeline.py
```

### Run Specific Steps
```bash
# Step 1: Download data
python hallucination_pipeline.py --step 1

# Step 2: Compute neural topology
python hallucination_pipeline.py --step 2

# Step 3: Train probes
python hallucination_pipeline.py --step 3

# Step 4: Evaluate probes
python hallucination_pipeline.py --step 4

# Step 5: Graph statistics analysis
python hallucination_pipeline.py --step 5
```

---

## 🎯 Common Use Cases

### 1. Quick Test with Small Model
```bash
python hallucination_pipeline.py \
  --llm_model_name pythia-70m \
  --llm_layer 5 \
  --batch_size 8 \
  --num_layers 1
```

### 2. High-Performance Setup
```bash
python hallucination_pipeline.py \
  --llm_model_name qwen2.5-0.5b \
  --llm_layer 12 \
  --batch_size 32 \
  --num_layers 2 \
  --hidden_channels 64 \
  --lr 0.001
```

### 3. Memory-Constrained Environment
```bash
python hallucination_pipeline.py \
  --batch_size 4 \
  --sparse \
  --network_density 0.2 \
  --num_workers 2
```

### 4. Resume Interrupted Run
```bash
python hallucination_pipeline.py \
  --skip_data_prep \
  --skip_topology
```

---

## 📊 Key Parameters

### Model Selection
```bash
--llm_model_name <name>    # e.g., "pythia-70m", "qwen2.5-0.5b"
--llm_layer <id>           # Layer to probe (0 to num_layers-1)
--ckpt_step <step>         # Checkpoint step (-1 for final)
```

### Probe Configuration
```bash
--probe_input <type>       # "corr" (recommended) or "activation"
--num_layers <n>           # >0: GNN, 0: Linear, <0: MLP
--hidden_channels <n>      # Hidden layer dimensions (default: 32)
--lr <rate>                # Learning rate (0.001 for GNN, 0.00001 for Linear)
--dropout <rate>           # Dropout rate (0.0 - 0.5)
```

### Performance Tuning
```bash
--batch_size <n>           # Batch size for processing (default: 16)
--eval_batch_size <n>      # Batch size for evaluation (default: 16)
--gpu_id <id>              # GPU device ID (0, 1, etc.)
--num_workers <n>          # Number of worker processes (default: 4)
```

### Graph Configuration
```bash
--network_density <f>      # Graph density 0.0-1.0 (default: 1.0)
--sparse                   # Enable sparse graph storage
```

---

## 📂 Output Files

### Data Directory
```
data/hallucination/
├── truthfulqa-validation.csv              # Step 1: Dataset
└── <model_name>/                          # Step 2: Neural topology
    ├── 0/
    │   ├── layer_<N>_corr.npy            # Correlation matrix
    │   └── layer_<N>_activation.npy      # Activations
    └── ...
```

### Model Directory
```
saves/hallucination/<model_name>/layer_<N>/
└── best_model_density-<d>_dim-<h>_hop-<l>_input-<i>.pth  # Step 3: Trained model
```

### Logs
```
hallucination_pipeline.log                 # Execution log
runs/                                       # TensorBoard logs
```

---

## 🔧 Troubleshooting

### Out of Memory
```bash
# Solution 1: Reduce batch size
python hallucination_pipeline.py --batch_size 4

# Solution 2: Use sparse graphs
python hallucination_pipeline.py --sparse --network_density 0.2

# Solution 3: Use CPU
python hallucination_pipeline.py --gpu_id -1
```

### Dataset Download Fails
```bash
# Manually run data preparation
python -m hallucination.construct_dataset
```

### Poor Model Performance
```bash
# Try different learning rate
python hallucination_pipeline.py --step 3 --lr 0.0001

# Use correlation input
python hallucination_pipeline.py --step 3 --probe_input corr

# Add more layers
python hallucination_pipeline.py --step 3 --num_layers 3
```

---

## 📈 Expected Performance

### Runtime (Single GPU)
- **Small Model (70M params)**: ~30-40 minutes
- **Medium Model (0.5B params)**: ~1-1.5 hours
- **Large Model (1.4B params)**: ~3-4 hours

### Memory Requirements
- **Small Model**: ~2-4 GB GPU memory
- **Medium Model**: ~5-8 GB GPU memory
- **Large Model**: ~12-16 GB GPU memory

### Expected Accuracy
- **Baseline (Linear)**: 55-65%
- **GNN (1-layer)**: 70-80%
- **GNN (2-3 layers)**: 80-90%

---

## 🐍 Programmatic Usage

```python
from hallucination_pipeline import HallucinationPipeline

# Initialize pipeline
pipeline = HallucinationPipeline(
    llm_model_name="qwen2.5-0.5b",
    llm_layer=12,
    probe_input="corr",
    num_layers=2
)

# Run complete pipeline
results = pipeline.run_full_pipeline()

# Or run individual steps
pipeline.step_1_prepare_data()
pipeline.step_2_compute_neural_topology()
pipeline.step_3_train_probes()
pipeline.step_4_evaluate_probes()
pipeline.step_5_graph_statistics_analysis()
```

---

## 🔍 View Training Progress

```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Open browser to: http://localhost:6006
```

---

## 📚 Additional Resources

- **Full Guide**: See `HALLUCINATION_PIPELINE_GUIDE.md`
- **Examples**: Run `python example_pipeline_usage.py`
- **Main README**: See `README.md` for project overview
- **Paper**: [arXiv:2506.01042](https://arxiv.org/abs/2506.01042)

---

## ⚡ Multi-Layer Analysis Script

```bash
#!/bin/bash
# Analyze multiple layers
for layer in 6 12 18 24; do
  echo "Processing layer $layer..."
  python hallucination_pipeline.py \
    --llm_layer $layer \
    --skip_data_prep \
    --llm_model_name qwen2.5-0.5b
done
```

---

## 🔬 Experiment Templates

### Template 1: Architecture Comparison
```bash
# Linear probe
python hallucination_pipeline.py --step 3 --num_layers 0 --lr 0.00001

# MLP probe
python hallucination_pipeline.py --step 3 --num_layers -2 --lr 0.001

# GNN probe
python hallucination_pipeline.py --step 3 --num_layers 2 --lr 0.001
```

### Template 2: Sparsity Analysis
```bash
# Dense graph
python hallucination_pipeline.py --network_density 1.0

# 50% sparse
python hallucination_pipeline.py --sparse --network_density 0.5

# 20% sparse
python hallucination_pipeline.py --sparse --network_density 0.2
```

### Template 3: Cross-Model Study
```bash
for model in pythia-70m pythia-160m pythia-410m; do
  python hallucination_pipeline.py \
    --llm_model_name $model \
    --skip_data_prep
done
```

---

## ✅ Validation Checklist

Before running the pipeline:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GPU available (check with `nvidia-smi` or use `--gpu_id -1` for CPU)
- [ ] Sufficient disk space (~10GB for data + models)
- [ ] Internet connection (for dataset download in Step 1)

After each step:
- [ ] Check log file for errors
- [ ] Verify output files exist
- [ ] Monitor GPU memory usage
- [ ] Review TensorBoard metrics

---

**Last Updated**: December 19, 2025
