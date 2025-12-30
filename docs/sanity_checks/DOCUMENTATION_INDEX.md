# Complete Documentation Index: Correlation Matrices & GNN Architecture

## 📚 All Documentation Files

This repository now contains 6 comprehensive documentation files explaining the hallucination detection pipeline:

### **Core Documentation Files**

| File | Purpose | Read First? |
|------|---------|------------|
| [CORRELATION_SANITIZATION_README.md](CORRELATION_SANITIZATION_README.md) | **Index** for all sanitization documentation | ✓ START HERE |
| [CORRELATION_MATRICES_AND_GNN_INPUT.md](CORRELATION_MATRICES_AND_GNN_INPUT.md) | **Quick reference**: Which correlations, input sizes, exact tensors | ✓ THEN READ THIS |
| [GNN_INPUT_COMPLETE_EXAMPLE.md](GNN_INPUT_COMPLETE_EXAMPLE.md) | **End-to-end walkthrough**: Question #42 from input to output | ✓ THEN STUDY THIS |

### **Detailed Deep-Dive Documents**

| File | Purpose | Best For |
|------|---------|----------|
| [CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md](CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md) | Complete architecture of NaN/Inf protection across 3 layers | Understanding robustness & failure modes |
| [CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md](CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md) | Visual flow diagrams + pseudo-code for data transformations | Visual learners & code reviewers |
| [CORRELATION_SANITIZATION_EXAMPLES.md](CORRELATION_SANITIZATION_EXAMPLES.md) | 5 concrete numerical examples with real tensor values | Hands-on understanding |

---

## 🎯 Quick Answer to Your Questions

### Q1: Which correlation matrix is used to classify true/false labels?

**Answer**: Per-layer Pearson correlation matrices computed from LLM hidden states.

- **Type**: Pearson correlation between hidden dimensions within a single layer
- **Size**: For Layer 5 of Qwen2.5-0.5B: **1024×1024** matrix
  - Rows/Columns: 1024 hidden dimensions
  - Values: Correlation coefficients ∈ [-1.0, 1.0]
- **What it measures**: How strongly each hidden dimension co-activates with others
- **Why it works**: True vs. false statements produce different correlation patterns

**See**: [CORRELATION_MATRICES_AND_GNN_INPUT.md § 1.2](CORRELATION_MATRICES_AND_GNN_INPUT.md#12-which-matrix-is-used-for-truefalseclassification)

---

### Q2: What are the input correlation matrices to the GNN?

**Answer**: Sparse representations of the per-layer correlation matrix after thresholding.

```
Input Format:
  x:           [1024] node IDs
  edge_index:  [2, 630,000] (source-destination pairs)
  edge_attr:   [630,000] correlation values
  y:           scalar label (0 or 1)
```

- **Nodes**: 1024 (one per hidden dimension)
- **Edges**: ~630,000 (for network_density=1.0; varies with density setting)
- **Edge weights**: Correlation values directly used as message weights

**See**: [CORRELATION_MATRICES_AND_GNN_INPUT.md § 2.2](CORRELATION_MATRICES_AND_GNN_INPUT.md#22-exact-input-sizes-to-gnn)

---

### Q3: What are the exact sizes?

**Answer**: 

| Component | Size | Range |
|-----------|------|-------|
| **Correlation Matrix** | 1024×1024 | All hidden dims |
| **GNN Nodes** | 1024 | One per hidden dim |
| **GNN Edges** | 630K (density=1.0) | Variable with network_density |
| **Edge Features** | 630K values | [-1.0, 1.0] (correlations) |
| **GNN Output Dim** | 1 logit | Binary classification |
| **Hidden Channel** | 32 | Embedding dimension |
| **Batch Size** | 16 graphs | 16 questions at once |

**See**: [GNN_INPUT_COMPLETE_EXAMPLE.md § Summary](GNN_INPUT_COMPLETE_EXAMPLE.md#summary-data-shapes-through-pipeline)

---

## 📖 Reading Guide by Objective

### Objective: "Quick 5-minute Overview"
1. Read: [CORRELATION_MATRICES_AND_GNN_INPUT.md § Quick Answer](CORRELATION_MATRICES_AND_GNN_INPUT.md)
2. Skim: [GNN_INPUT_COMPLETE_EXAMPLE.md § Summary Table](GNN_INPUT_COMPLETE_EXAMPLE.md#summary-data-shapes-through-pipeline)

### Objective: "Understand the Architecture"
1. Read: [CORRELATION_MATRICES_AND_GNN_INPUT.md](CORRELATION_MATRICES_AND_GNN_INPUT.md) (20 min)
2. Study: [GNN_INPUT_COMPLETE_EXAMPLE.md § Step-by-step](GNN_INPUT_COMPLETE_EXAMPLE.md#step-1-question-text) (30 min)
3. Reference: [CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md](CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md)

### Objective: "Understand Data Robustness"
1. Read: [CORRELATION_SANITIZATION_README.md](CORRELATION_SANITIZATION_README.md) (10 min)
2. Study: [CORRELATION_SANITIZATION_EXAMPLES.md](CORRELATION_SANITIZATION_EXAMPLES.md) (40 min)
3. Deep-dive: [CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md](CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md) (60 min)

### Objective: "Trace a Question Through Pipeline"
1. Follow: [GNN_INPUT_COMPLETE_EXAMPLE.md § Question #42](GNN_INPUT_COMPLETE_EXAMPLE.md) (complete walkthrough)
2. Reference: [CORRELATION_MATRICES_AND_GNN_INPUT.md § Pipeline Visualization](CORRELATION_MATRICES_AND_GNN_INPUT.md#5-complete-pipeline-visualization)

### Objective: "Code Review / Debugging"
1. Check: [CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md § Code Locations](CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md) (exact line numbers)
2. Reference: [CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md § Pseudo-code](CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md)
3. Compare: [CORRELATION_SANITIZATION_EXAMPLES.md § Examples](CORRELATION_SANITIZATION_EXAMPLES.md)

---

## 🔍 Key Concepts

### Correlation Matrix
- **What**: Pearson correlation computed between hidden dimensions
- **Size**: N×N where N = hidden dimension (1024 for Qwen)
- **Values**: ∈ [-1, 1] (standard Pearson correlation range)
- **Why**: Pattern of which dimensions co-activate encodes semantic information about truthfulness

### GNN Input
- **Graph Nodes**: 1024 (hidden dimensions become graph nodes)
- **Graph Edges**: ~630K (correlations ≠ 0 become edges)
- **Edge Weights**: Correlation values (-1 to +1) directly weight message passing
- **Task**: Classify question answer as True/False based on correlation structure

### Three Sanitization Layers
1. **Computation**: np.isfinite() validation → exclude entire question if NaN/Inf
2. **Data Loading**: np.nan_to_num() + clipping → fix corrupted saved data
3. **Runtime**: torch.nan_to_num() + clamping → prevent NaN propagation in model

### Message Passing in GCN
```
For each edge (source_dim → dest_dim) with correlation weight ρ:
  message = embedding[source_dim] * ρ
  destination_dim accumulates all such messages
  
Result: Dimensions with strong correlations have stronger influence
```

---

## 📊 Pipeline Statistics (from Analysis)

**TruthfulQA Hallucination Detection (Qwen2.5-0.5B, Layer 5)**

| Metric | Value |
|--------|-------|
| Total Questions | 5,915 |
| Excluded (NaN/Inf) | ~65 (1%) |
| Training Set | ~4,732 |
| Test Set | ~1,118 |
| Test Accuracy | 52.2% |
| Precision | 50.1% |
| Recall | 71.5% |
| F1-Score | 59.0% |

**Interpretation**: GCN trained on correlation patterns achieves ~52% accuracy (slight improvement over 50% random baseline) on binary true/false classification.

---

## 🔗 File Cross-References

### By Topic

**Q: Where are correlations computed?**
- Compute: [hallucination/compute_llm_network.py](hallucination/compute_llm_network.py#L244) 
- Explained in: [CORRELATION_MATRICES_AND_GNN_INPUT.md § 1.1](CORRELATION_MATRICES_AND_GNN_INPUT.md#11-two-types-of-correlation-matrices-computed)

**Q: How are correlations loaded and sanitized?**
- Code: [hallucination/dataset.py](hallucination/dataset.py#L71)
- Explained in: [CORRELATION_MATRICES_AND_GNN_INPUT.md § 2.2](CORRELATION_MATRICES_AND_GNN_INPUT.md#22-exact-input-sizes-to-gnn)
- Examples: [CORRELATION_SANITIZATION_EXAMPLES.md § Example 3](CORRELATION_SANITIZATION_EXAMPLES.md#example-3-sparse-data-edge-cases)

**Q: How does the GCN model work?**
- Architecture: [utils/probing_model.py](utils/probing_model.py#L80)
- Explained in: [CORRELATION_MATRICES_AND_GNN_INPUT.md § 3](CORRELATION_MATRICES_AND_GNN_INPUT.md#3-gnn-architecture)
- Example: [GNN_INPUT_COMPLETE_EXAMPLE.md § Step 11](GNN_INPUT_COMPLETE_EXAMPLE.md#step-11-gcn-forward-pass-model-processing)

**Q: How is NaN/Inf handled?**
- Computation check: [hallucination/compute_llm_network.py](hallucination/compute_llm_network.py#L246)
- Data loading: [hallucination/dataset.py](hallucination/dataset.py#L82)
- Model runtime: [utils/probing_model.py](utils/probing_model.py#L50)
- All explained in: [CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md](CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md)

---

## 💡 Implementation Summary

### Complete Data Flow

```
TruthfulQA Question
    ↓
LLM Forward Pass (Qwen2.5-0.5B)
    ↓
Extract Layer 5 Hidden States: (N_tokens, 1024)
    ↓
Compute Pearson Correlation: (1024, 1024)
    ↓
Validate (NaN/Inf check) → EXCLUDE if invalid ❌
    ↓
Threshold & Sparsify: edge_index [2, ~630K], edge_attr [~630K]
    ↓
Sanitize (np.nan_to_num + clip) → SAFE TENSORS ✓
    ↓
GNN Processing
    ├─ Embed nodes: [1024, 32]
    ├─ GCN convolution with correlation weights
    ├─ Global pooling: [1, 64]
    └─ FC layers: [1] logit
    ↓
Binary Classification: True (1) or False (0)
    ↓
Optimize with Cross-Entropy Loss
```

---

## 📝 File Sizes & Reading Time

| Document | Lines | Words | Time |
|----------|-------|-------|------|
| CORRELATION_SANITIZATION_README.md | 190 | 1,200 | 5 min |
| CORRELATION_MATRICES_AND_GNN_INPUT.md | 450 | 3,800 | 20 min |
| GNN_INPUT_COMPLETE_EXAMPLE.md | 520 | 4,500 | 25 min |
| CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md | 420 | 5,000 | 30 min |
| CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md | 350 | 2,500 | 15 min |
| CORRELATION_SANITIZATION_EXAMPLES.md | 480 | 4,200 | 25 min |
| **TOTAL** | **2,410** | **21,200** | **120 min** |

---

## 🎓 Key Takeaways

1. **Correlations as Features**: The model learns to distinguish true/false statements by analyzing how LLM hidden dimensions correlate with each other

2. **Graph Representation**: 1024 hidden dimensions become graph nodes; correlation coefficients become edge weights

3. **Message Passing**: Strong correlations → strong influence through the GCN; weak correlations → isolated nodes

4. **Robustness**: Three-layer sanitization ensures no NaN/Inf propagates through training/inference

5. **Modest Performance**: 52% accuracy suggests correlation patterns have limited signal for truthfulness (task is hard; baseline = 50%)

---

## 🚀 Next Steps

- **To understand architecture**: Start with [CORRELATION_MATRICES_AND_GNN_INPUT.md](CORRELATION_MATRICES_AND_GNN_INPUT.md)
- **To trace an example**: Follow [GNN_INPUT_COMPLETE_EXAMPLE.md](GNN_INPUT_COMPLETE_EXAMPLE.md)
- **To understand robustness**: Study [CORRELATION_SANITIZATION_EXAMPLES.md](CORRELATION_SANITIZATION_EXAMPLES.md)
- **For code review**: Reference [CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md](CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md)

---

**Created**: December 30, 2025  
**Project**: LLM Graph Probing - Hallucination Detection  
**Model**: Qwen2.5-0.5B  
**Dataset**: TruthfulQA (5,915 questions)
