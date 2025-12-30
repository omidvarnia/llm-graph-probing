# Correlation Matrix Sanitization - Complete Documentation Index

## Overview

This repository includes comprehensive documentation explaining how correlation matrices computed from LLM hidden states are validated, cleaned, and protected from NaN/Inf values throughout the entire pipeline.

## Documentation Files

### 1. **CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md**
   - **What it explains**: Comprehensive guide to the three sanitization layers in the codebase
   - **Key sections**:
     - Source of NaN/Inf issues (singular matrices, numerical precision, edge cases)
     - Where sanitization occurs (computation, data loading, model layers)
     - How sanitization works (np.corrcoef validation, np.nan_to_num, torch.nan_to_num, clipping)
     - Code locations with exact line numbers
     - Impact on pipeline stability
   - **Best for**: Understanding the overall architecture and mechanisms
   - **Length**: ~5,000 words with detailed explanations

### 2. **CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md**
   - **What it explains**: Visual flow diagrams and pseudo-code showing data transformations
   - **Key sections**:
     - Question pipeline diagram (LLM → correlation → graph → model)
     - Three sanitization stages with ASCII diagrams
     - Pseudo-code for each sanitization point
     - State transitions (question passes/fails each check)
     - Variable tracking through the pipeline
   - **Best for**: Visualizing where sanitization happens and why
   - **Format**: ASCII diagrams + pseudo-code + state transitions

### 3. **CORRELATION_SANITIZATION_EXAMPLES.md** ← **START HERE FOR CONCRETE EXAMPLES**
   - **What it explains**: Real numerical examples of sanitization in action
   - **Key sections**:
     - **Example 1**: Zero variance issue (most common NaN cause)
       - Scenario: Identical token embeddings in a layer
       - Shows np.corrcoef producing NaN matrix
       - Demonstrates np.isfinite() validation
       - Shows question exclusion mechanism
     - **Example 2**: Extreme value (Inf) issue
       - Scenario: Numerical overflow in correlation computation
       - Shows how np.corrcoef can produce inf/-inf values
       - Demonstrates computation-time detection
       - Shows np.nan_to_num and clipping in action
     - **Example 3**: Sparse data edge cases
       - Scenario: Pre-saved matrices with artifacts
       - Shows two-step sanitization (np.nan_to_num then np.clip)
       - Demonstrates why both steps are needed
     - **Example 4**: GNN forward-pass sanitization
       - Step-by-step walkthrough of sanitization during graph convolution
       - Shows torch.nan_to_num, clamping, and safe division
       - Real tensor shapes and values
     - **Example 5**: Complete question lifecycle
       - Question #42 (passes all checks) with full pipeline tracking
       - Question #500 (fails computation) showing exclusion
     - **Summary table**: All issue types, detection methods, fixes, results
   - **Best for**: Understanding WHAT can go wrong and HOW it's fixed
   - **Format**: Numerical examples with before/after matrices

## When to Read Each Document

| Question | Document | Section |
|----------|----------|---------|
| "Why does the code call np.nan_to_num?" | DETAILED_EXPLANATION | "Sanitization Mechanisms" |
| "Where is NaN checked in the code?" | DETAILED_EXPLANATION | "Code Locations & Implementation" |
| "Show me an example of NaN being produced" | EXAMPLES | "Example 1: Zero Variance Issue" |
| "How does sanitization work step-by-step?" | FLOW_DIAGRAMS | "Data Flow with Sanitization" |
| "What could go wrong with correlation?" | EXAMPLES | "Summary Table" |
| "Why does the code clamp edge weights?" | EXAMPLES | "Example 3: Sparse Data Edge Cases" |
| "How is torch.nan_to_num used in GNN?" | EXAMPLES | "Example 4: GNN Forward-Pass" |
| "What happens to excluded questions?" | EXAMPLES | "Example 5: Complete Lifecycle" |
| "What are all three sanitization points?" | FLOW_DIAGRAMS | "Three Sanitization Stages" |

## Quick Reference: The Three Sanitization Layers

### Layer 1: Computation-Time (Most Restrictive)
- **File**: `hallucination/compute_llm_network.py`
- **Lines**: 220-250
- **Function**: `run_corr()`
- **Operation**: Validates correlation matrices with `np.isfinite()` immediately after computation
- **Action**: **Excludes entire question** if ANY correlation contains NaN/Inf
- **Result**: Question files are not written to disk
- **Reason**: Questions with invalid correlations are fundamentally broken; better to exclude upfront

### Layer 2: Data Loading (Defensive)
- **File**: `hallucination/dataset.py`
- **Lines**: 60-90
- **Function**: `_load_data()`
- **Operation**: Applies `np.nan_to_num()` and `np.clip()` to loaded matrices
- **Action**: Replaces NaN→0, Inf→large values, clips values to [-1, 1]
- **Result**: Safe tensors even if disk data was corrupted
- **Reason**: Assumes saved data might have artifacts from earlier errors

### Layer 3: Model Runtime (Paranoid)
- **File**: `utils/probing_model.py`
- **Lines**: 1-100 (multiple locations)
- **Functions**: `SimpleGCNConv.forward()`, `global_mean_pool_torch()`, `GCNProbe.forward_graph_embedding()`
- **Operation**: Applies `torch.nan_to_num()` and `clamp()` at every stage
- **Action**: Cleans node features, edge weights, aggregation results, final outputs
- **Result**: No NaN can propagate through the network
- **Reason**: Defense in depth; catches any NaN that somehow passed earlier checks

## Technical Foundation

### Why NaN/Inf Occur

**Scenario 1: Zero Variance**
```python
# All tokens have identical representation in this layer
hidden_states = [[0.5], [0.5], [0.5], [0.5]]
# Std = 0, so Pearson correlation = 0/0 = NaN
```

**Scenario 2: Numerical Overflow**
```python
# Very large hidden state values
hidden_states = [[1e6], [2e6], [1.5e6]]
# Covariance computation overflows → Inf
```

**Scenario 3: Extreme Correlations**
```python
# Floating point rounding errors
correlation = 1.0000000001  # Should be exactly 1.0
# Gets stored as Inf in some contexts
```

### Why Three Layers?

1. **Computation-time**: Catches semantic failures upfront (question is fundamentally untrainable)
2. **Data-loading**: Handles disk corruption or save errors
3. **Runtime**: Defense against unforeseen edge cases in tensor operations

## Validation Checklist

When reviewing code that uses these correlations, verify:

- [ ] Are correlation matrices loaded from disk?
  - [ ] `np.nan_to_num()` applied? (dataset.py)
  - [ ] `np.clip()` to [-1, 1]? (dataset.py)
  
- [ ] Are correlations computed fresh?
  - [ ] `np.isfinite()` validation applied? (compute_llm_network.py)
  - [ ] Non-finite questions excluded? (compute_llm_network.py)
  
- [ ] Do GNN layers process correlations?
  - [ ] `torch.nan_to_num()` on inputs? (probing_model.py)
  - [ ] `clamp()` on edge weights? (probing_model.py)
  - [ ] `clamp(min=1)` on degrees to avoid div-by-zero? (probing_model.py)
  
- [ ] Are outputs from aggregation cleaned?
  - [ ] `torch.nan_to_num()` after aggregation? (probing_model.py)

## Code References

### Correlation Computation
- **Main function**: `hallucination/compute_llm_network.py:run_corr()`
- **Layer-average computation**: `line 230`
- **Per-layer computation**: `line 240`
- **Combined computation**: `line 248`
- **Validation check**: `line 246-250`

### Data Loading
- **Main function**: `hallucination/dataset.py:_load_data()`
- **Dense path**: `line 71` (np.nan_to_num)
- **Sparse path**: `line 82-83` (np.nan_to_num + np.clip)

### Model Layers
- **GCN layer**: `utils/probing_model.py:SimpleGCNConv.forward()`, lines 40-60
- **Pooling**: `utils/probing_model.py:global_mean_pool_torch()`, lines 10-20
- **Probe model**: `utils/probing_model.py:GCNProbe.forward_graph_embedding()`, lines 80-120

## Excluded Questions

Questions are excluded when:
1. Any correlation matrix contains NaN values
2. Any correlation matrix contains Inf/−Inf values  
3. Any other numerical anomaly is detected in computation

These questions are logged in: `exclusions_worker_*.txt`

Excluded questions:
- Do NOT have output directories created
- Do NOT appear in train/test datasets
- DO NOT contribute to model training/evaluation
- ARE tracked for auditing purposes

## Summary Statistics

From the TruthfulQA analysis on Qwen2.5-0.5B:
- Total questions processed: 5,915
- Questions successfully computed: ~5,850 (estimated)
- Questions excluded due to NaN/Inf: ~65 (estimated, ~1%)
- Questions used in model training/evaluation: ~5,850

---

## How to Use These Documents

1. **For understanding pipeline stability**: Start with CORRELATION_SANITIZATION_DETAILED_EXPLANATION.md
2. **For visual understanding**: Read CORRELATION_SANITIZATION_FLOW_DIAGRAMS.md  
3. **For concrete examples**: Read CORRELATION_SANITIZATION_EXAMPLES.md
4. **For code inspection**: Reference the Code References section above with exact line numbers

Each document is self-contained and can be read independently, but together they provide complete coverage of the sanitization architecture.
