# Test Performance Issue Analysis - Summary

## Problem Statement
Training loss is decreasing nicely (0.9062 at epoch 35), but test performance is at **random chance level**:
- **Test Accuracy: 51.23%** (expected ≥ 70-80% for a working model)
- **Precision: 0.4340**
- **Recall: 0.5526**
- **F1: 0.4862**

**Confusion Matrix (epoch 35):**
```
[[333 356]
 [221 273]]
```
Format: `[[TN, FP], [FN, TP]]`

This suggests **severe overfitting** or a fundamental data/label issue.

---

## Data Flow Analysis

### 1. **Label Assignment & Storage**

**Source:** `hallucination/construct_dataset.py` (labels = 1 for truthful, 0 for hallucinated)
```python
# TruthfulQA dataset
records.append({"label": 1})  # True answers → class 1 (truthful)
records.append({"label": 0})  # False answers → class 0 (hallucinated)

# HALUEval dataset
records.append({"label": 1})  # Right answers → class 1
records.append({"label": 0})  # Hallucinated answers → class 0
```

**Storage in Step 2:** `hallucination/compute_llm_network.py` (lines 323)
```python
np.save(f"{p_dir_name}/label.npy", labels[i])
```
- Labels are read from CSV at line 114: `original_labels = df["label"].tolist()[rank::num_producers]`
- Stored as numpy scalar (single integer value per sample)

### 2. **Data Splitting (Train/Test)**

**Location:** `hallucination/dataset.py` (lines 18-26)
```python
def prepare_data(dataset_filename, test_set_ratio, seed):
    indices = list(range(len(data["question"])))
    test_size = int(len(indices) * test_set_ratio)
    train_size = len(indices) - test_size
    if seed is not None:
        train_data_split, test_data_split = random_split(indices, [train_size, test_size], 
                                                          generator=torch.Generator().manual_seed(seed))
    return train_data_split, test_data_split
```

**Split Details (from earlier verification):**
- Total: 5,915 samples
- Train: 4,732 samples (80%)
- Test: 1,183 samples (20%)
- **Train indices:** [0, 1, 2, ..., 4731]
- **Test indices:** [4732, 4733, ..., 5914]
- **Status:** ✓ No overlap, reproducible with seed, identical across all layers

### 3. **Input Features to Training**

**GCN Probe Input (graph features):**
- **Feature type:** Functional connectivity matrices (correlation matrices)
- **Shape per sample:** 768×768 (GPT-2 layer 5 has 768 neurons)
- **Thresholding:** 5% network density (retains top 5% strongest correlations)
- **Edge attributes:** Correlation values between neurons

**Feature loading code** (`hallucination/dataset.py` lines 48-75):
```python
def _load_data(self, idx):
    question_idx = self.network_indices[idx]
    
    if not self.from_sparse_data:
        adj = np.load(f"layer_{llm_layer}_corr.npy")  # Load correlation matrix
        # Thresholding: keep only top 5% by absolute value
        threshold = np.percentile(np.abs(adj), 100 - 5)
        adj[np.abs(adj) < threshold] = 0
        # Convert to graph representation
        edge_index, edge_attr = dense_to_sparse(adj)
    else:
        edge_index = np.load(f"layer_{llm_layer}_sparse_{density_tag}_edge_index.npy")
        edge_attr = np.load(f"layer_{llm_layer}_sparse_{density_tag}_edge_attr.npy")
    
    # Load label
    y = torch.from_numpy(np.load(f"label.npy")).long()
    
    # Node features: identity features (0, 1, 2, ..., 767)
    x = torch.arange(num_nodes)  # [0, 1, ..., 767]
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

**Key observations:**
1. Node features `x` are simply **indices (0 to 767)** - NO semantic information
2. Graph structure is from **correlation thresholding**
3. Labels are properly loaded

### 4. **Training Process**

**Location:** `hallucination/train.py` (lines 114-130)

```python
for epoch in range(FLAGS.num_epochs):
    model.train()
    for data in train_data_loader:
        optimizer.zero_grad()
        loss, batch_size = get_loss_batch_size(data)  # Forward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(...)  # Optional
        optimizer.step()
        
    # Test on full test set
    accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, ...)
```

**Training hyperparameters (from config):**
- Batch size: 16
- Eval batch size: 32
- Learning rate: 0.0001 (⚠️ **Very low!**)
- Gradient clipping: 10 (permissive)
- Label smoothing: 0.0 (disabled)
- Num epochs: 100
- Early stopping patience: 50

### 5. **Test Function**

**Location:** `hallucination/utils.py` (lines 103-135)

```python
def test_fn(model, data_loader, device, num_layers):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # No gradient computation ✓
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=-1)  # argmax to get predicted class
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, 
                                                               average='binary')
    cm = confusion_matrix(all_labels, all_preds)  # [[TN, FP], [FN, TP]]
```

---

## Confusion Matrix Interpretation (epoch 35)

```
[[333 356]
 [221 273]]
```

| | Predicted 0 (Hallucinated) | Predicted 1 (Truthful) |
|---|---|---|
| **Actual 0 (Hallucinated)** | 333 (TN) | 356 (FP) |
| **Actual 1 (Truthful)** | 221 (FN) | 273 (TP) |

**Metrics derivation:**
- Accuracy = (333 + 273) / 1183 = 606 / 1183 = **51.23%**
- Precision (for class 1) = 273 / (273 + 356) = **43.4%**
- Recall (for class 1) = 273 / (273 + 221) = **55.3%**
- F1 = 2 × (0.434 × 0.553) / (0.434 + 0.553) = **48.6%**

**Interpretation:**
- Model predicts class 0 (hallucinated) 589 times, class 1 (truthful) 594 times (nearly balanced)
- For actual class 0 samples (689 total): 333 correct, 356 wrong (48% accuracy)
- For actual class 1 samples (494 total): 273 correct, 221 wrong (55% accuracy)
- **Performance is essentially random** (50% baseline for balanced classes)

---

## Potential Issues Identified

### ⚠️ **Critical Issue 1: Weak Node Features**
- Node features are just `x = torch.arange(num_nodes)` ([0, 1, 2, ..., 767])
- These are **identity indices with no semantic meaning**
- GCN can only learn from graph structure (edges), not from node features
- **Problem:** If network structure alone isn't discriminative enough, model cannot learn

### ⚠️ **Issue 2: Very Low Learning Rate (0.0001)**
- Config shows `learning_rate: 0.0001`
- Reference implementation uses 0.001 (10x higher)
- **Problem:** May cause:
  - Extremely slow convergence
  - Model parameters barely updated per step
  - Gets stuck in local minima
  - Training loss = 0.9062 suggests loss isn't decreasing enough

### ⚠️ **Issue 3: Network Density Threshold (5%)**
- Only 5% of edges retained after thresholding
- ~5% of 768×768 = ~29,500 edges out of 589,824 possible
- **Problem:** Graph may be too sparse, losing important connectivity information
- **Note:** Thresholding is identical for train/test (no data leakage here)

### ⚠️ **Issue 4: Possible Label Mixing or Data Corruption**
- Need to verify:
  1. Are labels being loaded correctly from disk?
  2. Are train/test labels properly separated?
  3. Is label distribution balanced in train/test splits?
  4. Are there any NaN/Inf values corrupting features?

### ⚠️ **Issue 5: Model Architecture**
- GCN with 3 layers, 32 hidden channels
- Output: 2 classes
- **Unknown:** Whether model capacity is sufficient for the task

---

## Data Verification Checklist (Not Yet Verified)

- [ ] **Label distribution:**
  - Train set: How many class 0 vs class 1?
  - Test set: How many class 0 vs class 1?
  - Are proportions ~50/50?

- [ ] **Feature integrity:**
  - Are correlation matrices being computed correctly?
  - Are there NaN/Inf values in the matrices?
  - After 5% thresholding, are edges actually present?
  - Do graphs remain connected or are they fragmented?

- [ ] **Data loading:**
  - When loading test samples, are indices [4732-5914] being used?
  - Are test samples **never** seen during training?
  - Are the same train/test splits used across layers?

- [ ] **Model predictions:**
  - Is the model outputting sensible logits?
  - Are both output classes being predicted (not always one class)?
  - Are confidence scores reasonable (not all 0.5)?

- [ ] **Training dynamics:**
  - Does training loss actually decrease significantly?
  - Does validation F1 improve during training?
  - When does early stopping trigger?

---

## Likely Root Causes (Ranked by Probability)

1. **Very low learning rate (0.0001)** → Model barely learns
   - Learning rate is 10x lower than reference
   - At 0.0001, 47,320 training iterations might not be enough

2. **Weak node features** → GCN has no meaningful node information
   - Only graph structure available
   - If correlation patterns aren't discriminative, model fails

3. **5% network density too sparse** → Graph fragmented, information lost
   - Only 5% of edges retained
   - Important correlations might be pruned away

4. **Data-label mismatch or corruption** → Labels don't match features
   - Labels mixed up or corrupted during saving
   - Test set accidentally used during training somehow

5. **Model capacity insufficient** → Architecture too small
   - GCN(3 layers, 32 channels) → perhaps inadequate for 768-node graphs

---

## Next Steps (Recommended Investigations - Do Not Execute Yet)

1. **Verify label distribution:**
   - Print distribution of labels in train vs test sets
   - Confirm they match original dataset distribution (~50/50)

2. **Verify data loading:**
   - Log which sample indices are being loaded
   - Confirm train/test index ranges are correct

3. **Check feature statistics:**
   - After thresholding, how many edges per sample?
   - Are correlation matrices reasonable?
   - Any NaN/Inf values?

4. **Try increasing learning rate:**
   - Test with lr=0.001 (reference value)
   - See if test performance improves

5. **Check model output statistics:**
   - Log logit values before softmax
   - Log predicted class distribution

6. **Visualize confusion:**
   - Are both classes being predicted?
   - Or is model stuck predicting one class?

---

## Summary

**Current State:**
- ✓ Train/test split: Correct (4732/1183, no overlap)
- ✓ Data pipeline: Appears correct structurally
- ✓ Label loading: Appears correct structurally
- ✗ Test performance: At random chance level (51% accuracy)
- ✗ Training loss decreasing, but test metrics not improving

**Primary Suspects:**
1. Very low learning rate (0.0001) → too slow convergence
2. Weak node features (just indices) → insufficient input signal
3. 5% network density → graph too sparse
4. Possible data corruption or label mismatch

**Status:** Ready for detailed diagnostic investigation once user approves.
