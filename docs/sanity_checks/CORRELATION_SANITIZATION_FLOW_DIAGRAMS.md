# Correlation Matrix Sanitization - Code Flow Diagrams

## 1. COMPUTATION-TIME SANITIZATION FLOW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        run_corr() Function                              │
│                   (hallucination/compute_llm_network.py)                │
└─────────────────────────────────────────────────────────────────────────┘

Input: Hidden states from all transformer layers (batch of questions)
       │
       ├─ hidden_states_layer_average: [num_layers_avg, batch_size, seq_len]
       ├─ hidden_states: [num_selected_layers, batch_size, seq_len, hidden_dim]
       └─ attention_mask: [batch_size, seq_len]

                           ↓

    ┌─────────────────────────────────────┐
    │ For each question in batch:         │
    │  (Line 230-290)                     │
    └─────────────────────────────────────┘

                           ↓

    ┌─────────────────────────────────────────────────────────────┐
    │ VALIDATION STAGE 1: Pre-Computation Check (Line 239-243)   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ sentence_attention_mask = attention_mask[i]                │
    │ layer_avg_hidden = hidden_states_layer_avg[:, i,           │
    │                    sentence_attention_mask == 1]           │
    │                                                             │
    │ CHECK: size > 0 AND shape[1] >= 2 (at least 2 tokens)?    │
    │                                                             │
    │ IF YES ──────┐                                              │
    │ IF NO  ──────┼──→ excluded_count++                         │
    │             │    excluded_indices.append(question_idx)     │
    │             │    CONTINUE (skip to next question)          │
    └─────────────┼──────────────────────────────────────────────┘
                  │
                  ↓ (PASS validation)

    ┌─────────────────────────────────────────────────────────────┐
    │ COMPUTATION: Layer-Average Correlation (Line 244-245)      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ layer_average_corr = np.corrcoef(layer_avg_hidden)        │
    │                                                             │
    │ Input shape:  [num_layers_avg, num_valid_tokens]          │
    │ Output shape: [num_layers_avg, num_layers_avg]            │
    │ Output dtype: float64                                      │
    │                                                             │
    │ Possible NaN values if:                                    │
    │   - Any layer has zero variance (std=0)                    │
    │   - Numerical instability (large/small values)             │
    │   - Singular matrix                                        │
    └──────────────────────────────────────────────────────────────┘

                           ↓

    ┌─────────────────────────────────────────────────────────────┐
    │ VALIDATION STAGE 2: Post-Computation Check (Line 246-248)  │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ CHECK: np.isfinite(layer_average_corr).all()              │
    │                                                             │
    │ (checks ALL elements are not NaN and not Inf)             │
    │                                                             │
    │ IF YES ──────┐                                              │
    │ IF NO  ──────┼──→ excluded_count++                         │
    │             │    excluded_indices.append(question_idx)     │
    │             │    CONTINUE (skip to next question)          │
    └─────────────┼──────────────────────────────────────────────┘
                  │
                  ↓ (PASS validation)

    ┌─────────────────────────────────────────────────────────────┐
    │ VALIDATION STAGE 3: Per-Layer Check (Line 256-266)         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ FOR each layer in layer_list (5, 6, 7, 8, 9, 10, 11):    │
    │   │                                                        │
    │   ├─ Extract tokens: layer_hidden = hidden_states[j,i,    │
    │   │                  sentence_attention_mask==1].T        │
    │   │                                                        │
    │   ├─ CHECK: shape[1] >= 2?                                │
    │   │   IF NO → set invalid=True, break                     │
    │   │                                                        │
    │   ├─ Compute: corr = np.corrcoef(layer_hidden)            │
    │   │                                                        │
    │   ├─ CHECK: np.isfinite(corr).all()?                      │
    │   │   IF NO → set invalid=True, break                     │
    │   │                                                        │
    │   └─ PASS → store (layer_idx, corr)                       │
    │                                                             │
    │ IF invalid flag set:                                       │
    │   excluded_count++                                         │
    │   excluded_indices.append(question_idx)                    │
    │   CONTINUE (skip to next question)                         │
    └─────────────────────────────────────────────────────────────┘

                           ↓ (PASS all checks)

    ┌─────────────────────────────────────────────────────────────┐
    │ VALIDATION STAGE 4: Combined Layers (if aggregate_layers)  │
    │                                   (Line 269-278)           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ combined_states = np.concatenate(per_layer_states, axis=0)│
    │ combined_corr = np.corrcoef(combined_states)              │
    │                                                             │
    │ CHECK: np.isfinite(combined_corr).all()?                  │
    │                                                             │
    │ IF NO → excluded_count++, excluded_indices.append()        │
    │         CONTINUE (skip)                                    │
    └─────────────────────────────────────────────────────────────┘

                           ↓ (ALL validations PASS)

    ┌─────────────────────────────────────────────────────────────┐
    │ WRITE OUTPUT FILES (Line 283-350)                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │ CREATE: {output_dir}/{question_idx}/                      │
    │                                                             │
    │ SAVE:                                                      │
    │  ├─ label.npy                                              │
    │  ├─ layer_average_corr.npy                                │
    │  ├─ layer_average_activation.npy                          │
    │  ├─ layer_average_degree.npy                              │
    │  ├─ layer_5_corr.npy, layer_5_corr_thresh_05.npy, ...    │
    │  ├─ layer_6_corr.npy, ...                                 │
    │  └─ ... (all layers and combined)                          │
    │                                                             │
    │ processed_count++                                          │
    └─────────────────────────────────────────────────────────────┘

                           ↓

        ┌────────────────────────────────────┐
        │ SUMMARY AFTER ALL QUESTIONS        │
        ├────────────────────────────────────┤
        │ processed: X questions             │
        │ skipped: Y (already done)          │
        │ excluded_nan: Z (invalid corr)     │
        │                                    │
        │ Write files:                       │
        │ exclusions_worker_0.txt            │
        │ summary_worker_0.json              │
        └────────────────────────────────────┘
```

---

## 2. DATA-LOADING SANITIZATION FLOW

```
┌────────────────────────────────────────────────────────────────────┐
│        TruthfulQADataset._load_data() Method                       │
│              (hallucination/dataset.py, Line 65-85)                │
└────────────────────────────────────────────────────────────────────┘

Input: question_idx (integer question ID)
       │
       ├─ from_sparse_data: Boolean (load from sparse or dense files)
       └─ network_density: float (e.g., 0.05 for 5%)

                      ↓

    ┌──────────────────────────────────────┐
    │ Branch: Dense or Sparse Loading?     │
    └──────────────────────────────────────┘
           │
      ┌────┴─────┐
      │           │
      ↓           ↓

╔════════════════════════════════════════════════════════════════════╗
║                    DENSE LOADING PATH                              ║
║                   (from_sparse_data=False)                         ║
╚════════════════════════════════════════════════════════════════════╝

    Line 68-78:
    
    adj = np.load(data_path / self.dense_filename)
    # adj shape: [num_tokens, num_tokens]
    # adj dtype: float64
    # adj values: Pearson correlation [-1, 1] or NaN/Inf
    
                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 1: np.nan_to_num() (Line 71)        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ adj = np.nan_to_num(adj,                               │
    │         nan=0.0,      # NaN → 0.0                      │
    │         posinf=0.0,   # +Inf → 0.0                     │
    │         neginf=0.0)   # -Inf → 0.0                     │
    │                                                         │
    │ Example transformation:                                 │
    │ Before: [[ 1.0 , 0.5 , nan  , inf ]                   │
    │          [ 0.5 , 1.0 , -0.3 , -inf]                   │
    │          [ nan , -0.3, 1.0  , 0.2 ]                   │
    │          [ inf , -inf, 0.2  , 1.0 ]]                  │
    │                                                         │
    │ After:  [[ 1.0 , 0.5 , 0.0 , 0.0 ]                    │
    │          [ 0.5 , 1.0 , -0.3, 0.0 ]                    │
    │          [ 0.0 , -0.3, 1.0 , 0.2 ]                    │
    │          [ 0.0 , 0.0 , 0.2 , 1.0 ]]                   │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 2: Thresholding (Line 72-74)        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ percentile_threshold = network_density * 100  # 5      │
    │ threshold = np.percentile(np.abs(adj), 95)  # 95th %   │
    │                                                         │
    │ adj[np.abs(adj) < threshold] = 0                       │
    │                                                         │
    │ Purpose: Keep only top 5% of correlations by magnitude │
    │                                                         │
    │ Example:                                                │
    │ Top 95% correlations by abs value: [0.8, -0.7, 0.6]   │
    │ threshold = 0.6 (would be ~95th percentile)            │
    │                                                         │
    │ After: Keep 0.8, -0.7, 0.6; zero out smaller values   │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 3: Diagonal Reset (Line 75)         │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ np.fill_diagonal(adj, 0)                               │
    │                                                         │
    │ Purpose: Zero out main diagonal (self-loops)           │
    │                                                         │
    │ Before: [[ 1.0 , 0.8 , 0.0 ]                           │
    │          [ 0.8 , 1.0 , 0.0 ]                           │
    │          [ 0.0 , 0.0 , 1.0 ]]                          │
    │                                                         │
    │ After:  [[ 0.0 , 0.8 , 0.0 ]                           │
    │          [ 0.8 , 0.0 , 0.0 ]                           │
    │          [ 0.0 , 0.0 , 0.0 ]]                          │
    │                                                         │
    │ Note: Will be set to 1.0 during dense_to_sparse        │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ Convert to Sparse (Line 76-77)                         │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ adj = torch.from_numpy(adj).float()                    │
    │ edge_index, edge_attr = dense_to_sparse(adj)           │
    │                                                         │
    │ Result:                                                 │
    │  edge_index: [2, num_edges] (source, dest indices)    │
    │  edge_attr: [num_edges] (correlation weights)          │
    └─────────────────────────────────────────────────────────┘

                      ↓

╔════════════════════════════════════════════════════════════════════╗
║                    SPARSE LOADING PATH                             ║
║                   (from_sparse_data=True)                          ║
╚════════════════════════════════════════════════════════════════════╝

    Line 79-84:
    
    edge_index = torch.from_numpy(np.load(..._edge_index.npy)).long()
    edge_attr_np = np.load(..._edge_attr.npy).astype(np.float32)
    
                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 1: np.nan_to_num() (Line 82)        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ edge_attr_np = np.nan_to_num(edge_attr_np,            │
    │         nan=0.0,      # NaN → 0.0                      │
    │         posinf=0.0,   # +Inf → 0.0                     │
    │         neginf=0.0)   # -Inf → 0.0                     │
    │                                                         │
    │ Example sparse weights before:                          │
    │ [0.8, -0.6, nan, inf, 0.3, -inf, 0.2]                 │
    │                                                         │
    │ After:                                                  │
    │ [0.8, -0.6, 0.0, 0.0, 0.3, 0.0, 0.2]                  │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 2: Clamp to [-1, 1] (Line 83)       │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ edge_attr_np = np.clip(edge_attr_np, -1.0, 1.0)       │
    │                                                         │
    │ Purpose: Ensure correlations in valid range [-1, 1]   │
    │                                                         │
    │ Example transformation:                                 │
    │ Before: [0.8, -0.6, 0.0, 1.5, 0.3, -1.2, 0.2]         │
    │ After:  [0.8, -0.6, 0.0, 1.0, 0.3, -1.0, 0.2]         │
    │                                                         │
    │ Why clamp? Correlations theoretically in [-1, 1];      │
    │ values outside indicate numerical errors               │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ Convert to Tensor (Line 84)                            │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ edge_attr = torch.from_numpy(edge_attr_np).float()     │
    │                                                         │
    │ Result: Safe PyTorch tensor ready for GPU              │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ Create Data Object (Line 87-88)                        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ x = torch.arange(num_nodes)  # Node IDs               │
    │ return Data(x=x,                                       │
    │            edge_index=edge_index,                      │
    │            edge_attr=edge_attr,                        │
    │            y=y)                                        │
    │                                                         │
    │ Output: PyTorch Geometric Data object                  │
    │         Ready for DataLoader → GNN                     │
    └─────────────────────────────────────────────────────────┘
```

---

## 3. GNN FORWARD-PASS SANITIZATION FLOW

```
┌────────────────────────────────────────────────────────────────────┐
│         SimpleGCNConv.forward() Method                             │
│              (utils/probing_model.py, Line 44-71)                  │
└────────────────────────────────────────────────────────────────────┘

Input: x [N, C]                    (node features)
       edge_index [2, E]           (edge indices)
       edge_weight [E]             (correlation weights)

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 1: Input Validation (Line 50)       │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ messages = torch.nan_to_num(x[src])                    │
    │                                                         │
    │ Where: src = edge_index[1]  (source node indices)      │
    │                                                         │
    │ Purpose: Clean node features before aggregation        │
    │                                                         │
    │ Example:                                                │
    │ x[src] before:  [[0.5, nan, 0.3],                      │
    │                  [1.2, inf, -0.1]]                     │
    │ messages after: [[0.5, 0.0, 0.3],                      │
    │                  [1.2, 0.0, -0.1]]                     │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 2: Edge Weight Processing (Line 51-55)
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ IF edge_weight is not None:                            │
    │   │                                                    │
    │   ├─ ew = torch.nan_to_num(edge_weight,               │
    │   │         nan=0.0, posinf=0.0, neginf=0.0)          │
    │   │                                                    │
    │   │ Purpose: Clean edge weights                        │
    │   │ Before: [0.8, nan, -0.5, inf, 0.2]               │
    │   │ After:  [0.8, 0.0, -0.5, 0.0, 0.2]               │
    │   │                                                    │
    │   ├─ ew = ew.clamp(min=-1.0, max=1.0)                 │
    │   │                                                    │
    │   │ Purpose: Bound correlation to [-1, 1]             │
    │   │ Before: [0.8, 0.0, -0.5, 0.0, 1.5]               │
    │   │ After:  [0.8, 0.0, -0.5, 0.0, 1.0]               │
    │   │                                                    │
    │   └─ messages = messages * ew.view(-1, 1)             │
    │                                                         │
    │      Multiply node features by edge weights            │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ AGGREGATION: Index Add (Line 56-57)                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ out = torch.zeros_like(x)                              │
    │ out.index_add_(0, dst, messages)                       │
    │                                                         │
    │ Where: dst = edge_index[0]  (destination node indices) │
    │                                                         │
    │ Purpose: Aggregate weighted messages by destination    │
    │ Effect: Sum incoming messages for each node            │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ NORMALIZATION: Degree Normalization (Line 58-60)       │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ deg = torch.zeros((N, 1))                              │
    │ deg.index_add_(0, dst, ones)                           │
    │ out = out / deg.clamp(min=1)                           │
    │                                                         │
    │ Purpose: Divide by in-degree (with safeguard)          │
    │ Effect: Average aggregation instead of sum             │
    │                                                         │
    │ Clamp to 1 prevents division by zero                   │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION STEP 3: Output Cleaning (Line 60)        │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ out = torch.nan_to_num(out)                            │
    │                                                         │
    │ Purpose: Clean output after aggregation                │
    │ Why: Degree norm might introduce numerical issues      │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ FINAL: Linear Transformation (Line 61)                │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ return self.linear(out)                                │
    │                                                         │
    │ Output: [N, out_channels]                              │
    │         Clean, sanitized node embeddings               │
    └─────────────────────────────────────────────────────────┘
```

---

## 4. POOLING SANITIZATION FLOW

```
┌────────────────────────────────────────────────────────────────────┐
│         global_mean_pool_torch() Function                          │
│              (utils/probing_model.py, Line 7-18)                   │
└────────────────────────────────────────────────────────────────────┘

Input: x [N, C]           (all node features)
       batch [N]          (graph indices for each node)

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ SANITIZATION: Input Cleaning (Line 7)                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ x = torch.nan_to_num(x)                                │
    │                                                         │
    │ Purpose: Clean node features before pooling            │
    │ Before: [[0.5, nan, 0.3],                              │
    │          [inf, 0.1, -0.2]]                             │
    │ After:  [[0.5, 0.0, 0.3],                              │
    │          [0.0, 0.1, -0.2]]                             │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ AGGREGATION: Sum by Graph (Line 13)                   │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ sums.index_add_(0, batch, x)                           │
    │ counts.index_add_(0, batch, torch.ones(...))           │
    │                                                         │
    │ Result: Per-graph sums and per-graph node counts       │
    └─────────────────────────────────────────────────────────┘

                      ↓

    ┌─────────────────────────────────────────────────────────┐
    │ NORMALIZATION: Mean Computation (Line 14)             │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │ return sums / counts.clamp(min=1)                      │
    │                                                         │
    │ Purpose: Compute mean pooling (sum / count)            │
    │ Clamp prevents division by zero                        │
    │                                                         │
    │ Output: [G, C] (per-graph mean embedding)              │
    └─────────────────────────────────────────────────────────┘
```

---

## 5. COMPLETE QUESTION-TO-OUTPUT PIPELINE

```
Question ID: ~500  ("Is Earth flat?")
│
├─ EXTRACTION FROM LLM
│  └─ Get hidden states for all 12 layers
│     └─ Apply attention mask (keep valid tokens only)
│
├─ COMPUTATION-TIME SANITIZATION ◄─── FIRST DEFENSE
│  ├─ Check minimum 2 tokens ✓
│  ├─ Compute correlations
│  ├─ Validate for NaN/Inf → Found Inf in Layer 8!
│  └─ EXCLUDED: exclusions_worker_0.txt contains "500"
│     No output files written
│
├─ WOULD-BE DATA LOADING (SKIPPED - No files exist)
│
├─ WOULD-BE GNN FORWARD-PASS (SKIPPED)
│
└─ RESULT:
   - Question 500 NOT in processed set
   - Not in training or test datasets
   - In exclusions_worker_0.txt
   - Statistics show: excluded_nan: +1
```

**Alternative case: Question PASSES computation**

```
Question ID: ~250  ("Is Earth flat?")
│
├─ EXTRACTION FROM LLM
│  └─ Valid hidden states ✓
│
├─ COMPUTATION-TIME SANITIZATION ✓ PASS ALL CHECKS
│  └─ All correlations finite, all layers valid
│     Output files written:
│     - 250/label.npy
│     - 250/layer_5_corr.npy
│     - 250/layer_6_corr.npy
│     - ... (all layers)
│
├─ DATA LOADING ◄─── SECOND DEFENSE
│  ├─ Load 250/layer_5_corr.npy (from disk)
│  ├─ np.nan_to_num() (catch any disk corruption)
│  ├─ Clamp edge weights to [-1, 1]
│  └─ Return clean Data object
│
├─ GNN FORWARD-PASS ◄─── THIRD DEFENSE
│  ├─ torch.nan_to_num(node_features)
│  ├─ torch.nan_to_num(edge_weights)
│  ├─ torch.nan_to_num(aggregated_output)
│  └─ Return clean embeddings
│
└─ RESULT:
   - Question 250 in training set
   - Safe for GNN processing
   - Triple-sanitized at each stage
```
