# Pipeline Differences: Current Repo vs. Reference Package

## Overview
This report contrasts the hallucination-detection pipeline in the **current repo** with the **reference package** (`llm-graph-probing_reference_do_not_change`) from start to finish.

## High-Level Summary
- **Current pipeline** adds strong numerical sanitization, richer logging/artifacts, optional layer aggregation, and custom GCN ops that avoid torch_scatter/torch_sparse by using pure PyTorch aggregation with explicit clamping and NaN guards.
- **Reference pipeline** is simpler: no NaN/Inf validation, relies on torch_geometric GCNConv/pooling, and writes outputs without pre-checks or diagnostics.

## Stage-by-Stage Differences

### 1) Configuration & Paths
- **Current**: Paths derive from `MAIN_DIR` env; model names are sanitized for filesystem safety; supports `aggregate_layers`, density tags with zero-padding, and resume/skip checks.
- **Reference**: Paths are local (`data/hallucination/...`); model names used verbatim; no layer aggregation; sparse density encoded as raw float in filenames; no completeness checks.

### 2) LLM Forward Pass
- **Current**: Converts requested layers to zero-based indices with fallback to last layer if invalid; optional random hidden states for debugging; batches logged with progress labels.
- **Reference**: Uses provided `layer_list` directly (no zero-based adjustment); no fallback; simpler progress logging.

### 3) Correlation Computation & Validation
- **Current**: Three-tier validation before writing files—(a) layer-average corr, (b) per-layer corr, (c) optional combined corr; skips entire question if any NaN/Inf or <2 valid tokens; collects per-class stats; saves pre/post-threshold PNGs; optional combined-layer corr when `aggregate_layers=True`.
- **Reference**: Computes corr and always writes; no NaN/Inf checks; no class stats; no combined-layer corr; no plots; no token-count guard beyond whatever tokenizer produces.

### 4) Thresholding & Sparse Export
- **Current**: Dense path keeps signed correlations; threshold by |corr| then sets diagonal to 0; sparse path uses density tag (e.g., `05`) and saves edge_index/edge_attr; for sparse export, diag is first filled with 1.0 **after** threshold when building the sparse matrix.
- **Reference**: Dense path: threshold by |corr|, diag→0; sparse path: threshold by |corr|, diag→1.0, filenames use raw `network_density` float; no PNGs.

### 5) Saved Artifacts per Question
- **Current**: Writes label, per-layer corr, thresholded corr, degree, activation, activation_avg, word2vec stats, optional combined-layer outputs, plus visualization PNGs; tracks exclusions.
- **Reference**: Writes label, layer_average_corr/activation/degree, per-layer corr (or sparse), activation, activation_avg, word2vec stats; no exclusions, no combined-layer outputs, no plots.

### 6) Dataset Loading
- **Current**: Applies `np.nan_to_num` (nan/±inf→0) and clips edge_attr to [-1,1]; supports in-memory loading with tqdm; density tag zero-padded; uses `MAIN_DIR`; x = arange(num_nodes); diag set to 0 for dense graphs.
- **Reference**: No NaN/Inf cleaning or clipping; filenames use raw density float; in-memory loading without description; same thresholding logic but no sanitization; x = arange(num_nodes); diag set to 0 for dense graphs.

### 7) Model Architecture (GCN)
- **Current**: Custom `SimpleGCNConv` and pooling implemented in pure PyTorch (no torch_scatter); clamps edge weights to [-1,1]; uses `torch.nan_to_num` on messages and outputs; mean aggregation with manual degree clamp; global mean/max pooling implemented without torch_scatter; output is a single logit (binary).
- **Reference**: Uses `torch_geometric.nn.GCNConv` with `add_self_loops=False, normalize=False`; standard `global_mean_pool`/`global_max_pool`; no NaN handling or clamping; output dim defaults to 1 but models commonly instantiated with `num_output=2` (two-class logits).

### 8) Training Loop
- **Current**: Binary setup with `num_output=1`; label smoothing; gradient clipping; loss clamping; NaN/Inf guards on outputs and loss; warmup scheduler option; more verbose logging; dropout applied in GCN stack.
- **Reference**: Multiclass style with `num_output=2`; plain cross-entropy; no NaN checks or clipping; scheduler on accuracy with early stopping; simpler logging; dropout only via GCNConv defaults.

### 9) Resilience & Diagnostics
- **Current**: Excludes corrupt questions at compute time; sanitizes at load and at runtime; defensive pooling and GCN math; richer artifacts (PNGs, stats, exclusions list).
- **Reference**: No exclusion or sanitization; any NaN/Inf propagates; fewer artifacts and diagnostics.

## File Pointers
- Current pipeline: [hallucination/compute_llm_network.py](hallucination/compute_llm_network.py), [hallucination/dataset.py](hallucination/dataset.py), [hallucination/train.py](hallucination/train.py), [utils/probing_model.py](utils/probing_model.py)
- Reference pipeline: [llm-graph-probing_reference_do_not_change/hallucination/compute_llm_network.py](llm-graph-probing_reference_do_not_change/hallucination/compute_llm_network.py), [llm-graph-probing_reference_do_not_change/hallucination/dataset.py](llm-graph-probing_reference_do_not_change/hallucination/dataset.py), [llm-graph-probing_reference_do_not_change/hallucination/train.py](llm-graph-probing_reference_do_not_change/hallucination/train.py), [llm-graph-probing_reference_do_not_change/utils/probing_model.py](llm-graph-probing_reference_do_not_change/utils/probing_model.py)
