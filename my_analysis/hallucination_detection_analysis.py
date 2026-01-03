"""
=================================
HALLUCINATION DETECTION ANALYSIS
=================================

PURPOSE:
    This script orchestrates the 3-step hallucination detection pipeline:
    1. Construct dataset and extract LLM activations
    2. Compute functional connectivity (FC) networks
    3. Train and evaluate hallucination detection probes

CONFIGURATION:
    All pipeline parameters are loaded from a YAML config file.
    Pass config path via: --config /path/to/config.yaml

FILE & FOLDER NAMING CONVENTIONS:
=================================

1. MODEL NAME SANITIZATION:
    - Input:  "Qwen/Qwen2.5-0.5B"
    - Output: "Qwen_Qwen2_5_0_5B"
    - Rule: Replace '/', '-', and '.' with '_' to create safe folder names
   - Used in: model_tag, directory paths, filenames

2. DIRECTORY STRUCTURE:
   
   ${MAIN_DIR}/reports/hallucination_analysis/
   ├── {model_tag}/                           # e.g., Qwen_Qwen2_5_0_5B
   │   ├── pipeline_config.yaml               # Copy of config file (provenance)
   │   ├── layer_5/                           # Per-layer subdirectory
   │   │   ├── N1_construct_dataset.log       # Step 1: dataset construction log
   │   │   ├── N2_compute_network.log         # Step 2: FC matrix computation log
   │   │   ├── N3_train_probe.log             # Step 3: probe training log
   │   │   ├── summary.json                   # Step results summary
   │   │   ├── {step_name}/                   # Artifacts by step
   │   │   │   ├── train_history.json         # Training curves
   │   │   │   ├── train_losses.txt           # Per-epoch training losses
   │   │   │   ├── probe_performance.json     # Classification metrics
   │   │   │   └── ...                        # Additional artifacts
   │   │   ├── fc_healthy_layer_5.npy         # Correlation matrices (healthy)
   │   │   ├── fc_healthy_layer_5.png         # Visualization (healthy)
   │   │   ├── probe_weights/                 # Trained probe checkpoint
   │   │   │   └── best_model.pt
   │   │   └── tensorboard/                   # TensorBoard logs
   │   │       └── events.out.tfevents...
   │   └── ...                                # Additional layers

3. FILENAME PATTERNS:

   A. Correlation Matrices (NumPy files):
      - Format: fc_[condition]_layer_{layer_id}.npy
      - Examples:
        * fc_healthy_layer_5.npy      (healthy condition)
        * fc_hallucination_layer_5.npy (hallucination condition)
      - Dimensions: (n_samples, n_nodes, n_nodes)

   B. Visualization Images (PNG):
      - Format: fc_[condition]_layer_{layer_id}.png
      - Examples:
        * fc_healthy_layer_5.png
        * fc_hallucination_layer_5.png
      - Type: Heatmap of correlation matrix

   C. Log Files:
      - Format: N{step_number}_{step_name}.log
      - Examples:
        * N1_construct_dataset.log
        * N2_compute_network.log
        * N3_train_probe.log
      - Content: All stdout/stderr from subprocess calls

   D. Results Summaries (JSON):
      - summary.json - Top-level results summary
        * Fields: step, name, status, duration_sec, log, artifacts, metrics
      - probe_performance.json - Classification metrics
        * Fields: accuracy, precision, recall, f1, auc_roc, best_epoch, etc.
      - train_history.json - Training curves
        * Fields: train_loss, val_loss, val_acc, etc. (per epoch)

4. CHECKPOINT & MODEL PATHS:

   Saved Checkpoints:
   ${MAIN_DIR}/saves/
   ├── {model_tag}/                           # Sanitized model name
   │   └── {dataset_name}/
   │       └── {ckpt_step}/
   │           └── layer_{layer_id}/
   │               ├── best_model.pt          # Best probe checkpoint
   │               ├── final_model.pt         # Final epoch checkpoint
   │               └── optimizer.pt           # Optimizer state

5. ARTIFACT ORGANIZATION:

   Step-specific artifacts are organized in subdirectories:
   
   layer_5/
   ├── construct_dataset/
   │   └── dataset_stats.json
   ├── compute_network/
   │   ├── fc_health_layer_5.npy
   │   ├── fc_health_layer_5.png
   │   └── connectivity_stats.json
   └── train_probe/
       ├── probe_performance.json
       ├── train_history.json
       ├── best_model.pt
       └── tensorboard/

6. SPECIAL NAMING RULES:

   - Layer IDs: Always formatted as integers (5, 10, etc.)
   - Density levels: Stored separately in connectivity computation
   - Dataset names: Preserved as-is (truthfulqa, halueval, etc.)
   - Condition labels: healthy, hallucination, etc.
   - Model checkpoints: {step}.pt format (best_model.pt, final_model.pt)

=================================
EXAMPLE FULL PATH:
/ptmp/aomidvarnia/analysis_results/llm_graph/reports/hallucination_analysis/
  Qwen_Qwen2_5_0_5B/
  layer_5/
  fc_healthy_layer_5.npy

=================================
"""

import sys
from pathlib import Path
import logging
import os
import argparse
import json
import time
from datetime import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
import yaml
from tensorboard.backend.event_processing import event_accumulator
from ptpython.repl import embed


def load_config_from_yaml(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run(cmd, *, cwd: Path, env: dict, log_file: Path | None = None) -> int:
    """Stream subprocess output, filter noisy lines, and optionally tee to a log file.
    
    Detects fatal errors and raises an exception to stop the pipeline immediately.
    """
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8")

    assert process.stdout is not None
    last_tqdm_line = ""  # Track last tqdm line for carriage return handling in log
    fatal_error_lines = []  # Collect fatal error messages
    in_traceback = False  # Track if we're inside a traceback
    
    for raw_line in process.stdout:
        line = raw_line
        
        # Detect start of traceback
        if line.strip().startswith("Traceback (most recent call last):"):
            in_traceback = True
            fatal_error_lines.append(line.rstrip())
        
        # Collect traceback lines and error messages
        if in_traceback:
            fatal_error_lines.append(line.rstrip())
            # Tracebacks end with error type (e.g., RuntimeError:, ValueError:, etc.)
            if any(err in line for err in ["Error:", "Exception:"]):
                in_traceback = False
        
        # Detect FATAL errors - these should immediately stop the pipeline
        if "FATAL" in line or ("Error:" in line and not line.strip().startswith("#")):
            if line.rstrip() not in fatal_error_lines:
                fatal_error_lines.append(line.rstrip())
            print(f"🔴 ERROR DETECTED: {line.rstrip()}")
            if log_handle:
                log_handle.write(f"\n🔴 ERROR DETECTED: {line}\n")
                log_handle.flush()
        
        # Skip noisy lines
        if "amdgpu.ids" in line:
            continue
        # Skip known benign warnings (e.g., sliding window attention advisory)
        if "Sliding Window Attention is enabled" in line:
            continue
        # Skip PyTorch Geometric import warnings (torch-scatter, torch-cluster, etc.)
        if "torch_geometric/__init__.py" in line and "UserWarning" in line:
            continue
        if "undefined symbol:" in line and "_version_cuda.so" in line:
            continue
        if "import torch_geometric.typing" in line:
            continue
        # Skip absl logging format (e.g., "I0102 10:03:21.477567 22613446973248")
        # Pattern: Single letter + 4 digits (MMDD) + space + timestamp
        if line.startswith(("I", "W", "E", "F")) and len(line) > 5:
            if line[1:5].isdigit() and line[5] in (' ', ':'):
                continue

        # Detect tqdm progress lines and keep them inline
        tqdm_like = ("it/s" in line and "%" in line and "|" in line)
        if tqdm_like:
            clean = line.rstrip("\n")
            print(clean, end="\r", flush=True)
            # In logs, only write the most recent tqdm line (overwrite with carriage return)
            if log_handle:
                if last_tqdm_line:
                    log_handle.write("\r")  # Carriage return to overwrite last tqdm line
                log_handle.write(clean)
                log_handle.flush()
                last_tqdm_line = clean
            continue

        # Handle tqdm carriage-return updates on a single line
        if "\r" in line:
            segment = line.rstrip("\n").split("\r")[-1]
            if segment:
                print(segment, end="\r", flush=True)
                if log_handle and last_tqdm_line:
                    log_handle.write("\r" + segment)
                    log_handle.flush()
            continue

        # For regular lines, finalize last tqdm line with newline if any
        if log_handle and last_tqdm_line:
            log_handle.write("\n")
            last_tqdm_line = ""

        # Strip and print non-empty lines
        line = line.rstrip()
        if line:
            print(line)
            if log_handle:
                log_handle.write(line + "\n")
                log_handle.flush()

    process.wait()
    if log_handle:
        log_handle.close()
    
    # If fatal errors were detected, raise exception immediately
    if fatal_error_lines:
        error_msg = "\n".join(fatal_error_lines[:5])  # Show first 5 fatal errors
        raise RuntimeError(f"Pipeline stopped due to fatal error:\n{error_msg}")
    
    return process.returncode

# --------------
# Ref: https://github.com/omidvarnia/llm-graph-probing
# --------------

# Defaults
default_project_dir = '/u/aomidvarnia/GIT_repositories/llm-graph-probing'
default_main_dir = '/ptmp/aomidvarnia/analysis_results/llm_graph'

# Parse arguments - only config file path is required
parser = argparse.ArgumentParser(description="Graph Probing Analysis - Load all parameters from config file")
parser.add_argument("--config", type=str, required=True, help="Path to pipeline configuration YAML file")
parser.add_argument("--main_dir", type=str, default=None, help="Override root directory (optional)")
parser.add_argument("--project_dir", type=str, default=None, help="Override project directory (optional)")
args = parser.parse_args()

# Load configuration from YAML
config_path = Path(args.config).resolve()
if not config_path.exists():
    print(f"ERROR: Configuration file not found: {config_path}")
    sys.exit(1)

config = load_config_from_yaml(config_path)

# Extract common and hallucination-specific parameters from config
common_cfg = config.get('common', {})
hallu_cfg = config.get('hallucination', {})

# Build parameters from config with optional command-line overrides
main_dir = Path(common_cfg.get('main_dir') or args.main_dir or default_main_dir).resolve()
project_dir = Path(common_cfg.get('project_dir') or args.project_dir or default_project_dir).resolve()
dataset_name = common_cfg.get('dataset_name', 'truthfulqa')
model_name = common_cfg.get('model_name', 'gpt2')
ckpt_step = int(common_cfg.get('ckpt_step', -1))
batch_size = int(hallu_cfg.get('batch_size', 16))
layer_list = common_cfg.get('layer_list', '5')
probe_input = hallu_cfg.get('probe_input', 'corr')
network_density = float(common_cfg.get('density', 0.05))
eval_batch_size = int(hallu_cfg.get('eval_batch_size', 32))
num_channels = int(common_cfg.get('hidden_channels', 32))
num_layers = int(common_cfg.get('num_layers', 2))
learning_rate = float(hallu_cfg.get('learning_rate', 0.001))
num_epochs = int(hallu_cfg.get('num_epochs', 100))
early_stop_patience = int(hallu_cfg.get('early_stop_patience', 20))
# Label smoothing is optional: null means not applied, positive number means applied
label_smoothing = hallu_cfg.get('label_smoothing', 0.0)
if label_smoothing is not None:
    label_smoothing = float(label_smoothing)
# Gradient clipping is optional: null means not applied, positive number means applied
gradient_clip = hallu_cfg.get('gradient_clip', 1.0)
if gradient_clip is not None:
    gradient_clip = float(gradient_clip)

# =====================================
# SEED HANDLING FOR REPRODUCIBILITY
# =====================================
# If seed is None/null in config, generate a random seed once at the beginning
# This ensures ALL layers use the SAME train/test split for fair comparison
# The generated seed is saved and logged for reproducibility
seed = hallu_cfg.get('seed', None)  # None means random (no seed)
if seed is not None:
    seed = int(seed)
else:
    # Generate a random seed based on current time
    import random
    import time
    seed = int(time.time() * 1000000) % (2**32)  # 32-bit seed for PyTorch compatibility
    # Alternative: seed = random.randint(0, 2**32 - 1)

gpu_id = int(common_cfg.get('gpu_id', 0))
from_sparse_data = common_cfg.get('from_sparse_data', True)
if isinstance(from_sparse_data, str):
    from_sparse_data = from_sparse_data.lower() in ('true', '1', 'yes')
aggregate_layers = common_cfg.get('aggregate_layers', False)
if isinstance(aggregate_layers, str):
    aggregate_layers = aggregate_layers.lower() in ('true', '1', 'yes')

# Add the project root to the Python path
sys.path.insert(0, str(project_dir))

# Set up environment for subprocess calls
env = os.environ.copy()
# Set PYTHONPATH to include project root, ensuring utils package is found before local utils.py
env['PYTHONPATH'] = str(project_dir)
env['MAIN_DIR'] = str(main_dir)

# Ensure GPU environment variables are propagated to subprocesses
# These are set in the SLURM script but might not be inherited properly
if 'HIP_VISIBLE_DEVICES' in os.environ:
    env['HIP_VISIBLE_DEVICES'] = os.environ['HIP_VISIBLE_DEVICES']
if 'ROCR_VISIBLE_DEVICES' in os.environ:
    env['ROCR_VISIBLE_DEVICES'] = os.environ['ROCR_VISIBLE_DEVICES']
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']

# Ensure we use the same Python executable
python_exe = sys.executable

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info("="*10)
logging.info("HALLUCINATION DETECTION PIPELINE - DEVICE CONFIGURATION")
logging.info("="*10)

# =====================================
# CENTRALIZED DEVICE SELECTION
# =====================================
# Device selection happens ONCE here. The selected gpu_id is then passed to all
# subprocesses (Step 2, 3, 4). Each subprocess uses select_device(gpu_id) which
# will consistently use the same device.
#
# Priority: ROCm (AMD GPUs) → CUDA (NVIDIA GPUs) → CPU fallback
# If GPU not available or fails, gpu_id is set to -1 to force CPU mode everywhere.
# =====================================

import torch
device_type = "CPU"
device_info = "No GPU detected"
device_available = False

# Check if CUDA (includes ROCm via torch.cuda compatibility layer) is available
if torch.cuda.is_available():
    try:
        # Attempt to query the requested GPU
        device_name = torch.cuda.get_device_name(gpu_id)
        device_available = True
        
        # Detect if this is ROCm (AMD) or CUDA (NVIDIA)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        if is_rocm:
            device_type = "ROCm/HIP"
            device_info = f"AMD GPU: {device_name}"
            logging.info(f"✓ Device Type: {device_type}")
            logging.info(f"  {device_info}")
            logging.info(f"  ROCm Version: {torch.version.hip}")
        else:
            device_type = "CUDA"
            device_info = f"NVIDIA GPU: {device_name}"
            logging.info(f"✓ Device Type: {device_type}")
            logging.info(f"  {device_info}")
            logging.info(f"  CUDA Version: {torch.version.cuda}")
        
        logging.info(f"  Selected GPU ID: {gpu_id}")
        logging.info(f"  Total Available GPUs: {torch.cuda.device_count()}")
        
    except Exception as e:
        # GPU query failed - fall back to CPU
        logging.error(f"✗ Failed to access GPU {gpu_id}: {e}")
        logging.warning(f"  Falling back to CPU mode")
        device_type = "CPU"
        device_info = "GPU access failed"
        device_available = False
        gpu_id = -1  # Force CPU mode in all subprocesses
else:
    # No GPU available
    logging.warning(f"⚠ Device Type: {device_type}")
    logging.warning(f"  {device_info}")
    logging.warning(f"  Pipeline will run on CPU (very slow for LLM inference)")
    gpu_id = -1  # Force CPU mode in all subprocesses

logging.info(f"\nGPU Environment Variables:")
logging.info(f"  HIP_VISIBLE_DEVICES: {env.get('HIP_VISIBLE_DEVICES', 'Not set')}")
logging.info(f"  ROCR_VISIBLE_DEVICES: {env.get('ROCR_VISIBLE_DEVICES', 'Not set')}")
logging.info(f"  CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
logging.info("="*10 + "\n")

logging.info("Starting Graph Probing Analysis")
# embed(globals(), locals())





# -----------------------------
# Analysis Pipeline Configuration (from YAML)
# -----------------------------
# Auto-detect layers if layer_list is 'all'
if str(layer_list).strip().lower() == 'all':
    from utils.model_utils import get_model_num_layers
    num_model_layers = get_model_num_layers(model_name)
    layer_ids = list(range(num_model_layers))  # 0 to num_layers-1
    logging.info(f"Auto-detected {num_model_layers} layers in model {model_name}")
    logging.info(f"Processing all layers: {layer_ids}")
elif isinstance(layer_list, (list, tuple)):
    layer_ids = [int(x) for x in layer_list]
else:
    layer_ids = [int(x) for x in str(layer_list).replace(',', ' ').split() if x.strip()]
if not layer_ids:
    raise ValueError("layer_list must specify at least one layer (e.g., '5' or '5,6' or 'all')")
layer_id = layer_ids[0]

# Output directories for interim artifacts
# Sanitize model name: replace '/' and '-' with '_'
model_tag = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
# Add probe type to model tag for Step 3+ outputs (training, evaluation, analysis)
# Step 1 & 2 outputs (dataset, network) use model_tag WITHOUT probe type for reusability
probe_type = "gcn" if num_layers > 0 else "mlp"
model_tag_with_probe = f"{model_tag}_{probe_type}"

# =============================================================================
# SEED LOGGING AND PERSISTENCE
# =============================================================================
logging.info("="*10)
logging.info("REPRODUCIBILITY: Random Seed Configuration")
logging.info("="*10)

# Report the seed being used (whether from config or auto-generated)
seed_source = "from config file" if hallu_cfg.get('seed', None) is not None else "AUTO-GENERATED"
logging.info(f"Random Seed: {seed} ({seed_source})")
logging.info(f"  This seed will be used for ALL layers to ensure identical train/test splits")
logging.info(f"  Train/Test Split: 80/20 ({int(5915*0.8)}/{int(5915*0.2)} samples)")

# Save seed to a file for reproducibility
reports_root = main_dir / "reports" / "hallucination_analysis" / model_tag_with_probe
reports_root.mkdir(parents=True, exist_ok=True)
seed_file = reports_root / "random_seed.txt"
with open(seed_file, 'w') as f:
    f.write(f"# Random Seed Configuration\n")
    f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"# Source: {seed_source}\n")
    f.write(f"# Model: {model_name}\n")
    f.write(f"# Dataset: {dataset_name}\n")
    f.write(f"# Layers: {layer_ids}\n")
    f.write(f"\n")
    f.write(f"seed={seed}\n")
    f.write(f"\n")
    f.write(f"# To reproduce this exact analysis, add this to your config YAML:\n")
    f.write(f"# hallucination:\n")
    f.write(f"#   seed: {seed}\n")

logging.info(f"✓ Seed saved to: {seed_file}")
logging.info("  You can use this seed in future runs to reproduce the exact same train/test split")
logging.info("="*10 + "\n")

# =============================================================================
# CLEANUP: Selective cleanup - preserve Step 1 & 2 outputs for reuse
# =============================================================================
logging.info("="*10)
logging.info("SELECTIVE CLEANUP: Preserving Step 1 & 2 outputs")
logging.info("="*10)

# Controlled via config: common.enable_cleanup (default False)
# When True: remove Step 3+ outputs (reports/saves/runs) for a fresh run
# When False: keep all previous results intact
enable_cleanup_cfg = common_cfg.get('enable_cleanup', False)
if isinstance(enable_cleanup_cfg, str):
    enable_cleanup = enable_cleanup_cfg.strip().lower() in ("true", "1", "yes", "y")
else:
    enable_cleanup = bool(enable_cleanup_cfg)

if not enable_cleanup:
    logging.info("Skipping cleanup of Step 3+ outputs (preserving existing results)")
else:
    # Define all output directories that will be cleaned
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if ckpt_step == -1:
        model_dir_name = sanitized_model_name
    else:
        model_dir_name = f"{sanitized_model_name}_step{ckpt_step}"

    # Only remove Step 3+ outputs (training, evaluation, analysis)
    # Preserve Step 1 (dataset) and Step 2 (network) for reuse
    cleanup_dirs = [
        main_dir / "reports" / "hallucination_analysis" / model_tag_with_probe,
        main_dir / "saves" / "hallucination" / dataset_name / model_dir_name,
        main_dir / "runs" / "hallucination" / dataset_name / model_dir_name,
    ]

    logging.info("Removing Step 3+ outputs (training, evaluation, analysis)...")
    for cleanup_dir in cleanup_dirs:
        if cleanup_dir.exists():
            try:
                import shutil
                shutil.rmtree(cleanup_dir)
                logging.info(f"✓ Removed: {cleanup_dir}")
            except Exception as e:
                logging.warning(f"⚠ Failed to remove {cleanup_dir}: {e}")
        else:
            logging.info(f"  (not found, skipping): {cleanup_dir}")

    # Preserve Step 1 & 2 outputs
    preserved_dir = main_dir / "data" / "hallucination" / dataset_name / model_dir_name
    if preserved_dir.exists():
        logging.info(f"✓ Preserving Step 1 & 2 outputs: {preserved_dir}")
    else:
        logging.info(f"  No existing Step 1 & 2 outputs found at: {preserved_dir}")

    logging.info("="*10 + "\n")

# Create fresh output directories
reports_root = main_dir / "reports" / "hallucination_analysis" / model_tag_with_probe
reports_root.mkdir(parents=True, exist_ok=True)

# Copy pipeline config into the model folder for provenance
cfg_src = project_dir / "pipeline_config.yaml"
try:
    if cfg_src.exists():
        shutil.copy2(str(cfg_src), str(reports_root / "pipeline_config.yaml"))
except Exception:
    pass

# Layer-specific reports directory
# Ensure per-layer directory under the model folder
reports_dir = reports_root / f"layer_{layer_id}"
reports_dir.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _save_matrix_image(mat: np.ndarray, title: str, out_path: Path, vmax: float | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    vmax = vmax if vmax is not None else np.percentile(np.abs(mat), 99) if mat.size else 1.0
    plt.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_sparse_to_dense(edge_index_path: Path, edge_attr_path: Path, size: int) -> np.ndarray:
    edge_index = np.load(edge_index_path)
    edge_attr = np.load(edge_attr_path)
    dense = np.zeros((size, size), dtype=edge_attr.dtype)
    for (u, v), w in zip(edge_index.T, edge_attr):
        dense[u, v] = w
        dense[v, u] = w
    np.fill_diagonal(dense, 1.0)
    return dense


def _plot_series(series: list[tuple[float, float]], title: str, ylabel: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if series:
        xs, ys = zip(*series)
        plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

step_results = []

# =====================================
# STEP SELECTION AND VALIDATION
# =====================================
steps_config = common_cfg.get('steps', '1,2,3,4,5')
try:
    requested_steps = sorted(set(int(s.strip()) for s in str(steps_config).split(',') if s.strip()))
    if not requested_steps:
        raise ValueError("At least one step must be specified")
    if not all(1 <= s <= 5 for s in requested_steps):
        raise ValueError("Steps must be between 1 and 5")
except (ValueError, TypeError) as e:
    logging.error(f"✗ INVALID STEPS CONFIGURATION: {e}")
    logging.error(f"  Please specify steps as comma-separated integers (e.g., '1', '1,2,3', '3,4,5')")
    sys.exit(1)

logging.info(f"Requested steps: {requested_steps}")

# Validate that if not starting from step 1, previous step results exist
if requested_steps[0] > 1:
    logging.info("Pipeline not starting from Step 1 - checking for previous step results...")
    min_step = requested_steps[0]
    
    # Check Step 1 outputs (dataset)
    if min_step > 1:
        dataset_path = main_dir / "data/hallucination" / f"{dataset_name}.csv"
        if not dataset_path.exists():
            logging.error(f"✗ VALIDATION FAILED: Step 1 outputs not found")
            logging.error(f"  Expected dataset file: {dataset_path}")
            logging.error(f"  Cannot start from step {min_step} without Step 1 results")
            sys.exit(1)
        logging.info(f"✓ Step 1 outputs found: {dataset_path}")
    
    # Check Step 2 outputs (network)
    if min_step > 2:
        sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        if ckpt_step == -1:
            model_dir_name = sanitized_model_name
        else:
            model_dir_name = f"{sanitized_model_name}_step{ckpt_step}"
        network_base_dir = main_dir / "data" / "hallucination" / dataset_name / model_dir_name
        if not network_base_dir.exists():
            logging.error(f"✗ VALIDATION FAILED: Step 2 outputs not found")
            logging.error(f"  Expected network directory: {network_base_dir}")
            logging.error(f"  Cannot start from step {min_step} without Step 2 results")
            sys.exit(1)
        logging.info(f"✓ Step 2 outputs found: {network_base_dir}")
    
    # Check Step 3 outputs (training)
    if min_step > 3:
        model_tag = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        probe_type = "gcn" if num_layers > 0 else "mlp"
        model_tag_with_probe = f"{model_tag}_{probe_type}"
        sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        if ckpt_step == -1:
            model_dir_name = sanitized_model_name
        else:
            model_dir_name = f"{sanitized_model_name}_step{ckpt_step}"
        saves_dir = main_dir / "saves" / "hallucination" / dataset_name / model_dir_name
        if not saves_dir.exists() or len(list(saves_dir.glob("layer_*/best_model*"))) == 0:
            logging.error(f"✗ VALIDATION FAILED: Step 3 outputs not found")
            logging.error(f"  Expected trained models in: {saves_dir}")
            logging.error(f"  Cannot start from step {min_step} without Step 3 results")
            sys.exit(1)
        logging.info(f"✓ Step 3 outputs found: {saves_dir}")
    
    logging.info("✓ All required previous step outputs validated successfully\n")

# ========================================================================
# FILE AND FOLDER NAMING CONVENTION DOCUMENTATION
# ========================================================================
# This section documents the naming convention for all files, folders,
# and figures created throughout the analysis pipeline.
#
# KEY RULES:
#   - Model names with '/' and '-' are sanitized by replacing with '_'
#     Example: "Qwen/Qwen2.5-0.5B" → "Qwen_Qwen2_5_0_5B"
#   - All special characters in filenames/folder names are replaced with '_'
#   - No spaces, periods (except extensions), backslashes, or slashes
#
# DIRECTORY STRUCTURE:
#   {main_dir}/
#   ├── data/hallucination/
#   │   ├── {dataset_name}.csv                    [Input dataset file]
#   │   └── {dataset_name}/{sanitized_model_name}/
#   │       └── layer_{layer_id}/
#   │           ├── *_activations.npy            [LLM activations]
#   │           ├── *_correlations.npy           [Correlation matrices]
#   │           ├── *_sparse_*_edge_index.npy   [Sparse network edges]
#   │
#   ├── reports/hallucination_analysis/{sanitized_model_name}/
#   │   ├── pipeline_config.yaml                 [Copy of config]
#   │   └── layer_{layer_id}/
#   │       ├── step1_construct_dataset.log      [Dataset construction log]
#   │       ├── step2_compute_network.log        [Network computation log]
#   │       ├── step3_train.log                  [Probe training log]
#   │       ├── analysis_summary.json            [Results summary]
#   │       └── *.png                            [Visualization figures]
#   │
#   ├── saves/hallucination/{dataset_name}/{sanitized_model_dir}/
#   │   └── layer_{layer_id}/
#   │       └── best_model_density-{DD}_dim-{C}_hop-{L}_input-{T}.pth
#   │           where: DD=density%, C=channels, L=layers, T=input_type
#   │
#   └── runs/hallucination/{dataset_name}/{sanitized_model_dir}/
#       └── layer_{layer_id}/
#           └── events.* [TensorBoard event files]
#
# FILENAME PATTERNS:
#   - Network files: *_sparse_*_edge_index.npy, *_correlations.npy
#   - Model files: best_model_density-05_dim-32_hop-2_input-corr.pth
#   - Log files: step{N}_{operation}.log
#   - Figures: train_loss.png, test_metrics.png, fc_*.png
#
# VARIABLES IN FILENAMES:
#   - {dataset_name}: truthfulqa, halueval, medhallu, helm
#   - {sanitized_model_name}: Original model name with /, - → _
#     Example: "Qwen/Qwen2.5-0.5B" becomes "Qwen_Qwen2_5_0_5B"
#   - {layer_id}: Integer layer number (e.g., 5)
#   - {DD}: Density as 2-digit percentage (e.g., 05 for 0.05)
#   - {C}: Number of hidden channels (e.g., 32)
#   - {L}: Number of GNN layers (e.g., 2)
#   - {T}: Probe input type (corr, activation, etc.)
#
# ========================================================================

# ========================================================================
# Step 1: Prepare the dataset (with reuse check)
# ========================================================================
if 1 in requested_steps:
    print("\n" + "="*10)
    print("Step 1 - Dataset Construction (Extracting LLM Activations)")
    print("="*10 + "\n")
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Model: {model_name} (sanitized: {model_tag})")
    logging.info(f"Checkpoint step: {ckpt_step}")
    logging.info(f"Batch size: {batch_size}")

step1_log = reports_dir / "step1_construct_dataset.log"
step_start = time.monotonic()

# Check if Step 1 output already exists and can be reused
dataset_path = main_dir / "data/hallucination" / f"{dataset_name}.csv"
skip_step1 = False

if dataset_path.exists():
    try:
        # Validate existing dataset
        df = pd.read_csv(dataset_path)
        required_cols = ['label', 'answer', 'question']
        if all(col in df.columns for col in required_cols) and len(df) > 0:
            logging.info(f"✓ REUSING existing dataset: {dataset_path}")
            logging.info(f"  - Total rows: {len(df)}")
            logging.info(f"  - Columns: {list(df.columns)}")
            skip_step1 = True
        else:
            logging.warning(f"⚠ Existing dataset invalid, will reconstruct")
    except Exception as e:
        logging.warning(f"⚠ Failed to validate existing dataset: {e}")
        logging.warning("  Will reconstruct dataset")

if not skip_step1:
    logging.info(f"Using Python: {sys.executable}")
    logging.info("Executing construct_dataset.py...")
    
    try:
        result = run(
            [
                python_exe,
                "hallucination/construct_dataset.py",
                f"--dataset_name={dataset_name}",
            ],
            cwd=project_dir,
            env=env,
            log_file=step1_log,
        )
    except RuntimeError as e:
        logging.error(f"✗ FATAL ERROR in Step 1: {e}")
        logging.error("Pipeline stopping due to fatal error")
        sys.exit(1)
    
    if result != 0:
        logging.error(f"✗ Dataset construction failed with return code {result}")
        logging.error("Pipeline stopping due to error")
        sys.exit(1)
    
    logging.info("✓ Dataset constructed successfully")
else:
    logging.info("✓ Skipped: Using existing dataset")

step_duration = time.monotonic() - step_start

# Sanity checks for dataset
try:
    dataset_path = main_dir / "data/hallucination/truthfulqa.csv"
    if not dataset_path.exists():
        logging.error(f"✗ SANITY CHECK FAILED: Dataset file not found at {dataset_path}")
        sys.exit(1)
    
    logging.info(f"✓ SANITY CHECK: Dataset file exists at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logging.info(f"✓ SANITY CHECK: Dataset loaded successfully")
    logging.info(f"  - Total rows: {len(df)}")
    logging.info(f"  - Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = ['label', 'answer']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"✗ SANITY CHECK FAILED: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    logging.info(f"✓ SANITY CHECK: All required columns present: {required_cols}")
    
    # Check label distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts().sort_index()
        logging.info(f"✓ SANITY CHECK: Label distribution: {dict(label_counts)}")
        
        # Check if we have both classes
        if len(label_counts) < 2:
            logging.error(f"✗ SANITY CHECK FAILED: Only {len(label_counts)} class(es) found, expected 2")
            sys.exit(1)
        
        logging.info(f"✓ SANITY CHECK: Both classes present")
        
        # Create visualizations
        label_plot = reports_dir / "dataset_label_distribution.png"
        label_counts.plot(kind="bar")
        plt.title(f"TruthfulQA label distribution (n={len(df)})")
        plt.xlabel("label")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(label_plot)
        plt.close()
        logging.info(f"✓ Saved label distribution plot to {label_plot}")
    
    # Save dataset head
    dataset_head = reports_dir / "dataset_head.csv"
    df.head(20).to_csv(dataset_head, index=False)
    logging.info(f"✓ Saved dataset head (20 rows) to {dataset_head}")
    
    logging.info("✓ Dataset is ready for processing")
    step_results.append({"step": 1, "name": "construct_dataset", "status": "ok", "duration_sec": step_duration, "log": str(step1_log), "dataset_path": str(dataset_path), "rows": len(df), "classes": len(label_counts) if "label" in df.columns else 0})
except Exception as e:
    logging.error(f"✗ SANITY CHECK FAILED: Could not load/validate dataset: {e}")
    step_results.append({"step": 1, "name": "construct_dataset", "status": "error", "duration_sec": step_duration, "log": str(step1_log)})

# -----------------------------
# Step 2: Generate the neural topology (with reuse check)
# -----------------------------
print("\n" + "="*10)
print("Step 2 - Neural Topology (Network Graph) Computation")
print("="*10 + "\n")
logging.info(f"Layer IDs: {layer_ids}")
logging.info(f"Network density: {network_density}")
logging.info(f"Sparse mode: {from_sparse_data}")

# Sanity checks for Step 2
if not (0.01 <= network_density <= 1.0):
    logging.error(f"✗ SANITY CHECK FAILED: Network density must be between 0.01 and 1.0, got {network_density}")
    sys.exit(1)
logging.info(f"✓ SANITY CHECK: network_density={network_density} is valid")

if not layer_ids:
    logging.error(f"✗ SANITY CHECK FAILED: No layer IDs specified")
    sys.exit(1)
logging.info(f"✓ SANITY CHECK: {len(layer_ids)} layer(s) to process")

step2_log = reports_dir / "step2_compute_network.log"
step_start = time.monotonic()

# Check if Step 2 outputs already exist and can be reused
sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if ckpt_step == -1:
    model_dir_name = sanitized_model_name
else:
    model_dir_name = f"{sanitized_model_name}_step{ckpt_step}"

network_base_dir = main_dir / "data" / "hallucination" / dataset_name / model_dir_name
skip_step2 = False

if network_base_dir.exists():
    try:
        # Check if network files exist for the target layer and density
        density_tag = f"{int(round(network_density * 100)):02d}"
        sample_question_dirs = list(network_base_dir.glob('[0-9]*'))[:5]  # Check first 5 questions
        
        if len(sample_question_dirs) >= 3:  # Need at least 3 samples to be valid
            valid_outputs = True
            for qdir in sample_question_dirs:
                # Check for required files based on mode
                if from_sparse_data:
                    required_files = [
                        qdir / f"layer_{layer_id}_sparse_{density_tag}_edge_index.npy",
                        qdir / f"layer_{layer_id}_sparse_{density_tag}_edge_attr.npy",
                        qdir / "label.npy"
                    ]
                else:
                    required_files = [
                        qdir / f"layer_{layer_id}_corr.npy",
                        qdir / "label.npy"
                    ]
                
                if not all(f.exists() for f in required_files):
                    valid_outputs = False
                    break
            
            if valid_outputs:
                logging.info(f"✓ REUSING existing network outputs: {network_base_dir}")
                logging.info(f"  - Layer: {layer_id}")
                logging.info(f"  - Density: {network_density} (tag: {density_tag})")
                logging.info(f"  - Sparse mode: {from_sparse_data}")
                logging.info(f"  - Sample questions checked: {len(sample_question_dirs)}")
                skip_step2 = True
            else:
                logging.warning(f"⚠ Existing network outputs incomplete, will recompute")
        else:
            logging.warning(f"⚠ Insufficient network outputs found ({len(sample_question_dirs)} dirs), will recompute")
    except Exception as e:
        logging.warning(f"⚠ Failed to validate existing network outputs: {e}")
        logging.warning("  Will recompute networks")

if not skip_step2:
    logging.info("Executing compute_llm_network.py...")
    
    try:
        result = run(
            [
                python_exe,
                "-m",
                "hallucination.compute_llm_network",
                f"--dataset_name={dataset_name}",
                f"--llm_model_name={model_name}",
                f"--ckpt_step={ckpt_step}",
                *[f"--llm_layer={lid}" for lid in layer_ids],
                f"--batch_size={batch_size}",
                f"--network_density={network_density}",
                f"--gpu_id={gpu_id}",
                ] + (["--sparse"] if from_sparse_data else [])
                    + (["--aggregate_layers"] if aggregate_layers else []),
            cwd=project_dir,
            env=env,
            log_file=step2_log,
        )
    except RuntimeError as e:
        logging.error(f"✗ FATAL ERROR in Step 2: {e}")
        logging.error("Pipeline stopping due to fatal error")
        sys.exit(1)
    
    if result != 0:
        logging.error(f"✗ Network computation failed with return code {result}")
        logging.error("Pipeline stopping due to error")
        sys.exit(1)
    
    logging.info("✓ Network computation completed")
else:
    logging.info("✓ Skipped: Using existing network outputs")

step_duration = time.monotonic() - step_start

# Sanity checks for network outputs
# The correct path is: data/hallucination/{dataset_name}/{model_dir}/{question_id}/layer_{layer_id}_sparse_{density_tag}_edge_index.npy
# where density_tag is formatted as 2-digit integer percentage (e.g., 0.05 → "05")
sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
if ckpt_step == -1:
    model_dir_name = sanitized_model_name
else:
    model_dir_name = f"{sanitized_model_name}_step{ckpt_step}"

network_base_dir = main_dir / "data" / "hallucination" / dataset_name / model_dir_name

# Convert density to the same format used by compute_llm_network.py
density_tag = f"{int(round(network_density * 100)):02d}"
sparse_file_pattern = f"**/layer_*_sparse_{density_tag}_edge_index.npy"

if network_base_dir.exists():
    sparse_files = list(network_base_dir.glob(sparse_file_pattern))
else:
    sparse_files = []
    logging.error(f"✗ SANITY CHECK FAILED: Network output directory not found: {network_base_dir}")
    sys.exit(1)

if sparse_files:
    logging.info(f"✓ SANITY CHECK: Found {len(sparse_files)} sparse network file(s) in {network_base_dir}")
    # Log a few examples
    for i, f in enumerate(sparse_files[:3]):
        logging.info(f"  Example {i+1}: {f.relative_to(network_base_dir)}")
else:
    logging.error(f"✗ SANITY CHECK FAILED: No sparse network files found in {network_base_dir}")
    logging.error(f"  Expected pattern: {sparse_file_pattern}")
    logging.error(f"  Step 2 completed but produced no output files")
    logging.error("  Pipeline cannot continue - training step requires these files")
    sys.exit(1)

step_results.append({"step": 2, "name": "compute_network", "status": "ok", "duration_sec": step_duration, "log": str(step2_log), "sparse_files": len(sparse_files)})

# Additional summary of network outputs
try:
    topology_root = main_dir / "data/hallucination" / dataset_name / model_dir_name
    num_questions = len([p for p in topology_root.iterdir() if p.is_dir()]) if topology_root.exists() else 0
    # Save example functional connectivity matrices (before/after threshold)
    fc_before = reports_dir / "fc_before_threshold.png"
    fc_after = reports_dir / "fc_after_threshold.png"
    try:
        # Try to find a question with valid data
        sample_q = None
        dense = None
        for q_idx in range(min(num_questions, 10)):  # Try first 10 questions
            q_dir = topology_root / str(q_idx)
            dense_path = q_dir / f"layer_{layer_id}_corr.npy"
            if dense_path.exists():
                sample_q = q_idx
                dense = np.load(dense_path)
                break
        
        if dense is not None:
            if dense.shape[0] > 200:  # limit size for visualization
                dense_viz = dense[:200, :200]
            else:
                dense_viz = dense.copy()
            _save_matrix_image(dense_viz, f"{model_name} layer {layer_id} (before threshold) q{sample_q}", fc_before)
            logging.info(f"✓ Saved FC matrix before threshold (question {sample_q}): {fc_before}")
        
        # Apply thresholding to generate "after threshold" visualization
        if dense is not None:
            # Apply same thresholding logic as in dataset.py
            percentile_threshold = network_density * 100
            threshold = np.percentile(np.abs(dense), 100 - percentile_threshold)
            dense_thresholded = dense.copy()
            dense_thresholded[np.abs(dense_thresholded) < threshold] = 0
            np.fill_diagonal(dense_thresholded, 0)
            
            if dense_thresholded.shape[0] > 200:
                dense_thresh_viz = dense_thresholded[:200, :200]
            else:
                dense_thresh_viz = dense_thresholded
            
            _save_matrix_image(dense_thresh_viz, f"{model_name} layer {layer_id} (after threshold {network_density}) q{sample_q}", fc_after)
            logging.info(f"✓ Saved FC matrix after threshold (question {sample_q}): {fc_after}")
            
            # Log statistics about the thresholding
            num_edges_before = np.count_nonzero(dense)
            num_edges_after = np.count_nonzero(dense_thresholded)
            logging.info(f"  Threshold: {threshold:.4f}, Edges before: {num_edges_before}, after: {num_edges_after}")
    except Exception as viz_err:
        logging.warning(f"Could not generate connectivity previews: {viz_err}")
except Exception as e:
        logging.warning(f"Could not summarize network outputs: {e}")
        step_results.append({"step": 2, "name": "compute_network", "status": "ok", "duration_sec": step_duration, "log": str(step2_log)})

# ========================================================================
# Step 3: Train the probes
# ========================================================================
logging.info("="*10)
logging.info("STEP 3: Training Graph Neural Network Probes")
logging.info("="*10)
logging.info(f"Probe input: {probe_input}")
logging.info(f"Density: {network_density}")
logging.info(f"Num channels: {num_channels}")
logging.info(f"Num layers: {num_layers}")
logging.info(f"Learning rate: {learning_rate}")
logging.info(f"Batch size: {batch_size}")
logging.info(f"Eval batch size: {eval_batch_size}")
logging.info(f"Num epochs: {num_epochs}")
logging.info(f"Early stop patience: {early_stop_patience}")
logging.info(f"Label smoothing: {label_smoothing if label_smoothing is not None else 'None (disabled)'}")
logging.info(f"Gradient clipping: {gradient_clip if gradient_clip is not None else 'None (disabled)'}")
logging.info(f"Random seed: {seed} ({seed_source})")  # Show seed source (config or auto-generated)
logging.info(f"GPU ID: {gpu_id}")

# Sanity checks for Step 3
if num_epochs <= 0:
    logging.error(f"✗ SANITY CHECK FAILED: num_epochs must be > 0, got {num_epochs}")
    sys.exit(1)
logging.info(f"✓ SANITY CHECK: num_epochs={num_epochs} is valid")

if learning_rate <= 0:
    logging.error(f"✗ SANITY CHECK FAILED: learning_rate must be > 0, got {learning_rate}")
    sys.exit(1)
logging.info(f"✓ SANITY CHECK: learning_rate={learning_rate} is valid")

if label_smoothing is not None and not (0 <= label_smoothing < 1.0):
    logging.error(f"✗ SANITY CHECK FAILED: label_smoothing must be 0 <= x < 1.0 or None, got {label_smoothing}")
    sys.exit(1)
logging.info(f"✓ SANITY CHECK: label_smoothing={label_smoothing if label_smoothing is not None else 'disabled'} is valid")

# Train probes for ALL layers
for lid in layer_ids:
    print("\n" + "="*10)
    print(f"Layer {lid}: Step 3 - Training Probe")
    print("="*10 + "\n")
    logging.info(f"Starting probe training for layer {lid}")
    lid_reports_dir = reports_root / f"layer_{lid}"
    lid_reports_dir.mkdir(parents=True, exist_ok=True)
    step3_log = lid_reports_dir / "step3_train.log"
    logging.info("Executing train.py...")
    logging.info("This may take a while...")
    step_start = time.monotonic()
    try:
        result = run(
            [
                python_exe,
                "-m",
                "hallucination.train",
                f"--dataset_name={dataset_name}",
                f"--llm_model_name={model_name}",
                f"--ckpt_step={ckpt_step}",
                f"--llm_layer={lid}",
                f"--probe_input={probe_input}",
                f"--density={network_density}",
                f"--batch_size={batch_size}",
                f"--eval_batch_size={eval_batch_size}",
                f"--num_layers={num_layers}",
                f"--hidden_channels={num_channels}",
                f"--lr={learning_rate}",
                f"--num_epochs={num_epochs}",
                f"--early_stop_patience={early_stop_patience}",
                f"--gpu_id={gpu_id}",
                f"--seed={seed}",  # Always pass seed (auto-generated if not in config)
            ] + ([f"--label_smoothing={label_smoothing}"] if label_smoothing is not None else []) + ([f"--gradient_clip={gradient_clip}"] if gradient_clip is not None else []) + (["--from_sparse_data"] if from_sparse_data else []),
            cwd=project_dir,
            env=env,
            log_file=step3_log,
        )
    except RuntimeError as e:
        logging.error(f"✗ FATAL ERROR in Step 3 (Layer {lid}): {e}")
        logging.error("Pipeline stopping due to fatal error")
        sys.exit(1)
    step_duration = time.monotonic() - step_start

    if result != 0:
        logging.error(f"✗ Training failed for layer {lid} with return code {result}")
        logging.error(f"Pipeline stopping due to error in layer {lid}")
        sys.exit(1)
    
    logging.info(f"✓ Probe trained successfully for layer {lid}")
    
    # Sanity checks for training outputs
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{ckpt_step}"
    density_tag = f"{int(round(network_density * 100)):02d}"
    best_model_path = main_dir / "saves" / f"hallucination/truthfulqa/{model_dir}" / f"layer_{lid}" / f"best_model_density-{density_tag}_dim-{num_channels}_hop-{num_layers}_input-{probe_input}.pth"
    
    model_size = 0
    if best_model_path.exists():
        model_size = best_model_path.stat().st_size
        logging.info(f"✓ SANITY CHECK: Best model saved at {best_model_path} (size: {model_size/1e6:.2f} MB)")
    else:
        logging.warning(f"⚠ SANITY CHECK: Best model file not found at {best_model_path}")
    
    logging.info("Trained models are ready for evaluation")
    
    # TensorBoard scalars
    tb_dir = main_dir / "runs" / f"hallucination/truthfulqa/{model_dir}" / f"layer_{lid}"
    loss_plot = lid_reports_dir / "train_loss.png"
    metrics_plot = lid_reports_dir / "test_metrics.png"
    try:
        event_files = sorted(tb_dir.glob("events.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if event_files:
            logging.info(f"✓ SANITY CHECK: TensorBoard event files found in {tb_dir}")
            ea = event_accumulator.EventAccumulator(str(event_files[0]))
            ea.Reload()
            if "train/loss" in ea.Tags().get("scalars", []):
                loss_events = [(e.step, e.value) for e in ea.Scalars("train/loss")]
                _plot_series(loss_events, f"Train Loss ({model_name} layer {lid})", "loss", loss_plot)
            metrics = {}
            for tag in ["test/accuracy", "test/precision", "test/recall", "test/f1"]:
                if tag in ea.Tags().get("scalars", []):
                    metrics[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
            if metrics:
                plt.figure(figsize=(7, 5))
                for tag, series in metrics.items():
                    xs, ys = zip(*series)
                    plt.plot(xs, ys, marker="o", label=tag.split("/")[-1])
                plt.title(f"Test Metrics ({model_name} layer {lid})")
                plt.xlabel("step")
                plt.ylabel("value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(metrics_plot)
                plt.close()
    except Exception as tb_err:
        logging.warning(f"Could not generate training plots: {tb_err}")

    step_results.append({
        "step": 3,
        "layer": lid,
        "name": "train",
        "status": "ok",
        "duration_sec": step_duration,
        "log": str(step3_log),
        "best_model": str(best_model_path),
        "best_model_bytes": model_size,
        "train_loss_plot": str(loss_plot),
        "test_metrics_plot": str(metrics_plot),
    })

# -----------------------------
# Step 4: Evaluate the probes
# -----------------------------
print("\n" + "="*10)
print("Step 4 - Evaluating Trained Probes")
print("="*10 + "\n")
logging.info(f"Network density: {network_density}")
logging.info(f"Layers to evaluate: {layer_ids}")

for lid in layer_ids:
    print("\n" + "="*10)
    print(f"Layer {lid}: Step 4 - Evaluation")
    print("="*10 + "\n")
    logging.info(f"Starting evaluation for layer {lid}")
    lid_reports_dir = reports_root / f"layer_{lid}"
    step4_log = lid_reports_dir / "step4_eval.log"
    logging.info("Executing eval.py...")
    logging.info("Computing evaluation metrics...")
    step_start = time.monotonic()
    try:
        result = run(
            [
                python_exe,
                "-m",
                "hallucination.eval",
                f"--dataset_name={dataset_name}",
                f"--llm_model_name={model_name}",
                f"--ckpt_step={ckpt_step}",
                f"--llm_layer={lid}",
                f"--probe_input={probe_input}",
                f"--density={network_density}",
                f"--num_layers={num_layers}",
                f"--gpu_id={gpu_id}",
            ],
            cwd=project_dir,
            env=env,
            log_file=step4_log,
        )
    except RuntimeError as e:
        logging.error(f"✗ FATAL ERROR in Step 4 (Layer {lid}): {e}")
        logging.error("Pipeline stopping due to fatal error")
        sys.exit(1)
    step_duration = time.monotonic() - step_start

    if result != 0:
        logging.error(f"Evaluation failed for layer {lid} with return code {result}")
        logging.error(f"Pipeline stopping due to error in layer {lid}")
        sys.exit(1)
    
    logging.info(f"✓ Evaluation completed successfully for layer {lid}")
    
    # Extract metrics from eval log
    try:
        import re
        with open(step4_log, 'r') as f:
            log_content = f.read()
        
        # Find confusion matrix
        cm_match = re.search(r'\[\[(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\]\]', log_content)
        if cm_match:
            tn, fp, fn, tp = map(int, cm_match.groups())
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logging.info(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            logging.info(f"  F1 Score: {f1:.4f}")
            logging.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            logging.info(f"  Above chance (>50%): {'✓ YES' if accuracy > 0.5 else '✗ NO'}")
            
            step_results.append({
                "step": 4,
                "layer": lid,
                "name": "eval",
                "status": "ok",
                "duration_sec": step_duration,
                "log": str(step4_log),
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "above_chance": bool(accuracy > 0.5)
                }
            })
        else:
            logging.warning(f"Could not extract confusion matrix from eval log for layer {lid}")
            step_results.append({"step": 4, "layer": lid, "name": "eval", "status": "ok", "duration_sec": step_duration, "log": str(step4_log)})
    except Exception as e:
        logging.warning(f"Could not parse evaluation metrics for layer {lid}: {e}")
        step_results.append({"step": 4, "layer": lid, "name": "eval", "status": "ok", "duration_sec": step_duration, "log": str(step4_log)})

# -----------------------------
# Step 5: Graph statistics
# -----------------------------
print("\n" + "="*10)
print("Step 5 - Graph Statistics & Analysis")
print("="*10 + "\n")
logging.info(f"Layers to analyze: {layer_ids}")
logging.info(f"Network density: {network_density}")
layers_completed: list[int] = []
for lid in layer_ids:
    print("\n" + "="*10)
    print(f"Layer {lid}: Step 5 - Graph Analysis")
    print("="*10 + "\n")
    logging.info(f"Executing graph_analysis.py for layer {lid}...")
    lid_reports_dir = reports_root / f"layer_{lid}"
    lid_reports_dir.mkdir(parents=True, exist_ok=True)
    step5_log = lid_reports_dir / "step5_graph_analysis.log"
    step_start = time.monotonic()
    try:
        result = run(
            [
                python_exe,
                "-m",
                "hallucination.graph_analysis",
                f"--dataset_name={dataset_name}",
                f"--llm_model_name={model_name}",
                f"--ckpt_step={ckpt_step}",
                f"--layer={lid}",
                f"--feature={probe_input}",
            ],
            cwd=project_dir,
            env=env,
            log_file=step5_log,
        )
    except RuntimeError as e:
        logging.error(f"✗ FATAL ERROR in Step 5 (Layer {lid}): {e}")
        logging.error("Pipeline stopping due to fatal error")
        sys.exit(1)
    step_duration = time.monotonic() - step_start

    if result != 0:
        logging.error(f"Graph analysis failed for layer {lid} with return code {result}")
        logging.error(f"Pipeline stopping due to error in layer {lid}")
        sys.exit(1)

    logging.info(f"✓ Graph analysis completed successfully for layer {lid}")
    logging.info("Graph analysis results are ready")
    npy_file = lid_reports_dir / f"{model_tag}_layer_{lid}_{probe_input}_intra_vs_inter.npy"
    hist_path = lid_reports_dir / "intra_vs_inter_hist.png"
    summary_csv = lid_reports_dir / "intra_vs_inter_summary.csv"
    try:
        if npy_file.exists():
            data = np.load(npy_file, allow_pickle=True)
            if data.size > 0:
                plt.hist(data, bins=50)
                plt.title("Intra-vs-Inter Correlation Metric")
                plt.xlabel("metric")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                stats = {
                    "count": int(data.size),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "median": float(np.median(data)),
                }
                pd.DataFrame([stats]).to_csv(summary_csv, index=False)
                step_results.append({"step": 5, "layer": lid, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "npy_file": str(npy_file), "histogram": str(hist_path), "summary_csv": str(summary_csv), "stats": stats})
            else:
                step_results.append({"step": 5, "layer": lid, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "npy_file": str(npy_file), "note": "empty data"})
        else:
            step_results.append({"step": 5, "layer": lid, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "note": "npy file not found"})
    except Exception as e:
        logging.warning(f"Could not summarize graph analysis outputs for layer {lid}: {e}")
        step_results.append({"step": 5, "layer": lid, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log)})
    layers_completed.append(lid)

# -----------------------------
# Generate Classification Metrics Summary
# -----------------------------
logging.info("="*10)
logging.info("CLASSIFICATION METRICS SUMMARY")
logging.info("="*10)

# Collect evaluation metrics for all layers
eval_results = [r for r in step_results if r.get("step") == 4 and r.get("status") == "ok" and "metrics" in r]

if eval_results:
    logging.info("\n{:<8} {:<10} {:<10} {:<10} {:<10} {:<12}".format(
        "Layer", "Accuracy", "Precision", "Recall", "F1 Score", "Above Chance"))
    logging.info("-" * 20)
    
    for result in sorted(eval_results, key=lambda x: x.get("layer", 0)):
        layer = result.get("layer", "?")
        metrics = result.get("metrics", {})
        acc = metrics.get("accuracy", 0)
        prec = metrics.get("precision", 0)
        rec = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        above_chance = "\u2713 YES" if metrics.get("above_chance", False) else "\u2717 NO"
        
        logging.info("{:<8} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<12}".format(
            layer, acc, prec, rec, f1, above_chance))
        
        cm = metrics.get("confusion_matrix", {})
        logging.info(f"         Confusion Matrix: TN={cm.get('tn', 0)}, FP={cm.get('fp', 0)}, "
                    f"FN={cm.get('fn', 0)}, TP={cm.get('tp', 0)}")
    
    # Save metrics summary to CSV
    metrics_summary_path = reports_root / "classification_metrics_summary.csv"
    try:
        summary_rows = []
        for result in sorted(eval_results, key=lambda x: x.get("layer", 0)):
            layer = result.get("layer", "?")
            metrics = result.get("metrics", {})
            cm = metrics.get("confusion_matrix", {})
            summary_rows.append({
                "layer": layer,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1", 0),
                "above_chance": metrics.get("above_chance", False),
                "tn": cm.get("tn", 0),
                "fp": cm.get("fp", 0),
                "fn": cm.get("fn", 0),
                "tp": cm.get("tp", 0),
            })
        pd.DataFrame(summary_rows).to_csv(metrics_summary_path, index=False)
        logging.info(f"\n\u2713 Metrics summary saved to: {metrics_summary_path}")
    except Exception as e:
        logging.warning(f"Could not save metrics summary CSV: {e}")
else:
    logging.warning("No evaluation metrics found in results")

# -----------------------------
# Persist summary and quick visuals
# -----------------------------
summary_json = reports_root / "summary.json"
try:
    durations = [(r.get("name"), r.get("duration_sec", 0.0)) for r in step_results if "duration_sec" in r]
    if durations:
        names, vals = zip(*durations)
        plt.barh(names, vals)
        plt.xlabel("seconds")
        plt.title("Step durations")
        plt.tight_layout()
        plt.savefig(reports_root / "step_durations.png")
        plt.close()
    save_json(summary_json, {"steps": step_results})
except Exception as e:
    logging.warning(f"Could not write summary: {e}")

# -----------------------------
# Generate LaTeX Report
# -----------------------------
logging.info("="*10)
logging.info("Generating publication-ready LaTeX report...")
logging.info("="*10)

try:
    from hallucination.generate_report import generate_latex_report
    
    latex_file = generate_latex_report(
        model_name=model_name,
        dataset_name=dataset_name,
        reports_dir=reports_root,
        layer_ids=layer_ids,
        config=config,
        step_results=step_results
    )
    
    logging.info(f"✓ LaTeX report generated: {latex_file}")
    logging.info(f"  Compile with: pdflatex {latex_file.name}")
    logging.info(f"  Location: {latex_file.parent}")
    
    # Attempt to compile the LaTeX document
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", latex_file.name],
            cwd=latex_file.parent,
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            pdf_file = latex_file.with_suffix('.pdf')
            logging.info(f"✓ PDF report compiled successfully: {pdf_file}")
        else:
            logging.warning("LaTeX compilation failed. Install pdflatex to enable automatic PDF generation.")
            logging.info(f"  LaTeX source available at: {latex_file}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logging.info("pdflatex not available. LaTeX source file saved for manual compilation.")
        logging.info(f"  Install texlive-latex-base to compile: sudo apt-get install texlive-latex-base texlive-latex-extra")
        
except Exception as e:
    logging.warning(f"Could not generate LaTeX report: {e}")
    logging.info("Continuing without report generation...")

# ============================================
# Generate Figure 5 Comparison Plots
# ============================================
logging.info("="*10)
logging.info("Step 6: Generating comparison plots (Figure 5b-c)")
logging.info("="*10)

try:
    from hallucination.comparison import create_comparison_figure, create_layer_metrics_plot
    
    # Create output directory for figures
    figures_dir = reports_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Figure 5(b): Accuracy comparison across layers
    logging.info("Generating accuracy comparison plot (Figure 5b equivalent)...")
    create_layer_metrics_plot(
        reports_root_dir=reports_root,
        layer_ids=layer_ids,
        output_dir=figures_dir
    )
    
    # Generate Figure 5(c): Coupling index distribution
    logging.info("Generating coupling index distribution (Figure 5c equivalent)...")
    create_comparison_figure(
        reports_root_dir=reports_root,
        output_dir=figures_dir
    )
    
    logging.info(f"✓ Comparison plots saved to: {figures_dir}")
    logging.info(f"  - hallucination_accuracy_comparison.png (Figure 5b)")
    logging.info(f"  - coupling_index_distribution.png (Figure 5c)")
    logging.info(f"  - coupling_index_per_layer.png (Additional analysis)")
    
except Exception as e:
    logging.warning(f"Could not generate comparison plots: {e}")
    logging.info("Continuing without plot generation...")

logging.info("="*10)
logging.info("✓ Graph Probing Analysis Complete!")
logging.info("="*10)
logging.info("Summary:")
logging.info(f"  - Dataset: {dataset_name}")
logging.info(f"  - Model: {model_name}")
logging.info(f"  - Layers analyzed: {', '.join(str(x) for x in layer_ids)}")
logging.info(f"  - Probe input type: {probe_input}")
logging.info("All steps completed successfully!")
logging.info(f"Reports saved under: {reports_root}")
logging.info(f"Figures saved under: {reports_root / 'figures'}")