"""
=================================
NEUROPATHOLOGY ANALYSIS
=================================

PURPOSE:
    This script orchestrates the 6-step neuropathology analysis pipeline:
    N1-N3: Baseline healthy network analysis (same as hallucination detection)
    N4-N6: Disease-pattern simulation and analysis
    
    1. (N1) Construct dataset and extract LLM activations
    2. (N2) Compute functional connectivity (FC) networks
    3. (N3) Train hallucination detection probes on healthy FC
    4. (N4) Transform healthy FC to simulate disease-like patterns
    5. (N5) Evaluate probes on pathological (disease) FC matrices
    6. (N6) Compute graph-theoretic metrics (healthy vs. pathological)

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
   
   ${MAIN_DIR}/reports/neuropathology_analysis/
   ├── {model_tag}/                           # e.g., Qwen_Qwen2_5_0_5B
   │   ├── {disease_pattern}/                 # e.g., epilepsy_like_d01
   │   │   ├── pipeline_config.yaml           # Copy of config file (provenance)
   │   │   ├── layer_5/                       # Per-layer subdirectory
   │   │   │   ├── N1_construct_dataset.log   # Step 1 log
   │   │   │   ├── N2_compute_network.log     # Step 2 log
   │   │   │   ├── N3_train_probe.log         # Step 3 log (healthy)
   │   │   │   ├── N4_neuropathology_connectivity.log
   │   │   │   ├── N5_neuropathology_eval.log
   │   │   │   ├── N6_neuropathology_graph_metrics.log
   │   │   │   ├── summary.json               # Step results summary
   │   │   │   ├── {step_name}/               # Artifacts by step
   │   │   │   │   ├── fc_healthy_layer_5.npy
   │   │   │   │   ├── fc_patho_epilepsy_like_layer_5.npy
   │   │   │   │   ├── probe_performance_healthy.json
   │   │   │   │   ├── probe_performance_patho.json
   │   │   │   │   ├── metrics_summary_layer_5.json
   │   │   │   │   └── ...
   │   │   │   └── tensorboard/               # TensorBoard logs
   │   │   └── ...                            # Additional layers

3. FILENAME PATTERNS FOR NEUROPATHOLOGY:

   A. Correlation Matrices (NumPy files):
      - Healthy: fc_healthy_layer_{layer_id}.npy
      - Pathological: fc_patho_{disease_pattern}_layer_{layer_id}.npy
      - Examples:
        * fc_healthy_layer_5.npy
        * fc_patho_epilepsy_like_layer_5.npy
        * fc_patho_parkinson_like_layer_5.npy

   B. Visualization Images (PNG):
      - Format: fc_{condition}_layer_{layer_id}.png
      - Examples:
        * fc_healthy_layer_5.png
        * fc_patho_epilepsy_like_layer_5.png

   C. Performance Metrics (JSON):
      - Healthy probe on healthy FC: probe_performance_healthy.json
      - Healthy probe on pathological FC: probe_performance_patho.json
      - Disease pattern: Embedded in filename
      - Fields: accuracy, precision, recall, f1, auc_roc, etc.

   D. Graph Metrics Summary:
      - metrics_summary_layer_{layer_id}.json
      - Contains: clustering_coeff, degree_dist, path_length, etc.
      - For both healthy and pathological networks

   E. Log Files:
      - Format: N{step_number}_{step_name}.log
      - Examples:
        * N1_construct_dataset.log
        * N2_compute_network.log
        * N3_train_probe.log
        * N4_neuropathology_connectivity.log
        * N5_neuropathology_eval.log
        * N6_neuropathology_graph_metrics.log

4. DISEASE PATTERN NAMING:

   Disease patterns are combined with density levels in directory names:
   - Format: {disease_pattern}_d{density_without_dot}
   - Examples:
     * epilepsy_like_d01    (from density=0.1)
     * parkinson_like_d05   (from density=0.5)
     * epilepsy_like_d10    (from density=1.0)
   - Rule: Density decimal points replaced with nothing (0.1 → 01)

5. CHECKPOINT & MODEL PATHS:

   Saved Checkpoints (same as hallucination):
   ${MAIN_DIR}/saves/
   ├── {model_tag}/
   │   └── {dataset_name}/
   │       └── {ckpt_step}/
   │           └── layer_{layer_id}/
   │               ├── best_model.pt   (healthy probe)
   │               └── final_model.pt

6. ARTIFACT ORGANIZATION:

   Step-specific artifacts grouped by phase:
   
   BASELINE (N1-N3):
   ├── construct_dataset/
   │   └── dataset_stats.json
   ├── compute_network/
   │   ├── fc_healthy_layer_5.npy
   │   ├── fc_healthy_layer_5.png
   │   └── connectivity_stats.json
   └── train_probe/
       ├── probe_performance.json   (healthy vs healthy)
       └── tensorboard/

   PATHOLOGY (N4-N6):
   ├── neuropathology_connectivity/
   │   ├── fc_patho_epilepsy_like_layer_5.npy
   │   ├── fc_patho_epilepsy_like_layer_5.png
   │   └── transformation_log.json
   ├── neuropathology_eval/
   │   └── probe_performance_patho.json   (healthy probe on pathological)
   └── neuropathology_graph_metrics/
       └── metrics_summary_layer_5.json

7. SPECIAL NAMING RULES FOR PATHOLOGY:

   - Disease patterns: epilepsy_like, parkinson_like, etc.
   - Cluster rewiring parameters: within_scale, between_scale, rewiring_prob
   - Number of clusters: num_clusters (e.g., 8)
   - Distance threshold: Optional, only if specified
   - Density formatting: Removes decimal (0.1 → 01, 0.5 → 05)

=================================
EXAMPLE FULL PATHS:
/ptmp/aomidvarnia/analysis_results/llm_graph/reports/neuropathology_analysis/
  Qwen_Qwen2_5_0_5B/
  epilepsy_like_d01/
  layer_5/
  fc_healthy_layer_5.npy
  fc_patho_epilepsy_like_layer_5.npy
  metrics_summary_layer_5.json

=================================
"""

import sys
from pathlib import Path
import logging
import os
import argparse
import json
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
import yaml


def load_config_from_yaml(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run(cmd, *, cwd: Path, env: dict, log_file: Path | None = None) -> int:
    """Stream subprocess output, filter noisy lines, and optionally tee to a log file."""
    # Force unbuffered output
    if cmd[0].endswith('python'):
        cmd = [cmd[0], '-u'] + cmd[1:]
    
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,  # Unbuffered
    )

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", encoding="utf-8")

    assert process.stdout is not None
    last_tqdm_line = ""
    for raw_line in process.stdout:
        line = raw_line
        # Skip only very specific noisy lines
        if "amdgpu.ids" in line or "torch-scatter" in line or "excessive worker creation" in line:
            continue
        # Only filter absl logging if it matches the exact format (e.g., "I1226 22:05:07...")
        if line.startswith(("I", "W", "E")) and len(line) > 15 and line[1:5].isdigit() and line[5] == ' ':
            continue
        
        # Handle tqdm progress bars
        tqdm_like = ("it/s" in line or "sample/s" in line) and ("|" in line or "%" in line)
        if tqdm_like or "\r" in line:
            clean = line.rstrip("\n")
            print(clean, end="\r", flush=True)
            if log_handle:
                log_handle.write(clean + "\n")
                log_handle.flush()
            continue
        
        # Regular output
        line = line.rstrip()
        if line:
            print(line, flush=True)
            if log_handle:
                log_handle.write(line + "\n")
                log_handle.flush()

    process.wait()
    if log_handle:
        log_handle.close()
    return process.returncode


# Defaults
default_project_dir = '/u/aomidvarnia/GIT_repositories/llm-graph-probing'
default_main_dir = '/ptmp/aomidvarnia/analysis_results/llm_graph'

# Parse arguments - only config file path is required
parser = argparse.ArgumentParser(description="Neuropathology Simulation Analysis - Load all parameters from config file")
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

# Extract common and neuropathology-specific parameters from config
common_cfg = config.get('common', {})
neuro_cfg = config.get('neuropathology', {})

# Build parameters from config with optional command-line overrides
main_dir = Path(common_cfg.get('main_dir') or args.main_dir or default_main_dir).resolve()
project_dir = Path(common_cfg.get('project_dir') or args.project_dir or default_project_dir).resolve()
dataset_name = common_cfg.get('dataset_name', 'truthfulqa')
llm_model_name = common_cfg.get('model_name', 'gpt2')
ckpt_step = int(common_cfg.get('ckpt_step', -1))
layer_list = common_cfg.get('layer_list', '5')
density = float(common_cfg.get('density', 0.05))
num_layers = int(common_cfg.get('num_layers', 2))
hidden_channels = int(common_cfg.get('hidden_channels', 32))
gpu_id = int(common_cfg.get('gpu_id', 0))
early_stop_patience = int(common_cfg.get('early_stop_patience', 20))
disease_pattern = neuro_cfg.get('disease_pattern', 'epilepsy_like')
num_clusters = int(neuro_cfg.get('num_clusters', 8))
within_scale = float(neuro_cfg.get('within_scale', 1.2))
between_scale = float(neuro_cfg.get('between_scale', 0.7))
rewiring_prob = float(neuro_cfg.get('rewiring_prob', 0.15))
distance_threshold = neuro_cfg.get('distance_threshold')
if distance_threshold is not None:
    distance_threshold = int(distance_threshold) if distance_threshold else None
from_sparse_data = common_cfg.get('from_sparse_data', True)
if isinstance(from_sparse_data, str):
    from_sparse_data = from_sparse_data.lower() in ('true', '1', 'yes')
aggregate_layers = common_cfg.get('aggregate_layers', False)
if isinstance(aggregate_layers, str):
    aggregate_layers = aggregate_layers.lower() in ('true', '1', 'yes')
sys.path.insert(0, str(project_dir))

env = os.environ.copy()
env['PYTHONPATH'] = str(project_dir)
env['MAIN_DIR'] = str(main_dir)
python_exe = sys.executable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Neuropathology Simulation Analysis")

layer_list = getattr(args, "layer_list", "").strip()
layer_ids = [int(x) for x in layer_list.replace(',', ' ').split() if x.strip()]
primary_layer = layer_ids[0]


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Sanitize model name: replace '/' and '-' with '_'
model_tag = llm_model_name.replace('/', '_').replace('-', '_')
model_reports_root = main_dir / "reports" / "neuropathology_analysis" / model_tag
model_reports_root.mkdir(parents=True, exist_ok=True)

# Copy pipeline config into the model folder for provenance
cfg_src = project_dir / "pipeline_config.yaml"
try:
    if cfg_src.exists():
        shutil.copy2(str(cfg_src), str(model_reports_root / "pipeline_config.yaml"))
except Exception:
    pass

# Layer-specific reports directory (with disease subfolder)
reports_dir = model_reports_root / f"layer_{primary_layer}" / disease_pattern
reports_dir.mkdir(parents=True, exist_ok=True)

step_results = []

# N1: Construct dataset
logging.info("="*10)
logging.info("N1: Constructing dataset (truthfulqa.csv)...")
step_log = reports_dir / "n1_construct_dataset.log"
start = time.monotonic()
rc = run([python_exe, "hallucination/construct_dataset.py"], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N1", "name": "construct_dataset", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    dataset_path = main_dir / "data/hallucination/truthfulqa.csv"
    step_results.append({"step": "N1", "name": "construct_dataset", "status": "ok", "duration_sec": dur, "log": str(step_log), "dataset_path": str(dataset_path)})

# N2: Compute healthy connectivity
logging.info("="*10)
logging.info("N2: Computing healthy functional connectivity...")
logging.info("This step may take several minutes depending on dataset size...")
step_log = reports_dir / "n2_compute_network.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.compute_llm_network",
    f"--dataset_name={dataset_name}",
    f"--llm_model_name={llm_model_name}",
    f"--ckpt_step={ckpt_step}",
    *[f"--llm_layer={lid}" for lid in layer_ids],
    f"--batch_size=16",
    f"--network_density={density}",
    f"--gpu_id={gpu_id}",
] + (["--sparse"] if from_sparse_data else [])
    + (["--aggregate_layers"] if aggregate_layers else []), cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N2", "name": "compute_network", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    step_results.append({"step": "N2", "name": "compute_network", "status": "ok", "duration_sec": dur, "log": str(step_log)})

# N3: Train probe on healthy graphs
logging.info("="*10)
logging.info("N3: Training GNN probe on healthy graphs...")
logging.info("This step may take several minutes...")
step_log = reports_dir / "n3_train.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.train",
    f"--dataset_name={dataset_name}",
    f"--llm_model_name={llm_model_name}",
    f"--ckpt_step={ckpt_step}",
    f"--llm_layer={primary_layer}",
    f"--probe_input=corr",
    f"--density={density}",
    f"--batch_size=16",
    f"--eval_batch_size=32",
    f"--num_layers={num_layers}",
    f"--hidden_channels={hidden_channels}",
    f"--lr=0.001",
    f"--num_epochs=10",
    f"--early_stop_patience={early_stop_patience}",
    f"--gpu_id={gpu_id}",
] + (["--from_sparse_data"] if from_sparse_data else []), cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N3", "name": "train", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    # Sanitize model name: replace '/' and '-' with '_'
    model_dir = llm_model_name.replace('/', '_').replace('-', '_') if ckpt_step == -1 else f"{llm_model_name.replace('/', '_').replace('-', '_')}_step{ckpt_step}"
    density_tag = f"{int(round(density * 100)):02d}"
    best_model = main_dir / "saves" / f"hallucination/{dataset_name}/{model_dir}" / f"layer_{primary_layer}" / f"best_model_density-{density_tag}_dim-{hidden_channels}_hop-{num_layers}_input-corr.pth"
    step_results.append({"step": "N3", "name": "train", "status": "ok", "duration_sec": dur, "log": str(step_log), "best_model": str(best_model)})

# N4: Generate pathological connectivity
logging.info("="*10)
logging.info("N4: Generating pathological connectivity...")
logging.info("Transforming healthy FC matrices to simulate disease-like patterns...")
step_log = reports_dir / "n4_neuropathology_connectivity.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_connectivity",
    f"--dataset_name={dataset_name}",
    f"--llm_model_name={llm_model_name}",
    f"--ckpt_step={ckpt_step}",
    f"--llm_layer={primary_layer}",
    f"--density={density}",
    f"--disease_pattern={disease_pattern}",
    f"--num_clusters={num_clusters}",
    f"--within_scale={within_scale}",
    f"--between_scale={between_scale}",
    f"--rewiring_prob={rewiring_prob}",
    f"--distance_threshold={distance_threshold}",
], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N4", "name": "neuropathology_connectivity", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    artifacts = {
        "group_fc_healthy": str(reports_dir / f"group_fc_healthy_layer_{primary_layer}.npy"),
        "group_fc_patho": str(reports_dir / f"group_fc_patho_{disease_pattern}_layer_{primary_layer}.npy"),
        "group_fc_healthy_png": str(reports_dir / f"group_fc_healthy_layer_{primary_layer}.png"),
        "group_fc_patho_png": str(reports_dir / f"group_fc_patho_{disease_pattern}_layer_{primary_layer}.png"),
    }
    step_results.append({"step": "N4", "name": "neuropathology_connectivity", "status": "ok", "duration_sec": dur, "log": str(step_log), "artifacts": artifacts})

# N5: Evaluate probe on healthy vs pathological
logging.info("="*10)
logging.info("N5: Evaluating probe behavior (healthy vs patho)...")
step_log = reports_dir / "n5_neuropathology_eval.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_eval",
    f"--dataset_name={dataset_name}",
    f"--llm_model_name={llm_model_name}",
    f"--ckpt_step={ckpt_step}",
    f"--llm_layer={primary_layer}",
    f"--probe_input=corr",
    f"--density={density}",
    f"--disease_pattern={disease_pattern}",
    f"--num_layers={num_layers}",
    f"--hidden_channels={hidden_channels}",
    f"--gpu_id={gpu_id}",
], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N5", "name": "neuropathology_eval", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    eval_summary = reports_dir / "summary_eval.json"
    delta_prob = None
    try:
        if eval_summary.exists():
            s = json.loads(eval_summary.read_text())
            delta_prob = s.get("delta_probs_mean")
    except Exception:
        pass
    step_results.append({"step": "N5", "name": "neuropathology_eval", "status": "ok", "duration_sec": dur, "log": str(step_log), "delta_probs_mean": delta_prob, "summary": str(eval_summary)})

# N6: Graph metrics comparison
logging.info("="*10)
logging.info("N6: Computing graph-theoretic metrics (healthy vs patho)...")
step_log = reports_dir / "n6_neuropathology_graph_metrics.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_graph_metrics",
    f"--llm_model_name={llm_model_name}",
    f"--ckpt_step={ckpt_step}",
    f"--llm_layer={primary_layer}",
    f"--dataset_name={dataset_name}",
    f"--density={density}",
    f"--disease_pattern={disease_pattern}",
    f"--num_clusters={num_clusters}",
], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N6", "name": "neuropathology_graph_metrics", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    summary_metrics = reports_dir / f"metrics_summary_layer_{primary_layer}.json"
    step_results.append({"step": "N6", "name": "neuropathology_graph_metrics", "status": "ok", "duration_sec": dur, "log": str(step_log), "metrics_summary": str(summary_metrics)})


# Persist final summary
summary_json = reports_dir / "summary.json"
try:
    durations = [(r.get("name"), r.get("duration_sec", 0.0)) for r in step_results if "duration_sec" in r]
    if durations:
        names, vals = zip(*durations)
        plt.barh(names, vals)
        plt.xlabel("seconds")
        plt.title("Neuropathology step durations")
        plt.tight_layout()
        plt.savefig(reports_dir / "step_durations.png")
        plt.close()
    save_json(summary_json, {"steps": step_results})
except Exception as e:
    logging.warning(f"Could not write summary: {e}")

logging.info("="*10)
logging.info("✓ Neuropathology Simulation Analysis Complete!")
logging.info("="*10)
logging.info(f"Reports saved to: {reports_dir}")
