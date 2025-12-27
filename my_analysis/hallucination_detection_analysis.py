import sys
from pathlib import Path
import logging
import os
import argparse
import json
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from ptpython.repl import embed


def run(cmd, *, cwd: Path, env: dict, log_file: Path | None = None) -> int:
    """Stream subprocess output, filter noisy lines, and optionally tee to a log file."""
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
    
    for raw_line in process.stdout:
        line = raw_line
        # Skip noisy lines
        if "amdgpu.ids" in line:
            continue
        # Skip absl logging format (e.g., "I1221 23:19:42.675042 123456")
        if line.startswith(("I", "W", "E")) and len(line) > 4 and line[1].isdigit():
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
    return process.returncode

# --------------
# Ref: https://github.com/omidvarnia/llm-graph-probing
# --------------

# Defaults
default_project_dir = '/u/aomidvarnia/GIT_repositories/llm-graph-probing'
default_main_dir = '/ptmp/aomidvarnia/analysis_results/llm_graph'

# Parse arguments (allow overriding main directory for data/outputs and project location)
parser = argparse.ArgumentParser(description="Graph Probing Analysis")
parser.add_argument("--main_dir", type=str, default=str(default_main_dir), help="Root directory to save data and results")
parser.add_argument("--project_dir", type=str, default=str(default_project_dir), help="Project directory containing code")
parser.add_argument("--dataset_name", type=str, default="truthfulqa", help="Dataset name: truthfulqa, halueval, medhallu, helm")
parser.add_argument("--model_name", type=str, default="gpt2", help="Model name: gpt2, gpt2-medium, gpt2-large, pythia-160m, etc.")
parser.add_argument("--ckpt_step", type=int, default=-1, help="Checkpoint step (-1 for main checkpoint)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for LLM inference")
parser.add_argument("--layer_id", type=int, default=5, help="LLM layer to analyze")
parser.add_argument("--probe_input", type=str, default="corr", help="Probe input type: corr or activation")
parser.add_argument("--network_density", type=float, default=0.05, help="Network density (0.01 to 1.0)")
parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
parser.add_argument("--num_channels", type=int, default=32, help="Hidden channels in GNN")
parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--from_sparse_data", action="store_true", default=True, help="Use sparse data representation")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--early_stop_patience", type=int, default=20, help="Early stopping patience (epochs without improvement)")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
args, unknown = parser.parse_known_args()

main_dir = Path(args.main_dir).resolve()
project_dir = Path(args.project_dir).resolve()

# Add the project root to the Python path
sys.path.insert(0, str(project_dir))

# Set up environment for subprocess calls
env = os.environ.copy()
# Set PYTHONPATH to include project root, ensuring utils package is found before local utils.py
env['PYTHONPATH'] = str(project_dir)
env['MAIN_DIR'] = str(main_dir)
# Ensure we use the same Python executable
python_exe = sys.executable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Graph Probing Analysis")
# embed(globals(), locals())





# -----------------------------
# Analysis Pipeline Configuration
# -----------------------------
dataset_name = args.dataset_name
model_name = args.model_name
ckpt_step = args.ckpt_step
batch_size = args.batch_size
layer_id = args.layer_id
probe_input = args.probe_input
network_density = args.network_density
eval_batch_size = args.eval_batch_size
num_channels = args.num_channels
num_layers = args.num_layers
learning_rate = args.learning_rate
from_sparse_data = args.from_sparse_data
num_epochs = args.num_epochs
early_stop_patience = args.early_stop_patience

# Output directories for interim artifacts
reports_dir = main_dir / "reports" / "hallucination_analysis"
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

# -----------------------------
# Step 1: Prepare the dataset
# -----------------------------
logging.info("\n" + "="*60)
logging.info("Step 1: Constructing dataset...")
logging.info(f"  Dataset: {dataset_name}")
logging.info(f"  Model: {model_name}")
logging.info(f"  Checkpoint step: {ckpt_step}")
logging.info(f"  Batch size: {batch_size}")
logging.info(f"Using Python: {sys.executable}")
logging.info("Executing construct_dataset.py...")
step1_log = reports_dir / "step1_construct_dataset.log"
step_start = time.monotonic()
result = run(
    [
        python_exe,
        "hallucination/construct_dataset.py"
    ],
    cwd=project_dir,
    env=env,
    log_file=step1_log,
)
step_duration = time.monotonic() - step_start

if result != 0:
    logging.error(f"Dataset construction failed with return code {result}")
    step_results.append({"step": 1, "name": "construct_dataset", "status": "error", "duration_sec": step_duration, "log": str(step1_log)})
else:
    logging.info("✓ Dataset constructed successfully")
    logging.info("Dataset is ready for processing")
    dataset_path = main_dir / "data/hallucination/truthfulqa.csv"
    label_plot = reports_dir / "dataset_label_distribution.png"
    dataset_head = reports_dir / "dataset_head.csv"
    try:
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            df.head(20).to_csv(dataset_head, index=False)
            if "label" in df.columns:
                label_counts = df["label"].value_counts().sort_index()
                label_counts.plot(kind="bar")
                plt.title("TruthfulQA label distribution")
                plt.xlabel("label")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(label_plot)
                plt.close()
        step_results.append({"step": 1, "name": "construct_dataset", "status": "ok", "duration_sec": step_duration, "log": str(step1_log), "dataset_path": str(dataset_path), "dataset_head": str(dataset_head), "label_plot": str(label_plot)})
    except Exception as e:
        logging.warning(f"Could not summarize dataset: {e}")
        step_results.append({"step": 1, "name": "construct_dataset", "status": "ok", "duration_sec": step_duration, "log": str(step1_log), "dataset_path": str(dataset_path)})

# -----------------------------
# Step 2: Generate the neural topology
# -----------------------------
logging.info("\n" + "="*60)
logging.info("\nStep 2: Generating neural topology (network graph)...")
logging.info(f"  Layer ID: {layer_id}")
logging.info(f"  Network density: {network_density}")
logging.info(f"  Sparse mode: {from_sparse_data}")
logging.info("Executing compute_llm_network.py...")
step2_log = reports_dir / "step2_compute_network.log"
step_start = time.monotonic()
result = run(
    [
        python_exe,
        "-m",
        "hallucination.compute_llm_network",
        f"--dataset_name={dataset_name}",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--llm_layer={layer_id}",
        f"--batch_size={batch_size}",
        f"--network_density={network_density}",
        f"--gpu_id={args.gpu_id}",
    ] + (["--sparse"] if from_sparse_data else []),
    cwd=project_dir,
    env=env,
    log_file=step2_log,
)
step_duration = time.monotonic() - step_start

if result != 0:
    logging.error(f"Network computation failed with return code {result}")
    step_results.append({"step": 2, "name": "compute_network", "status": "error", "duration_sec": step_duration, "log": str(step2_log)})
else:
    logging.info("✓ Neural topology (network graph) computed successfully")
    logging.info("Network topology is ready for probe training")
    try:
        if ckpt_step == -1:
            model_dir = model_name
        else:
            model_dir = f"{model_name}_step{ckpt_step}"
        topology_root = main_dir / "data/hallucination/truthfulqa" / model_dir
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

        step_results.append({"step": 2, "name": "compute_network", "status": "ok", "duration_sec": step_duration, "log": str(step2_log), "topology_root": str(topology_root), "num_questions": num_questions, "fc_before": str(fc_before), "fc_after": str(fc_after)})
    except Exception as e:
        logging.warning(f"Could not summarize network outputs: {e}")
        step_results.append({"step": 2, "name": "compute_network", "status": "ok", "duration_sec": step_duration, "log": str(step2_log)})

# -----------------------------
# Step 3: Train the probes
# -----------------------------
logging.info("\n" + "="*60)
logging.info("\nStep 3: Training graph neural network probes...")
logging.info(f"  Probe input: {probe_input}")
logging.info(f"  Density: {network_density}")
logging.info(f"  Num channels: {num_channels}")
logging.info(f"  Num layers: {num_layers}")
logging.info(f"  Learning rate: {learning_rate}")
logging.info(f"  Batch size: {batch_size}")
logging.info(f"  Eval batch size: {eval_batch_size}")
logging.info(f"  Num epochs: {num_epochs}")
logging.info(f"  Early stop patience: {early_stop_patience}")
logging.info("Executing train.py...")
logging.info("This may take a while...")
step3_log = reports_dir / "step3_train.log"
step_start = time.monotonic()
result = run(
    [
        python_exe,
        "-m",
        "hallucination.train",
        f"--dataset_name={dataset_name}",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--llm_layer={layer_id}",
        f"--probe_input={probe_input}",
        f"--density={network_density}",
        f"--batch_size={batch_size}",
        f"--eval_batch_size={eval_batch_size}",
        f"--num_layers={num_layers}",
        f"--hidden_channels={num_channels}",
        f"--lr={learning_rate}",
        f"--num_epochs={num_epochs}",
        f"--early_stop_patience={early_stop_patience}",
        f"--gpu_id={args.gpu_id}",
    ] + (["--from_sparse_data"] if from_sparse_data else []),
    cwd=project_dir,
    env=env,
    log_file=step3_log,
)
step_duration = time.monotonic() - step_start

if result != 0:
    logging.error(f"Training failed with return code {result}")
    step_results.append({"step": 3, "name": "train", "status": "error", "duration_sec": step_duration, "log": str(step3_log)})
else:
    logging.info("✓ Probes trained successfully")
    logging.info("Trained models are ready for evaluation")
    logging.info("\nNote: The FC (functional connectivity) matrix is STATIC and does NOT change during training.")
    logging.info("      It is computed from the LLM's hidden activations. The probe (GCN/MLP) learns to")
    logging.info("      extract hallucination signals from this fixed graph structure.")
    if ckpt_step == -1:
        model_dir = model_name
    else:
        model_dir = f"{model_name}_step{ckpt_step}"
    best_model_path = main_dir / "saves" / f"hallucination/truthfulqa/{model_dir}" / f"layer_{layer_id}" / f"best_model_density-{network_density}_dim-{num_channels}_hop-{num_layers}_input-{probe_input}.pth"
    model_size = best_model_path.stat().st_size if best_model_path.exists() else 0
    # TensorBoard scalars
    tb_dir = main_dir / "runs" / f"hallucination/truthfulqa/{model_dir}" / f"layer_{layer_id}"
    loss_plot = reports_dir / "train_loss.png"
    metrics_plot = reports_dir / "test_metrics.png"
    try:
        event_files = sorted(tb_dir.glob("events.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if event_files:
            ea = event_accumulator.EventAccumulator(str(event_files[0]))
            ea.Reload()
            if "train/loss" in ea.Tags().get("scalars", []):
                loss_events = [(e.step, e.value) for e in ea.Scalars("train/loss")]
                _plot_series(loss_events, f"Train Loss ({model_name} layer {layer_id})", "loss", loss_plot)
            metrics = {}
            for tag in ["test/accuracy", "test/precision", "test/recall", "test/f1"]:
                if tag in ea.Tags().get("scalars", []):
                    metrics[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
            if metrics:
                plt.figure(figsize=(7, 5))
                for tag, series in metrics.items():
                    xs, ys = zip(*series)
                    plt.plot(xs, ys, marker="o", label=tag.split("/")[-1])
                plt.title(f"Test Metrics ({model_name} layer {layer_id})")
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
logging.info("\n" + "="*60)
logging.info("\nStep 4: Evaluating trained probes...")
logging.info(f"  Network density: {network_density}")
logging.info("Executing eval.py...")
logging.info("Computing evaluation metrics...")
step4_log = reports_dir / "step4_eval.log"
step_start = time.monotonic()
result = run(
    [
        python_exe,
        "-m",
        "hallucination.eval",
        f"--dataset_name={dataset_name}",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--llm_layer={layer_id}",
        f"--probe_input={probe_input}",
        f"--density={network_density}",
        f"--num_layers={num_layers}",
        f"--gpu_id={args.gpu_id}",
    ],
    cwd=project_dir,
    env=env,
    log_file=step4_log,
)
step_duration = time.monotonic() - step_start

if result != 0:
    logging.error(f"Evaluation failed with return code {result}")
    step_results.append({"step": 4, "name": "eval", "status": "error", "duration_sec": step_duration, "log": str(step4_log)})
else:
    logging.info("✓ Evaluation completed successfully")
    logging.info("Evaluation results are ready")
    step_results.append({"step": 4, "name": "eval", "status": "ok", "duration_sec": step_duration, "log": str(step4_log)})

# -----------------------------
# Step 5: Graph statistics
# -----------------------------
logging.info("\n" + "="*60)
logging.info("\nStep 5: Computing graph statistics...")
logging.info(f"  Layer ID: {layer_id}")
logging.info(f"  Network density: {network_density}")
logging.info("Executing graph_analysis.py...")
step5_log = reports_dir / "step5_graph_analysis.log"
step_start = time.monotonic()
result = run(
    [
        python_exe,
        "-m",
        "hallucination.graph_analysis",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--layer={layer_id}",
        f"--feature={probe_input}",
    ],
    cwd=project_dir,
    env=env,
    log_file=step5_log,
)
step_duration = time.monotonic() - step_start

if result != 0:
    logging.error(f"Graph analysis failed with return code {result}")
    step_results.append({"step": 5, "name": "graph_analysis", "status": "error", "duration_sec": step_duration, "log": str(step5_log)})
else:
    logging.info("✓ Graph analysis completed successfully")
    logging.info("Graph analysis results are ready")
    npy_file = project_dir / f"{model_name}_layer_{layer_id}_{probe_input}_intra_vs_inter.npy"
    hist_path = reports_dir / "intra_vs_inter_hist.png"
    summary_csv = reports_dir / "intra_vs_inter_summary.csv"
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
                step_results.append({"step": 5, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "npy_file": str(npy_file), "histogram": str(hist_path), "summary_csv": str(summary_csv), "stats": stats})
            else:
                step_results.append({"step": 5, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "npy_file": str(npy_file), "note": "empty data"})
        else:
            step_results.append({"step": 5, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log), "note": "npy file not found"})
    except Exception as e:
        logging.warning(f"Could not summarize graph analysis outputs: {e}")
        step_results.append({"step": 5, "name": "graph_analysis", "status": "ok", "duration_sec": step_duration, "log": str(step5_log)})

# -----------------------------
# Persist summary and quick visuals
# -----------------------------
summary_json = reports_dir / "summary.json"
try:
    durations = [(r.get("name"), r.get("duration_sec", 0.0)) for r in step_results if "duration_sec" in r]
    if durations:
        names, vals = zip(*durations)
        plt.barh(names, vals)
        plt.xlabel("seconds")
        plt.title("Step durations")
        plt.tight_layout()
        plt.savefig(reports_dir / "step_durations.png")
        plt.close()
    save_json(summary_json, {"steps": step_results})
except Exception as e:
    logging.warning(f"Could not write summary: {e}")

logging.info("\n" + "="*60)
logging.info("✓ Graph Probing Analysis Complete!")
logging.info("="*60)
logging.info("Summary:")
logging.info(f"  - Dataset: {dataset_name}")
logging.info(f"  - Model: {model_name}")
logging.info(f"  - Layer analyzed: {layer_id}")
logging.info(f"  - Probe input type: {probe_input}")
logging.info("All steps completed successfully!")
logging.info(f"Reports saved to: {reports_dir}")