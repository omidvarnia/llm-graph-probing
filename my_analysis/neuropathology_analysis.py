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

parser = argparse.ArgumentParser(description="Neuropathology Simulation Analysis")
parser.add_argument("--main_dir", type=str, default=str(default_main_dir), help="Root directory to save data and results")
parser.add_argument("--project_dir", type=str, default=str(default_project_dir), help="Project directory containing code")
parser.add_argument("--dataset_name", type=str, default="truthfulqa")
parser.add_argument("--llm_model_name", type=str, default="gpt2")
parser.add_argument("--ckpt_step", type=int, default=-1)
parser.add_argument("--llm_layer", type=int, default=5)
parser.add_argument("--density", type=float, default=0.05)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--hidden_channels", type=int, default=32)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--early_stop_patience", type=int, default=20)
parser.add_argument("--disease_pattern", type=str, default="epilepsy_like")
parser.add_argument("--num_clusters", type=int, default=8)
parser.add_argument("--within_scale", type=float, default=1.3)
parser.add_argument("--between_scale", type=float, default=0.5)
parser.add_argument("--rewiring_prob", type=float, default=0.15)
parser.add_argument("--distance_threshold", type=int, default=50)
parser.add_argument("--from_sparse_data", action="store_true")
args, unknown = parser.parse_known_args()

main_dir = Path(args.main_dir).resolve()
project_dir = Path(args.project_dir).resolve()
sys.path.insert(0, str(project_dir))

env = os.environ.copy()
env['PYTHONPATH'] = str(project_dir)
env['MAIN_DIR'] = str(main_dir)
python_exe = sys.executable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting Neuropathology Simulation Analysis")


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


reports_dir = main_dir / "reports" / "neuropathology_analysis" / args.disease_pattern
reports_dir.mkdir(parents=True, exist_ok=True)

step_results = []

# N1: Construct dataset
logging.info("\n" + "="*60)
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
logging.info("\n" + "="*60)
logging.info("N2: Computing healthy functional connectivity...")
logging.info("This step may take several minutes depending on dataset size...")
step_log = reports_dir / "n2_compute_network.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.compute_llm_network",
    f"--dataset_name={args.dataset_name}",
    f"--llm_model_name={args.llm_model_name}",
    f"--ckpt_step={args.ckpt_step}",
    f"--llm_layer={args.llm_layer}",
    f"--batch_size=16",
    f"--network_density={args.density}",
    f"--gpu_id={args.gpu_id}",
] + (["--sparse"] if args.from_sparse_data else []), cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N2", "name": "compute_network", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    step_results.append({"step": "N2", "name": "compute_network", "status": "ok", "duration_sec": dur, "log": str(step_log)})

# N3: Train probe on healthy graphs
logging.info("\n" + "="*60)
logging.info("N3: Training GNN probe on healthy graphs...")
logging.info("This step may take several minutes...")
step_log = reports_dir / "n3_train.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.train",
    f"--dataset_name={args.dataset_name}",
    f"--llm_model_name={args.llm_model_name}",
    f"--ckpt_step={args.ckpt_step}",
    f"--llm_layer={args.llm_layer}",
    f"--probe_input=corr",
    f"--density={args.density}",
    f"--batch_size=16",
    f"--eval_batch_size=32",
    f"--num_layers={args.num_layers}",
    f"--hidden_channels={args.hidden_channels}",
    f"--lr=0.001",
    f"--num_epochs=10",
    f"--early_stop_patience={args.early_stop_patience}",
    f"--gpu_id={args.gpu_id}",
] + (["--from_sparse_data"] if args.from_sparse_data else []), cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N3", "name": "train", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    model_dir = args.llm_model_name if args.ckpt_step == -1 else f"{args.llm_model_name}_step{args.ckpt_step}"
    best_model = main_dir / "saves" / f"hallucination/{args.dataset_name}/{model_dir}" / f"layer_{args.llm_layer}" / f"best_model_density-{args.density}_dim-{args.hidden_channels}_hop-{args.num_layers}_input-corr.pth"
    step_results.append({"step": "N3", "name": "train", "status": "ok", "duration_sec": dur, "log": str(step_log), "best_model": str(best_model)})

# N4: Generate pathological connectivity
logging.info("\n" + "="*60)
logging.info("N4: Generating pathological connectivity...")
logging.info("Transforming healthy FC matrices to simulate disease-like patterns...")
step_log = reports_dir / "n4_neuropathology_connectivity.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_connectivity",
    f"--dataset_name={args.dataset_name}",
    f"--llm_model_name={args.llm_model_name}",
    f"--ckpt_step={args.ckpt_step}",
    f"--llm_layer={args.llm_layer}",
    f"--density={args.density}",
    f"--disease_pattern={args.disease_pattern}",
    f"--num_clusters={args.num_clusters}",
    f"--within_scale={args.within_scale}",
    f"--between_scale={args.between_scale}",
    f"--rewiring_prob={args.rewiring_prob}",
    f"--distance_threshold={args.distance_threshold}",
], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N4", "name": "neuropathology_connectivity", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    artifacts = {
        "group_fc_healthy": str(reports_dir / f"group_fc_healthy_layer_{args.llm_layer}.npy"),
        "group_fc_patho": str(reports_dir / f"group_fc_patho_{args.disease_pattern}_layer_{args.llm_layer}.npy"),
        "group_fc_healthy_png": str(reports_dir / f"group_fc_healthy_layer_{args.llm_layer}.png"),
        "group_fc_patho_png": str(reports_dir / f"group_fc_patho_{args.disease_pattern}_layer_{args.llm_layer}.png"),
    }
    step_results.append({"step": "N4", "name": "neuropathology_connectivity", "status": "ok", "duration_sec": dur, "log": str(step_log), "artifacts": artifacts})

# N5: Evaluate probe on healthy vs pathological
logging.info("\n" + "="*60)
logging.info("N5: Evaluating probe behavior (healthy vs patho)...")
step_log = reports_dir / "n5_neuropathology_eval.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_eval",
    f"--dataset_name={args.dataset_name}",
    f"--llm_model_name={args.llm_model_name}",
    f"--ckpt_step={args.ckpt_step}",
    f"--llm_layer={args.llm_layer}",
    f"--probe_input=corr",
    f"--density={args.density}",
    f"--disease_pattern={args.disease_pattern}",
    f"--num_layers={args.num_layers}",
    f"--hidden_channels={args.hidden_channels}",
    f"--gpu_id={args.gpu_id}",
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
logging.info("\n" + "="*60)
logging.info("N6: Computing graph-theoretic metrics (healthy vs patho)...")
step_log = reports_dir / "n6_neuropathology_graph_metrics.log"
start = time.monotonic()
rc = run([
    python_exe, "-m", "hallucination.neuropathology_graph_metrics",
    f"--llm_model_name={args.llm_model_name}",
    f"--ckpt_step={args.ckpt_step}",
    f"--llm_layer={args.llm_layer}",
    f"--dataset_name={args.dataset_name}",
    f"--density={args.density}",
    f"--disease_pattern={args.disease_pattern}",
    f"--num_clusters={args.num_clusters}",
], cwd=project_dir, env=env, log_file=step_log)
dur = time.monotonic() - start
if rc != 0:
    step_results.append({"step": "N6", "name": "neuropathology_graph_metrics", "status": "error", "duration_sec": dur, "log": str(step_log)})
else:
    summary_metrics = reports_dir / f"metrics_summary_layer_{args.llm_layer}.json"
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

logging.info("\n" + "="*60)
logging.info("✓ Neuropathology Simulation Analysis Complete!")
logging.info("="*60)
logging.info(f"Reports saved to: {reports_dir}")
