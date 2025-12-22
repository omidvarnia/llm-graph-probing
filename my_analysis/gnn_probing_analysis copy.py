import sys
from pathlib import Path
import logging
import os
import argparse
from ptpython.repl import embed
import subprocess


def run(cmd, *, cwd: Path, env: dict) -> int:
    """Stream subprocess output, filtering noisy amdgpu.ids and redundant logging lines."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        # Skip noisy lines
        if "amdgpu.ids" in line:
            continue
        # Skip absl logging format (e.g., "I1221 23:19:42.675042 123456")
        if line.startswith(("I", "W", "E")) and len(line) > 4 and line[1].isdigit():
            continue

        # Handle tqdm carriage-return updates on a single line
        if "\r" in line:
            segment = line.rstrip("\n").split("\r")[-1]
            if segment:
                print(segment, end="\r", flush=True)
            continue

        # Strip and print non-empty lines
        line = line.rstrip()
        if line:
            print(line)

    process.wait()
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
dataset_name = "openwebtext"
model_name = "gpt2"  # Valid options: gpt2, gpt2-medium, gpt2-large, pythia-160m, etc.
ckpt_step = -1  # -1 for main checkpoint, or specific step number for finetuned models
batch_size = 16
layer_id = 5
probe_input = "corr" # corr or activation
network_density = 0.05
eval_batch_size = 32
num_channels = 32
num_layers = 3
learning_rate = 0.001
from_sparse_data = True
num_epochs = 20

# -----------------------------
# Step 1: Prepare the dataset
# -----------------------------
logging.info("Step 1: Constructing dataset...")
logging.info(f"  Dataset: {dataset_name}")
logging.info(f"  Model: {model_name}")
logging.info(f"  Checkpoint step: {ckpt_step}")
logging.info(f"  Batch size: {batch_size}")
logging.info(f"Using Python: {sys.executable}")
logging.info("Executing construct_dataset.py...")
# result = run(
#     [
#         python_exe,
#         "graph_probing/construct_dataset.py",
#         f"--dataset={dataset_name}",
#         f"--llm_model_name={model_name}",
#         f"--ckpt_step={ckpt_step}",
#         f"--batch_size={batch_size}",
#         "--gpu_id=0",
#     ],
#     cwd=project_dir,
#     env=env,
# )

# if result != 0:
#     logging.error(f"Dataset construction failed with return code {result}")
# else:
#     logging.info("✓ Dataset constructed successfully")
#     logging.info("Dataset is ready for processing")

# -----------------------------
# Step 2: Generate the neural topology
# -----------------------------
logging.info("\nStep 2: Generating neural topology (network graph)...")
logging.info(f"  Layer ID: {layer_id}")
logging.info(f"  Network density: {network_density}")
logging.info("Executing compute_llm_network.py...")
# result = run(
#     [
#         python_exe,
#         "graph_probing/compute_llm_network.py",
#         f"--dataset={dataset_name}",
#         f"--llm_model_name={model_name}",
#         f"--ckpt_step={ckpt_step}",
#         f"--llm_layer={layer_id}",
#         f"--batch_size={batch_size}",
#         f"--network_density={network_density}",
#         "--gpu_id=0",
#     ],
#     cwd=project_dir,
#     env=env,
# )

# if result != 0:
#     logging.error(f"Network computation failed with return code {result}")
# else:
#     logging.info("✓ Neural topology (network graph) computed successfully")
#     logging.info("Network topology is ready for probe training")

# -----------------------------
# Step 3: Train the probes
# -----------------------------
logging.info("\nStep 3: Training graph neural network probes...")
logging.info(f"  Probe input: {probe_input}")
logging.info(f"  Density: {network_density}")
logging.info(f"  Num channels: {num_channels}")
logging.info(f"  Num layers: {num_layers}")
logging.info(f"  Learning rate: {learning_rate}")
logging.info(f"  Batch size: {batch_size}")
logging.info(f"  Eval batch size: {eval_batch_size}")
logging.info(f"  Num epochs: {num_epochs}")
logging.info("Executing train.py...")
logging.info("This may take a while...")
result = run(
    [
        python_exe,
        "graph_probing/train.py",
        f"--dataset={dataset_name}",
        f"--probe_input={probe_input}",
        f"--density={network_density}",
        f"--from_sparse_data={from_sparse_data}",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--llm_layer={layer_id}",
        f"--batch_size={batch_size}",
        f"--eval_batch_size={eval_batch_size}",
        "--nonlinear_activation",
        f"--num_channels={num_channels}",
        f"--num_layers={num_layers}",
        f"--num_epochs={num_epochs}",
        f"--lr={learning_rate}",
        "--in_memory",
        "--gpu_id=0",
    ],
    cwd=project_dir,
    env=env,
)

if result != 0:
    logging.error(f"Training failed with return code {result}")
else:
    logging.info("✓ Probes trained successfully")
    logging.info("Trained models are ready for evaluation")

# -----------------------------
# Step 4: Evaluate the probes
# -----------------------------
logging.info("\nStep 4: Evaluating trained probes...")
logging.info(f"  Network density: {network_density}")
logging.info("Executing eval.py...")
logging.info("Computing evaluation metrics...")
result = run(
    [
        python_exe,
        "graph_probing/eval.py",
        f"--dataset={dataset_name}",
        f"--probe_input={probe_input}",
        f"--density={network_density}",
        f"--from_sparse_data={from_sparse_data}",
        f"--llm_model_name={model_name}",
        f"--ckpt_step={ckpt_step}",
        f"--llm_layer={layer_id}",
        f"--batch_size={batch_size}",
        f"--eval_batch_size={eval_batch_size}",
        "--nonlinear_activation",
        f"--num_channels={num_channels}",
        f"--num_layers={num_layers}",
        "--in_memory",
        "--gpu_id=0",
    ],
    cwd=project_dir,
    env=env,
)

if result != 0:
    logging.error(f"Evaluation failed with return code {result}")
else:
    logging.info("✓ Evaluation completed successfully")
    logging.info("Evaluation results are ready")

logging.info("\n" + "="*60)
logging.info("✓ Graph Probing Analysis Complete!")
logging.info("="*60)
logging.info("Summary:")
logging.info(f"  - Dataset: {dataset_name}")
logging.info(f"  - Model: {model_name}")
logging.info(f"  - Layer analyzed: {layer_id}")
logging.info(f"  - Probe input type: {probe_input}")
logging.info("All steps completed successfully!")
