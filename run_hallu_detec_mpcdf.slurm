#!/bin/bash -l
#
#SBATCH --job-name=hallu_gpt2
#SBATCH --output=/ptmp/aomidvarnia/analysis_results/llm_graph/slurm_logs/%j.out
#SBATCH --error=/ptmp/aomidvarnia/analysis_results/llm_graph/slurm_logs/%j.err
#
#SBATCH --ntasks=1
#SBATCH --array=0-0
#
# --- Use single GPU for LLM and GNN training ---
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --mail-user=a.omidvarnia@fz-juelich.de
#SBATCH --mail-type=FAIL,END
#SBATCH --time=24:00:00

module purge
module load gcc/14 openmpi/5.0 rocm/7.0

# Set OMP threads to a valid positive integer
if [[ "${SLURM_CPUS_PER_TASK:-}" =~ ^[0-9]+$ ]] && [ "${SLURM_CPUS_PER_TASK}" -gt 0 ]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
else
    echo "WARNING: SLURM_CPUS_PER_TASK invalid; defaulting OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
fi

# Set GPU visibility for ROCm/HIP - use SLURM_LOCALID for single GPU jobs
if [ -n "${SLURM_LOCALID:-}" ]; then
    export HIP_VISIBLE_DEVICES="${SLURM_LOCALID}"
else
    export HIP_VISIBLE_DEVICES="0"
fi
# Also set for CUDA/ROCm compatibility
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"

set -euo pipefail

# Record start time
JOB_START_TIME=$(date +%s)

# =============================================================================
# Paths (usually don't need to change these)
# =============================================================================
CONFIG_FILE=/u/aomidvarnia/GIT_repositories/llm-graph-probing/config_files/pipeline_config_gpt2.yaml
ENV_PATH=/ptmp/aomidvarnia/uv_envs/llm_graph
PYTHON=${ENV_PATH}/bin/python

# Derive project_dir and main_dir from config (with defaults)
PROJECT_DIR=$(${PYTHON} - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_FILE}", "r"))
common = cfg.get("common", {}) if cfg else {}
print(common.get("project_dir", "/u/aomidvarnia/GIT_repositories/llm-graph-probing"))
PY
)
MAIN_DIR=$(${PYTHON} - <<PY
import yaml
cfg = yaml.safe_load(open("${CONFIG_FILE}", "r"))
common = cfg.get("common", {}) if cfg else {}
print(common.get("main_dir", "/ptmp/aomidvarnia/analysis_results/llm_graph"))
PY
)

# Create log directory
mkdir -p ${MAIN_DIR}/slurm_logs

# Activate env
source "${ENV_PATH}/bin/activate"

# Set environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MAIN_DIR="${MAIN_DIR}"

# =============================================================================
# GPU/ROCm Test
# =============================================================================
echo ""
echo "============================================================"
echo "Testing PyTorch GPU/ROCm Availability"
echo "============================================================"
echo "System GPU Info:"
echo "  Total GPUs on node: $(nvidia-smi -L 2>/dev/null | wc -l || rocm-smi --showid 2>/dev/null | grep -c 'GPU' || echo '2 (AMD MI250X - 2 GCDs per node)')"
echo "  SLURM GPU allocation: ${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-1}} GPU(s) allocated to this job"
echo ""
${PYTHON} -c "
import torch
import os
print('PyTorch Configuration:')
print('  PyTorch version:', torch.__version__)
print('  HIP version:', torch.version.hip if hasattr(torch.version, 'hip') else 'N/A')
print('  CUDA available:', torch.cuda.is_available())
print('  Device count visible to PyTorch:', torch.cuda.device_count())
print('')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute units: {props.multi_processor_count}')
        print(f'    Architecture: {props.gcnArchName if hasattr(props, \"gcnArchName\") else \"N/A\"}')
        print('')
else:
    print('WARNING: No GPU detected by PyTorch!')
    print('Environment variables:')
    print('  HIP_VISIBLE_DEVICES:', os.getenv('HIP_VISIBLE_DEVICES', 'Not set'))
    print('  ROCR_VISIBLE_DEVICES:', os.getenv('ROCR_VISIBLE_DEVICES', 'Not set'))
    print('  CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES', 'Not set'))
print('='*60)
"
echo "============================================================"
echo ""

# =============================================================================
# Configuration Parameters (modify as needed)
# =============================================================================
# Configuration file path (main input to the pipeline)
# Already set above as CONFIG_FILE

# Primary layer for analysis (from config)
PRIMARY_LAYER=5  # Will be extracted from config by the Python script

# =============================================================================
# Log all input parameters
# =============================================================================
echo ""
echo "============================================================"
echo "HALLUCINATION DETECTION ANALYSIS - JOB PARAMETERS"
echo "============================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Job Name: ${SLURM_JOB_NAME}"
echo "Job Submission Time: $(date)"
echo ""
echo "--- Configuration ---"
echo "Config File: ${CONFIG_FILE}"
echo ""
echo "--- Paths and Environment ---"
echo "Project Directory: ${PROJECT_DIR}"
echo "Main Directory: ${MAIN_DIR}"
echo "Python Executable: ${PYTHON}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""
echo "--- GPU Configuration ---"
echo "GPU(s) Requested: ${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-1}}"
echo "GPU Devices (HIP_VISIBLE_DEVICES): ${HIP_VISIBLE_DEVICES}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK}"
echo "Memory: $(sinfo -N -n $(hostname) -o '%m')"
echo "Node: ${SLURM_NODELIST}"

# Log file and folder naming conventions
echo "FILE & FOLDER NAMING CONVENTIONS:"
echo "=============================================================================="
echo ""
echo "1. MODEL NAME SANITIZATION:"
echo "   - Input:  'Qwen/Qwen2.5-0.5B'"
echo "   - Output: 'Qwen_Qwen2_5_0_5B'"
echo "   - Rule: Replace '/' and '-' with '_' to create safe folder names"
echo ""
echo "2. DIRECTORY STRUCTURE:"
echo "   \${MAIN_DIR}/reports/hallucination_analysis/{model_tag}/"
echo "   └── layer_{layer_id}/"
echo "       ├── N1_construct_dataset.log       (Step 1 log)"
echo "       ├── N2_compute_network.log         (Step 2 log)"
echo "       ├── N3_train_probe.log             (Step 3 log)"
echo "       ├── summary.json                   (Step results)"
echo "       ├── fc_healthy_layer_{layer_id}.npy"
echo "       ├── fc_healthy_layer_{layer_id}.png"
echo "       ├── probe_performance.json         (Classification metrics)"
echo "       ├── train_history.json             (Training curves)"
echo "       ├── probe_weights/"
echo "       │   └── best_model.pt"
echo "       └── tensorboard/"
echo ""
echo "3. FILENAME PATTERNS:"
echo "   - Correlation matrices: fc_[condition]_layer_{layer_id}.npy"
echo "     * fc_healthy_layer_5.npy"
echo "   - Visualizations: fc_[condition]_layer_{layer_id}.png"
echo "   - Logs: N{step}_{step_name}.log"
echo "   - Performance: probe_performance.json"
echo "   - Training: train_history.json, train_losses.txt"
echo ""
echo "4. SAVED CHECKPOINTS:"
echo "   \${MAIN_DIR}/saves/{model_tag}/{dataset_name}/{ckpt_step}/layer_{layer_id}/"
echo "   ├── best_model.pt"
echo "   ├── final_model.pt"
echo "   └── optimizer.pt"
echo ""
echo "EXAMPLE OUTPUT PATH:"
echo "/ptmp/aomidvarnia/analysis_results/llm_graph/reports/hallucination_analysis/"
echo "  Qwen_Qwen2_5_0_5B/"
echo "  layer_5/"
echo "  fc_healthy_layer_5.npy"
echo ""
echo "=============================================================================="
echo ""

echo ""
echo "============================================================"
echo ""

echo "Running hallucination detection analysis..."
${PYTHON} ${PROJECT_DIR}/my_analysis/hallucination_detection_analysis.py \
  --config ${CONFIG_FILE}

# Calculate job duration
JOB_END_TIME=$(date +%s)
JOB_DURATION=$((JOB_END_TIME - JOB_START_TIME))
JOB_HOURS=$((JOB_DURATION / 3600))
JOB_MINUTES=$(((JOB_DURATION % 3600) / 60))
JOB_SECONDS=$((JOB_DURATION % 60))

echo ""
echo "============================================================"
echo "Hallucination detection analysis completed successfully!"
echo "============================================================"
echo "Config File: ${CONFIG_FILE}"
echo "Results saved to: ${MAIN_DIR}/reports/hallucination_analysis/"
echo "Job duration: ${JOB_HOURS}h ${JOB_MINUTES}m ${JOB_SECONDS}s"
echo "============================================================"


