#!/bin/bash -l
#
#SBATCH --job-name=neuropath_analysis
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
export HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

set -euo pipefail

# =============================================================================
# Paths (usually don't need to change these)
# =============================================================================
CONFIG_FILE=/u/aomidvarnia/GIT_repositories/llm-graph-probing/config_files/pipeline_config_qwen.yaml
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
# Configuration Parameters (modify as needed)
# =============================================================================
# Configuration file path (main input to the pipeline)
# Already set above as CONFIG_FILE

# Primary layer for analysis (from config)
PRIMARY_LAYER=5  # Will be extracted from config by the Python script

# Record start time for duration calculation
JOB_START_TIME=$(date +%s)

# Log all input parameters
echo ""
echo "============================================================"
echo "NEUROPATHOLOGY ANALYSIS - JOB PARAMETERS"
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
echo "GPU(s) Allocated: ${SLURM_JOB_GPUS}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK}"
echo "Memory: $(sinfo -N -n $(hostname) -o '%m')"

# Log file and folder naming conventions for neuropathology
echo "FILE & FOLDER NAMING CONVENTIONS (NEUROPATHOLOGY ANALYSIS):"
echo "=============================================================================="
echo ""
echo "1. MODEL NAME SANITIZATION:"
echo "   - Input:  'Qwen/Qwen2.5-0.5B'"
echo "   - Output: 'Qwen_Qwen2_5_0_5B'"
echo "   - Rule: Replace '/' and '-' with '_'"
echo ""
echo "2. DIRECTORY STRUCTURE:"
echo "   \${MAIN_DIR}/reports/neuropathology_analysis/{model_tag}/"
echo "   └── {disease_pattern}_d{density}/"
echo "       └── layer_{layer_id}/"
echo "           ├── N1_construct_dataset.log"
echo "           ├── N2_compute_network.log"
echo "           ├── N3_train_probe.log (healthy)"
echo "           ├── N4_neuropathology_connectivity.log"
echo "           ├── N5_neuropathology_eval.log"
echo "           ├── N6_neuropathology_graph_metrics.log"
echo "           ├── summary.json"
echo "           ├── fc_healthy_layer_{layer_id}.npy"
echo "           ├── fc_patho_{disease_pattern}_layer_{layer_id}.npy"
echo "           ├── probe_performance_healthy.json"
echo "           ├── probe_performance_patho.json"
echo "           ├── metrics_summary_layer_{layer_id}.json"
echo "           └── tensorboard/"
echo ""
echo "3. DISEASE PATTERN NAMING:"
echo "   - Format: {disease_pattern}_d{density_without_dot}"
echo "   - Examples:"
echo "     * epilepsy_like_d01     (density=0.1)"
echo "     * parkinson_like_d05    (density=0.5)"
echo "     * epilepsy_like_d10     (density=1.0)"
echo ""
echo "4. PATHOLOGICAL FC FILENAMES:"
echo "   - Format: fc_patho_{disease_pattern}_layer_{layer_id}.npy"
echo "   - Example: fc_patho_epilepsy_like_layer_5.npy"
echo ""
echo "5. PERFORMANCE METRICS:"
echo "   - Healthy probe on healthy FC: probe_performance_healthy.json"
echo "   - Healthy probe on pathological FC: probe_performance_patho.json"
echo ""
echo "6. GRAPH METRICS:"
echo "   - Filename: metrics_summary_layer_{layer_id}.json"
echo "   - Contains: clustering_coeff, degree_dist, path_length, etc."
echo "   - For both healthy and pathological networks"
echo ""
echo "EXAMPLE OUTPUT PATHS:"
echo "/ptmp/aomidvarnia/analysis_results/llm_graph/reports/neuropathology_analysis/"
echo "  Qwen_Qwen2_5_0_5B/"
echo "  epilepsy_like_d01/"
echo "  layer_5/"
echo "  fc_healthy_layer_5.npy"
echo "  fc_patho_epilepsy_like_layer_5.npy"
echo "  metrics_summary_layer_5.json"
echo ""
echo "=============================================================================="
echo ""

echo ""
echo "============================================================"
echo ""

echo "Running neuropathology analysis..."
${PYTHON} ${PROJECT_DIR}/my_analysis/neuropathology_analysis.py \
  --config ${CONFIG_FILE}



