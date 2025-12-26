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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

set -euo pipefail

# =============================================================================
# Configuration Parameters (modify as needed)
# =============================================================================

# Dataset and Model Configuration
DATASET_NAME="truthfulqa"         # Dataset: truthfulqa, halueval, medhallu, helm
MODEL_NAME="gpt2"                 # Model: gpt2, gpt2-medium, gpt2-large, pythia-160m, etc.
CKPT_STEP=-1                      # Checkpoint step (-1 for main checkpoint)
LAYER_ID=5                        # LLM layer to analyze

# Network and Probe Configuration
DENSITY=0.05                      # Network density (0.01 to 1.0)
NUM_LAYERS=3                      # Number of GNN layers
HIDDEN_CHANNELS=32                # Hidden channels in GNN
GPU_ID=0                          # GPU ID to use
FROM_SPARSE_DATA="--from_sparse_data"  # Use sparse data (remove to disable)

# Disease Pattern Configuration
DISEASE_PATTERN="epilepsy_like"   # Disease preset: epilepsy_like, dementia_like, autism_like
NUM_CLUSTERS=8                    # Number of functional modules
WITHIN_SCALE=1.3                  # Scale factor for within-module correlations (>1 increases segregation)
BETWEEN_SCALE=0.5                 # Scale factor for between-module correlations (<1 decreases aggregation)
REWIRING_PROB=0.15                # Edge rewiring probability to reduce small-worldness
DISTANCE_THRESHOLD=50             # Index distance threshold for aggregation decrease

# =============================================================================
# Paths (usually don't need to change these)
# =============================================================================
PROJECT_DIR=/u/aomidvarnia/GIT_repositories/llm-graph-probing
MAIN_DIR=/ptmp/aomidvarnia/analysis_results/llm_graph
ENV_PATH=/ptmp/aomidvarnia/uv_envs/llm_graph
PYTHON=${ENV_PATH}/bin/python

# Create log directory
mkdir -p ${MAIN_DIR}/slurm_logs

# Activate env
source "${ENV_PATH}/bin/activate"

# Set environment variables
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MAIN_DIR="${MAIN_DIR}"

# Run neuropathology analysis
${PYTHON} ${PROJECT_DIR}/my_analysis/neuropathology_analysis.py \
  --main_dir ${MAIN_DIR} \
  --project_dir ${PROJECT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --llm_model_name ${MODEL_NAME} \
  --ckpt_step ${CKPT_STEP} \
  --llm_layer ${LAYER_ID} \
  --density ${DENSITY} \
  --num_layers ${NUM_LAYERS} \
  --hidden_channels ${HIDDEN_CHANNELS} \
  --gpu_id ${GPU_ID} \
  --disease_pattern ${DISEASE_PATTERN} \
  --num_clusters ${NUM_CLUSTERS} \
  --within_scale ${WITHIN_SCALE} \
  --between_scale ${BETWEEN_SCALE} \
  --rewiring_prob ${REWIRING_PROB} \
  --distance_threshold ${DISTANCE_THRESHOLD} \
  ${FROM_SPARSE_DATA}

echo ""
echo "============================================================"
echo "Neuropathology analysis completed successfully!"
echo "============================================================"
echo "Dataset: ${DATASET_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER_ID}"
echo "Disease pattern: ${DISEASE_PATTERN}"
echo "Network density: ${DENSITY}"
echo "Results saved to: ${MAIN_DIR}/reports/neuropathology_analysis/${DISEASE_PATTERN}/"
echo "============================================================"



