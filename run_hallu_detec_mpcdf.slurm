#!/bin/bash -l
#
#SBATCH --job-name=hallu_detection
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
DATASET_NAME="truthfulqa"        # Dataset: truthfulqa, halueval, medhallu, helm
MODEL_NAME="gpt2"                 # Model: gpt2, gpt2-medium, gpt2-large, pythia-160m, etc.
CKPT_STEP=-1                      # Checkpoint step (-1 for main checkpoint)
BATCH_SIZE=16                     # Batch size for LLM inference
LAYER_ID=5                        # LLM layer to analyze
PROBE_INPUT="corr"                # Probe input type: corr or activation
NETWORK_DENSITY=0.05              # Network density (0.01 to 1.0)
EVAL_BATCH_SIZE=32                # Batch size for evaluation
NUM_CHANNELS=32                   # Hidden channels in GNN
NUM_LAYERS=3                      # Number of GNN layers
LEARNING_RATE=0.001               # Learning rate
FROM_SPARSE_DATA=true             # Use sparse data representation
NUM_EPOCHS=200                     # Number of training epochs
GPU_ID=0                          # GPU ID to use

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

# Run hallucination detection analysis
${PYTHON} ${PROJECT_DIR}/my_analysis/hallucination_detection_analysis.py \
  --main_dir ${MAIN_DIR} \
  --project_dir ${PROJECT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --model_name ${MODEL_NAME} \
  --ckpt_step ${CKPT_STEP} \
  --batch_size ${BATCH_SIZE} \
  --layer_id ${LAYER_ID} \
  --probe_input ${PROBE_INPUT} \
  --network_density ${NETWORK_DENSITY} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --num_channels ${NUM_CHANNELS} \
  --num_layers ${NUM_LAYERS} \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --gpu_id ${GPU_ID} \
  $([ "${FROM_SPARSE_DATA}" = "true" ] && echo "--from_sparse_data" || echo "")

echo ""
echo "============================================================"
echo "Hallucination detection analysis completed successfully!"
echo "============================================================"
echo "Dataset: ${DATASET_NAME}"
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER_ID}"
echo "Probe input: ${PROBE_INPUT}"
echo "Network density: ${NETWORK_DENSITY}"
echo "Results saved to: ${MAIN_DIR}/reports/hallucination_analysis/"
echo "============================================================"

