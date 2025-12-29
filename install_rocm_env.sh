#!/bin/bash
# Install PyTorch environment with ROCm 6.1 support using uv
# Usage: bash install_rocm_env.sh

set -e  # Exit on error
set -u  # Treat unset variables as errors
set -o pipefail

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv is not installed or not in PATH."
  echo "Install uv: pip install uv  (or pipx install uv)"
  exit 1
fi

# Prepare uv virtual environment path
ENV_PATH="/ptmp/aomidvarnia/uv_envs/llm_graph"
echo ""
echo "=========================================="
echo "Preparing uv environment at: ${ENV_PATH}"
echo "=========================================="

# Remove existing environment if present
if [ -d "${ENV_PATH}" ]; then
  echo "Existing environment found. Removing: ${ENV_PATH}"
  rm -rf "${ENV_PATH}"
fi

# Create a fresh uv virtual environment
echo "Creating new uv environment..."
uv venv "${ENV_PATH}"

# Verify environment creation
if [ ! -x "${ENV_PATH}/bin/python" ]; then
  echo "FATAL: Failed to create uv environment at ${ENV_PATH}"
  exit 1
fi

# Activate the environment
echo "Activating environment: ${ENV_PATH}"
source "${ENV_PATH}/bin/activate"
echo "Using Python: $(python --version) at ${ENV_PATH}"

echo ""
echo "=========================================="
echo "Checking ROCm/HIP device visibility"
echo "=========================================="
if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "SLURM job detected: ${SLURM_JOB_ID}"
else
  echo "Note: No SLURM job detected. If on a login CPU node, GPUs won't be visible."
  echo "      Run interactively with a GPU, e.g.: srun --partition=gpu --gres=gpu:1 --pty bash -i"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  echo "\nrocm-smi (GPU listing):"
  rocm-smi || true
else
  echo "rocm-smi not found in PATH. Try: module load rocm/6.1"
fi

if command -v rocminfo >/dev/null 2>&1; then
  echo "\nrocminfo (hardware topology):"
  rocminfo | sed -n '1,120p' || true
else
  echo "rocminfo not found in PATH. Try: module load rocm/6.1"
fi

if [ -e /dev/kfd ]; then
  echo "/dev/kfd present"
else
  echo "WARNING: /dev/kfd missing (ROCm kernel driver not available to user)"
fi
if ls /dev/dri/render* >/dev/null 2>&1; then
  echo "DRI render nodes present: $(ls /dev/dri/render* | tr '\n' ' ')"
else
  echo "WARNING: No /dev/dri render nodes visible"
fi

echo "=========================================="
echo "Installing PyTorch + ROCm 6.1 environment"
echo "Using: uv pip install"
echo "=========================================="

# Step 1: Install all non-scatter dependencies
echo ""
echo "Step 1: Installing base dependencies..."
uv pip install \
  absl-py==2.1.0 \
  accelerate==1.6.0 \
  datasets==3.4.1 \
  evaluate==0.4.3 \
  gensim==4.4.0 \
  numpy==1.26.4 \
  pillow==11.2.1 \
  scikit-learn==1.5.1 \
  scipy==1.16.2 \
  setproctitle==1.3.5 \
  tensorboard==2.19.0 \
  tqdm==4.66.4 \
  transformers==4.50.2\
  matplotlib==3.8.1 \
  pyyaml \
  ptpython \

# Step 2: Install PyTorch with ROCm 6.1 (from official index)
echo ""
echo "Step 2: Installing PyTorch 2.5.1 + ROCm 6.1..."
uv pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.5.1+rocm6.1 \
  torchvision==0.20.1+rocm6.1 \
  torchaudio==2.5.1+rocm6.1

# Step 3: Install PyG and other libs
echo ""
echo "Step 3: Installing PyTorch Geometric and other libraries..."
uv pip install torch-geometric==2.6.1

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo "Environment ready for hallucination detection analysis"

echo ""
echo "=========================================="
echo "Validating ROCm installation in activated venv"
echo "=========================================="
python - <<'PY'
import importlib
import torch
print('PyTorch version:', torch.__version__)
print('HIP version:', getattr(torch.version, 'hip', 'N/A'))
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
  for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:', torch.cuda.get_device_name(i))
    props = torch.cuda.get_device_properties(i)
    try:
      arch = getattr(props, 'gcnArchName')
    except Exception:
      arch = 'N/A'
    print(f'    Memory: {props.total_memory/1024**3:.1f} GB | Compute units: {props.multi_processor_count} | Arch: {arch}')

def show_version(pkg):
  try:
    m = importlib.import_module(pkg)
    ver = getattr(m, '__version__', 'unknown')
    print(f'{pkg}:', ver)
  except Exception as e:
    print(f'{pkg}: NOT INSTALLED ({e})')

print('\nKey packages:')
for name in ['torch', 'torch_geometric']:
  show_version(name)
if not torch.cuda.is_available():
  print('\nNo GPUs visible to PyTorch. Common causes:')
  print('  - Not running on a GPU node (login CPU node)')
  print('  - Missing ROCm module: module load rocm/6.1')
  print('  - SLURM not allocating GPUs: use --gres=gpu:N')
  print('  - Visibility env vars set by scheduler (HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES)')
  print('\nTry: srun --partition=gpu --gres=gpu:1 --pty bash -i')
PY
