#!/usr/bin/env bash
# ==============================================================================
# KernelBench RL Environment Setup Script
# ==============================================================================
#
# This script sets up the environment for KernelBench RL training.
# Run from /workspace/kernel_dev/kernel-rl
#
# Prerequisites:
#   - RunPod container with pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
#   - CUDA 12.4 and cuDNN 9 available
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
#   cd /workspace/kernel_dev/kernel-rl
#   ./scripts/setup_env.sh
#
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="/workspace/kernel_dev"

log_info "Setting up KernelBench RL environment..."
log_info "Project root: ${PROJECT_ROOT}"
log_info "Workspace root: ${WORKSPACE_ROOT}"

# ==============================================================================
# 1. Clone KernelBench if missing
# ==============================================================================
if [ ! -d "${WORKSPACE_ROOT}/KernelBench" ]; then
    log_info "Cloning KernelBench..."
    cd "${WORKSPACE_ROOT}"
    git clone https://github.com/ScalingIntelligence/KernelBench.git
else
    log_info "KernelBench already exists, skipping clone"
fi

# ==============================================================================
# 2. Clone tinker-cookbook if missing
# ==============================================================================
if [ ! -d "${WORKSPACE_ROOT}/tinker-cookbook" ]; then
    log_info "Cloning tinker-cookbook..."
    cd "${WORKSPACE_ROOT}"
    git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
else
    log_info "tinker-cookbook already exists, skipping clone"
fi

# ==============================================================================
# 3. Create/update virtual environment with uv
# ==============================================================================
cd "${PROJECT_ROOT}"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    log_warn "uv not found in PATH. Attempting to source shell..."
    # Try common shell rc files
    for rc in ~/.bashrc ~/.zshrc ~/.profile; do
        if [ -f "$rc" ]; then
            source "$rc" 2>/dev/null || true
        fi
    done

    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed. Please install it first:"
        log_error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        log_error "  source ~/.bashrc  # or restart your shell"
        exit 1
    fi
fi

log_info "Using uv: $(which uv)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    log_info "Creating virtual environment..."
    uv venv
fi

# ==============================================================================
# 4. Install dependencies
# ==============================================================================
log_info "Installing dependencies with uv..."

# Install the main project in editable mode
uv pip install -e .

# Install KernelBench in editable mode
log_info "Installing KernelBench..."
uv pip install -e "${WORKSPACE_ROOT}/KernelBench"

# Install tinker-cookbook in editable mode
log_info "Installing tinker-cookbook..."
uv pip install -e "${WORKSPACE_ROOT}/tinker-cookbook"

# Install additional KernelBench dependencies
log_info "Installing additional dependencies..."
uv pip install pydra-config tomli litellm

# ==============================================================================
# 5. Set environment variables
# ==============================================================================
log_info "Setting up environment variables..."

# Create .env file if it doesn't exist
ENV_FILE="${PROJECT_ROOT}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    cat > "${ENV_FILE}" << EOF
# KernelBench RL Environment Variables
# =====================================

# Path to KernelBench repository
KERNELBENCH_ROOT=${WORKSPACE_ROOT}/KernelBench

# Tinker API key (get from https://console.tinker.thinkingmachines.ai)
# TINKER_API_KEY=your-api-key-here

# CUDA settings
CUDA_VISIBLE_DEVICES=0
EOF
    log_info "Created .env file at ${ENV_FILE}"
    log_warn "Please set your TINKER_API_KEY in ${ENV_FILE}"
else
    log_info ".env file already exists"
fi

# ==============================================================================
# 6. Verify installation
# ==============================================================================
log_info "Verifying installation..."

# Activate venv for verification
source .venv/bin/activate

# Check imports
python -c "
import torch
import tinker
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset
print('Tinker and tinker-cookbook imported successfully')

# Check CUDA
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available')
"

# Check KernelBench
python -c "
import sys
sys.path.insert(0, '${WORKSPACE_ROOT}/KernelBench')
from src.eval import eval_kernel_against_ref
print('KernelBench imported successfully')
"

# ==============================================================================
# Done
# ==============================================================================
log_info "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set your TINKER_API_KEY in ${ENV_FILE}"
echo "  2. Activate the environment: source .venv/bin/activate"
echo "  3. Source environment: source ${ENV_FILE}"
echo "  4. Start training:"
echo "     python -m kernel_rl.scripts.train_kernel_rl \\"
echo "         model_name=Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "         log_path=./runs/experiment_v1 \\"
echo "         level=1"
echo ""
