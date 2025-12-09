#!/usr/bin/env bash
# ==============================================================================
# KernelBench RL Environment Setup Script
# ==============================================================================
#
# This script sets up the environment for KernelBench RL training.
#
# Prerequisites:
#   - CUDA available (for GPU training)
#   - uv installed (https://docs.astral.sh/uv/)
#
# Usage:
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

log_info "Setting up KernelBench RL environment..."
log_info "Project root: ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}"

# ==============================================================================
# 1. Check uv is available
# ==============================================================================
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

# ==============================================================================
# 2. Sync dependencies (includes KernelBench and tinker-cookbook from git)
# ==============================================================================
log_info "Syncing dependencies with uv..."
uv sync

log_info "Dependencies installed (including KernelBench and tinker-cookbook from git)"

# ==============================================================================
# 3. Set environment variables
# ==============================================================================
log_info "Setting up environment variables..."

# Create .env file if it doesn't exist
ENV_FILE="${PROJECT_ROOT}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    cat > "${ENV_FILE}" << EOF
# KernelBench RL Environment Variables
# =====================================

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
# 4. Verify installation
# ==============================================================================
log_info "Verifying installation..."

# Check imports using uv run
uv run python -c "
import torch
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset
print('Tinker and tinker-cookbook imported successfully')

# Check CUDA
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available')
"

# Check KernelBench
uv run python -c "
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
echo "  2. Start training:"
echo "     uv run python -m kernel_rl.scripts.train_kernel_rl \\"
echo "         model_name=Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "         log_path=./runs/experiment_v1 \\"
echo "         level=1"
echo ""
