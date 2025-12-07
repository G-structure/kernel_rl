#!/bin/bash
# Setup script for kernel-rl
# Installs uv (Python package manager) and just (command runner)

set -e

echo "=== kernel-rl Setup ==="

# Install uv
if command -v uv &> /dev/null; then
    echo "✓ uv already installed ($(uv --version))"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed ($(uv --version))"
fi

# Install just
if command -v just &> /dev/null; then
    echo "✓ just already installed ($(just --version))"
else
    echo "Installing just..."
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
    echo "✓ just installed ($(just --version))"
fi

# Install git-lfs
if command -v git-lfs &> /dev/null; then
    echo "✓ git-lfs already installed ($(git-lfs --version))"
else
    echo "Installing git-lfs..."
    apt-get update && apt-get install -y git-lfs
    echo "✓ git-lfs installed ($(git-lfs --version))"
fi
git lfs install

# Sync Python dependencies
echo ""
echo "Syncing Python dependencies..."
uv sync
echo "✓ Dependencies installed"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  just train my_experiment  # Start training"
echo "  just status               # Check running jobs"
echo "  just metrics <run>        # View metrics"
echo "  just --list               # Show all commands"
