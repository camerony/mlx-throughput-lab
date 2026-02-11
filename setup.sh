#!/bin/bash
set -euo pipefail

echo "=== MLX Throughput Lab Setup ==="
echo ""

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Warning: MLX requires Apple Silicon (macOS). This script detected: $(uname)"
    echo "MLX will not work on this platform, but you can still install dependencies."
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "Detected Python version: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -lt 3 ]] || { [[ "$PYTHON_MAJOR" -eq 3 ]] && [[ "$PYTHON_MINOR" -lt 11 ]]; }; then
    echo "Error: Python 3.11+ is required. Found Python $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment
if [[ ! -d .venv ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"
echo ""

# Check if dialog is installed
if command -v dialog >/dev/null 2>&1; then
    echo "✓ dialog is installed"
else
    echo "⚠ dialog is not installed (needed for interactive launcher)"
    echo "  Install with: brew install dialog"
fi

# Check if nginx is installed
if command -v nginx >/dev/null 2>&1; then
    echo "✓ nginx is installed"
else
    echo "⚠ nginx is not installed (needed for round-robin tests)"
    echo "  Install with: brew install nginx"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the interactive launcher:"
echo "  ./run_mlx_tests.py"
echo ""
echo "To run a test directly:"
echo "  MLX_MODEL_PATH='mlx-community/Mistral-7B-Instruct-v0.3-4bit' \\"
echo "    python -m unittest tests/test_mlx_server_single.py"
echo ""
