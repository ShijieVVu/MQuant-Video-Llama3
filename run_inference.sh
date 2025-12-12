#!/bin/bash
# Wrapper script to run Qwen2VL inference - tries multiple methods to find conda

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Method 1: Try to initialize conda from common locations
CONDA_INIT=""
for init_path in \
    "/opt/conda/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/conda/etc/profile.d/conda.sh" \
    "/usr/local/conda/etc/profile.d/conda.sh"
do
    if [ -f "$init_path" ]; then
        CONDA_INIT="$init_path"
        echo "Found conda init at: $CONDA_INIT"
        source "$init_path"
        conda activate qwen2vl 2>/dev/null && break || true
    fi
done

# Method 2: Try to find conda python directly in common locations
PYTHON_BIN=""
if [ -z "$PYTHON_BIN" ]; then
    for path in \
        "/opt/conda/envs/qwen2vl/bin/python" \
        "$HOME/anaconda3/envs/qwen2vl/bin/python" \
        "$HOME/miniconda3/envs/qwen2vl/bin/python" \
        "$HOME/conda/envs/qwen2vl/bin/python" \
        "/usr/local/conda/envs/qwen2vl/bin/python"
    do
        if [ -f "$path" ]; then
            PYTHON_BIN="$path"
            echo "Found Python at: $PYTHON_BIN"
            break
        fi
    done
fi

# Method 3: Check if conda is already in PATH
if [ -z "$PYTHON_BIN" ] && command -v conda &> /dev/null; then
    echo "Conda found in PATH, trying to activate..."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate qwen2vl 2>/dev/null && PYTHON_BIN="$(which python)" || true
fi

# Method 4: Use system python3 as fallback
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
    echo "Using system Python: $PYTHON_BIN"
    echo "Warning: Make sure required packages are installed!"
    echo "You may need to install packages with: pip install datasets loguru pandas"
fi

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$SCRIPT_DIR"

# Run the script with all arguments
"$PYTHON_BIN" exam/quant_qwen2vl.py "$@"

