#!/bin/bash
# Run Qwen2VL inference with local OCRBench_debug.tsv file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Try to find conda python
PYTHON_BIN=""
if [ -f "$SCRIPT_DIR/find_conda_env.py" ]; then
    PYTHON_BIN=$(python3 "$SCRIPT_DIR/find_conda_env.py" 2>/dev/null || echo "")
fi

# If not found, try common locations
if [ -z "$PYTHON_BIN" ]; then
    for path in \
        "/opt/conda/envs/qwen2vl/bin/python" \
        "$HOME/anaconda3/envs/qwen2vl/bin/python" \
        "$HOME/miniconda3/envs/qwen2vl/bin/python"
    do
        if [ -f "$path" ]; then
            PYTHON_BIN="$path"
            break
        fi
    done
fi

# Fallback to system python
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
    echo "Warning: Using system Python. Make sure packages are installed!" >&2
fi

echo "Using Python: $PYTHON_BIN" >&2

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="$SCRIPT_DIR"

# Default arguments for inference with local dataset
LOCAL_FILE="${1:-/workspace/OCRBench_debug.tsv}"

# Run inference
"$PYTHON_BIN" exam/quant_qwen2vl.py \
    --model_name Qwen2-VL-7B-Instruct \
    --dataset_name OCRBench \
    --local_dataset_file "$LOCAL_FILE" \
    --nsamples 2 \
    --calib_num 2 \
    --eval_num 2 \
    --verbose \
    "${@:2}"

