#!/bin/bash
# Setup script for MQuant conda environment
# Run this script to complete the conda environment setup

set -e

# Add conda to PATH if not already there
export PATH="/opt/conda/bin:$PATH"

# Navigate to the MQuant directory
cd /workspace/MQuant

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo "Activating environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen2vl

echo "Installing fast-hadamard-transform..."
cd /tmp
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
cd /workspace/MQuant

echo "Installing VLMEvalKit requirements..."
cd third/VLMEvalKit
pip install -r requirements.txt
cd /workspace/MQuant

echo "Setup complete! To activate the environment, run:"
echo "  conda activate qwen2vl"

