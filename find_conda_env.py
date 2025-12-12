#!/usr/bin/env python3
"""Helper script to find conda qwen2vl environment"""
import os
import subprocess
import sys

def find_conda_python():
    """Find the Python executable in qwen2vl conda environment"""
    search_paths = [
        "/opt/conda/envs/qwen2vl/bin/python",
        os.path.expanduser("~/anaconda3/envs/qwen2vl/bin/python"),
        os.path.expanduser("~/miniconda3/envs/qwen2vl/bin/python"),
        os.path.expanduser("~/conda/envs/qwen2vl/bin/python"),
        "/usr/local/conda/envs/qwen2vl/bin/python",
    ]
    
    # Also search in common conda base locations
    conda_bases = [
        "/opt/conda",
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/conda"),
        "/usr/local/conda",
    ]
    
    for base in conda_bases:
        env_path = os.path.join(base, "envs", "qwen2vl", "bin", "python")
        if env_path not in search_paths:
            search_paths.append(env_path)
    
    for path in search_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    # Try to use conda command if available
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'qwen2vl' in line and not line.strip().startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = parts[-1]
                        python_path = os.path.join(env_path, "bin", "python")
                        if os.path.exists(python_path):
                            return python_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None

if __name__ == "__main__":
    python_path = find_conda_python()
    if python_path:
        print(python_path)
        sys.exit(0)
    else:
        print("ERROR: Could not find qwen2vl conda environment", file=sys.stderr)
        print("\nPlease provide the full path to the Python executable in your qwen2vl environment.", file=sys.stderr)
        print("Example: /path/to/conda/envs/qwen2vl/bin/python", file=sys.stderr)
        sys.exit(1)

