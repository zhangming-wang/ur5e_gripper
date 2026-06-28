#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$PROJECT_DIR/src"

source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

cd "$SRC_DIR" && PYTHONPATH="$SRC_DIR" python3 run_isaaclab.py "$@"
