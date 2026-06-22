#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir/benchmarks/python"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
python tinyllama_benchmark.py "$@"

