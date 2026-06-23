#!/usr/bin/env bash
# Run the upstream Arm llama.cpp Python baseline via uv (no manual venv).
# Pass-through args go to tinyllama_benchmark.py, e.g.:
#   scripts/python-baseline.sh --model /abs/path.gguf --tokens 64 --ctx 512 --prompt "..."
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir/benchmarks/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/). Install it or run the benchmark manually." >&2
  exit 1
fi

# uv builds an ephemeral env from requirements.txt (llama-cpp-python compiles on first run).
exec uv run --with-requirements requirements.txt python tinyllama_benchmark.py "$@"
