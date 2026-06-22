#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="$root_dir/models"
mkdir -p "$models_dir"

repo="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main"

for variant in Q4_K_M Q8_0; do
  file="tinyllama-1.1b-chat-v1.0.${variant}.gguf"
  url="$repo/$file"
  out="$models_dir/$file"
  if [[ -f "$out" ]]; then
    echo "Already present: $out"
  else
    echo "Downloading $file"
    curl -L "$url" -o "$out"
  fi
done

