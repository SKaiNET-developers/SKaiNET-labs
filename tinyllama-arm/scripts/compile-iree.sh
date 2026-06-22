#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/compile-iree.sh <input.mlir> <output.vmfb> [extra iree-compile args...]" >&2
  exit 2
fi

input_mlir="$1"
output_vmfb="$2"
shift 2
iree_compile="${IREE_COMPILE:-iree-compile}"
target_cpu="${IREE_TARGET_CPU:-generic}"
target_triple="${IREE_TARGET_TRIPLE:-aarch64-unknown-linux-gnu}"
hal_backends="${IREE_HAL_BACKENDS:-llvm-cpu}"

mkdir -p "$(dirname "$output_vmfb")"

"$iree_compile" "$input_mlir" \
  --iree-hal-target-backends="$hal_backends" \
  --iree-llvmcpu-target-triple="$target_triple" \
  --iree-llvmcpu-target-cpu="$target_cpu" \
  "$@" \
  -o "$output_vmfb"

echo "WROTE_VMFB $output_vmfb"
