#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/compile-iree-docker.sh <input.mlir> <output.vmfb> [extra iree-compile args...]

Required:
  IREE_DOCKER_IMAGE=<image-with-iree-compile>

Common environment:
  IREE_HAL_BACKENDS     default: llvm-cpu
  IREE_TARGET_TRIPLE    default: aarch64-unknown-linux-gnu
  IREE_TARGET_CPU       default: generic

Example:
  IREE_DOCKER_IMAGE=ghcr.io/your-org/iree-tooling:tag \
    scripts/compile-iree-docker.sh \
    build/stablehlo/tinyllama_step.mlir \
    build/iree/tinyllama_step.vmfb
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
input_mlir="$1"
output_vmfb="$2"
shift 2

image="${IREE_DOCKER_IMAGE:-}"
target_cpu="${IREE_TARGET_CPU:-generic}"
target_triple="${IREE_TARGET_TRIPLE:-aarch64-unknown-linux-gnu}"
hal_backends="${IREE_HAL_BACKENDS:-llvm-cpu}"

if [[ -z "$image" ]]; then
  echo "IREE_DOCKER_IMAGE is required; point it at the Linux-host IREE tooling image." >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for scripts/compile-iree-docker.sh." >&2
  exit 1
fi

if [[ ! -f "$input_mlir" ]]; then
  echo "Input MLIR not found: $input_mlir" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_vmfb")"

input_dir="$(cd "$(dirname "$input_mlir")" && pwd -P)"
input_base="$(basename "$input_mlir")"
output_dir="$(cd "$(dirname "$output_vmfb")" && pwd -P)"
output_base="$(basename "$output_vmfb")"

to_container_path() {
  local host_path="$1"
  if [[ "$host_path" == "$root_dir" ]]; then
    printf '/work'
  elif [[ "$host_path" == "$root_dir"/* ]]; then
    printf '/work/%s' "${host_path#"$root_dir"/}"
  else
    echo "Path must be inside the repository mounted at /work: $host_path" >&2
    exit 1
  fi
}

container_input="$(to_container_path "$input_dir/$input_base")"
container_output="$(to_container_path "$output_dir/$output_base")"

docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "$root_dir:/work" \
  -w /work \
  "$image" \
  iree-compile "$container_input" \
    --iree-hal-target-backends="$hal_backends" \
    --iree-llvmcpu-target-triple="$target_triple" \
    --iree-llvmcpu-target-cpu="$target_cpu" \
    "$@" \
    -o "$container_output"

echo "WROTE_VMFB $output_vmfb"
