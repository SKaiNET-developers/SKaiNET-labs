# Minimal image that provides `iree-compile`.
#
# There is no host iree-compile on macOS, so IREE compilation runs in Docker. This
# pulls the public IREE compiler from PyPI (iree-base-compiler), which ships the
# `iree-compile` binary and can cross-compile StableHLO to any --iree-llvmcpu-target-triple
# (e.g. aarch64-unknown-linux-gnu for the SL2619 board).
#
# Build:
#   docker build -t skainet-iree-compile:latest -f docker/iree-compile.Dockerfile .
# Use (via the wrapper script):
#   IREE_DOCKER_IMAGE=skainet-iree-compile:latest \
#     ./scripts/compile-iree-docker.sh build/stablehlo/tinyllama_step.mlir build/iree/tinyllama_step.vmfb
#
# Pin a specific compiler with: --build-arg IREE_VERSION=<x.y.z>
FROM python:3.11-slim

# Pinned to match the SL2619 board's iree-run-module bytecode version (16.0).
# Determined empirically:
#   3.7.x / 3.9.x -> bytecode 15.0  (too old: "bytecode version mismatch; runtime supports 16.0")
#   3.11.x        -> bytecode 16.0 but requires feature [Ch] the board lacks ([EXT_F32|EXT_F64])
#   3.10.0        -> bytecode 16.0, no [Ch]  => runs on the board.
# Override with --build-arg IREE_VERSION=<x.y.z> if the board runtime changes.
ARG IREE_VERSION=3.10.0
RUN pip install --no-cache-dir "iree-base-compiler==${IREE_VERSION}"

# compile-iree-docker.sh overrides the command with `iree-compile ...`; keep the
# default entrypoint empty so that command runs directly.
ENTRYPOINT []
CMD ["iree-compile", "--help"]
