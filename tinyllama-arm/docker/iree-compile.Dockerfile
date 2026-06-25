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

# Pinned to the latest IREE tools (3.11). NOTE: the board's iree-run-module MUST be
# updated to match (>= 3.11) — a 3.11-compiled vmfb requires module feature [Ch] that
# the board's *old* 3.10 runtime lacked ([EXT_F32|EXT_F64] only). History (board was
# bytecode 16.0 == 3.10): 3.7/3.9 -> bytecode 15.0 (too old); 3.10 -> 16.0 no [Ch];
# 3.11 -> 16.0 + [Ch]. With the board runtime refreshed to 3.11, [Ch] is satisfied.
# Override with --build-arg IREE_VERSION=<x.y.z>.
ARG IREE_VERSION=3.11.0
RUN pip install --no-cache-dir "iree-base-compiler==${IREE_VERSION}"

# compile-iree-docker.sh overrides the command with `iree-compile ...`; keep the
# default entrypoint empty so that command runs directly.
ENTRYPOINT []
CMD ["iree-compile", "--help"]
