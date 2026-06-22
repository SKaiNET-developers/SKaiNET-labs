# TinyLlama on SKaiNET, StableHLO, and IREE

Standalone SKaiNET/Kotlin port of Arm's TinyLlama edge benchmark:

- SKaiNET eager benchmark for TinyLlama GGUF.
- Linux ARM64 Kotlin/Native executable target for the Synaptics Astra class of boards.
- StableHLO MLIR export for a tiny Llama-shaped NN DSL graph.
- IREE compile wrappers for host-native tooling and Dockerized Linux tooling.
- The original Arm Python benchmark copied under `benchmarks/python/` for baseline comparison.

The Kotlin implementation is pure SKaiNET. The Python code is intentionally isolated as the upstream baseline.

## Model Setup

```bash
./scripts/download-models.sh
```

This downloads:

- `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- `models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf`

## JVM Eager Benchmark

```bash
./gradlew runJvm --args='eager --model Q4_K_M --tokens 64 --prompt "What is quantization?"'
./gradlew runJvm --args='eager --model Q8_0 --tokens 64 --prompt "What is quantization?"'
```

The JVM command prints load time, inference time, tokens/sec, and process RSS.
It intentionally follows the NN DSL eager path and streams GGUF through SKaiNET's packed/custom tensor storage. Quantized tensors such as Q4_K and Q8_0 stay in quantized storage for the backend dispatch path; dense scalar tensors are materialized only where the loader/runtime needs them. Keep the Python baseline for the original Arm comparison, and use the StableHLO/IREE path here to validate the SKaiNET compiler handoff.

## Native ARM64 Build

On the Linux ARM64 board:

```bash
./gradlew linkReleaseExecutableLinuxArm64
./build/bin/linuxArm64/releaseExecutable/tinyllama-skainet.kexe eager --model Q4_K_M --tokens 32
```

The native path mirrors the SKaiNET transformer DSL path: NN DSL eager execution with GGUF loaded through packed/custom tensor storage.

For the SL2619 board exposed over ADB:

```bash
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-board-run.sh help
```

See [docs/BOARD_SL2619.md](docs/BOARD_SL2619.md) for the observed Yocto image details and the local `libcrypt.so.1` compatibility shim used by the helper script.

## StableHLO and IREE

Export a tiny Llama-shaped graph from SKaiNET NN DSL to DAG to StableHLO:

```bash
./gradlew exportStableHlo
```

Compile the MLIR with IREE:

```bash
IREE_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
IREE_TARGET_CPU=generic \
./scripts/compile-iree.sh build/stablehlo/tinyllama_step.mlir build/iree/tinyllama_step.vmfb
```

Or, on the Linux host with Dockerized IREE tooling:

```bash
IREE_DOCKER_IMAGE=<image-with-iree-compile> \
IREE_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
IREE_TARGET_CPU=generic \
./scripts/compile-iree-docker.sh \
  build/stablehlo/tinyllama_step.mlir \
  build/iree/tinyllama_step.vmfb
```

The Docker wrapper mounts this repository at `/work` and writes the VMFB back
into `build/iree/`. Both compile wrappers accept additional `iree-compile`
flags after the output path; use that hook with vendor toolchain images.

`iree-compile` is not vendored here; install it on the host or use the Docker wrapper.
The SL2619 image has `iree-run-module` with `local-task://` and `torq://` devices;
Torq/NPU execution requires a VMFB compiled with the matching Synaptics vendor backend.

Run the compiled VMFB on the board:

```bash
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-iree-run.sh \
  build/iree/tinyllama_step.vmfb \
  --device=local-task:// \
  --function=tinyllama_step \
  --print_statistics=true
```

For a Torq-compiled VMFB, switch the runtime device to `--device=torq://`.

## Python Baseline

```bash
./scripts/python-baseline.sh --model Q4_K_M --tokens 64 --threads 4
./scripts/python-baseline.sh --model Q8_0 --tokens 64 --threads 4
```

## Current Compiler Boundary

SKaiNET-transformers currently marks full transformer StableHLO/IREE execution as still being wired beyond the tested paths. This repo therefore exports a small Llama-shaped graph for compiler-path validation and keeps full TinyLlama GGUF generation on eager SKaiNET. The eager path loads GGUF through SKaiNET's packed/custom tensor storage so quantized weights are not forced into one dense FP32 model image.
