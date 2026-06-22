# Synaptics Astra / SL2619 Run Notes

Target: Linux ARM64 (`aarch64`).

Observed over ADB at `192.168.3.26:5555` on 2026-06-22:

- device tree model: `Synaptics SL2619 RDK`
- `Linux sl2619 6.12.62 ... aarch64 GNU/Linux`
- Yocto/Poky `5.0.9 (scarthgap)`
- ADB shell runs as `root`
- rootfs is mounted read-only from `/dev/mmcblk0p12`
- rootfs is 2.3 GB with only about 54 MB free
- `/tmp` is writable tmpfs, about 958 MB with about 953 MB free
- `iree-run-module` is present; `iree-compile` is not present
- `iree-benchmark-module` is not present
- IREE HAL devices: `local-sync://`, `local-task://`, `torq://`
- IREE HAL drivers: `local-sync`, `local-task`, `torq`
- Torq runtime defaults include `--torq_hw_type=astra_machina` and
  `--torq_device_allocator=dmabuf`
- `/dev/torq` exists as char device `10:259`
- `/sys/class/misc/torq` points to `f7600000.synpu`
- kernel module `syna_npu` is loaded
- `/proc/interrupts` exposes `torq-npu-irq`
- DMA-BUF heaps exist under `/dev/dma_heap`, including `linux,cma`,
  `CMA-CUST-linux,cma`, and uncached/system heaps
- the image has `/usr/lib/libcrypt.so.2`, while Kotlin/Native currently needs `libcrypt.so.1`
- `readelf` is present; `ldd` is not

## CPU and Memory

CPU:

- 2 online/present CPUs: `0-1`
- Arm implementer `0x41`, part `0xd05` (Cortex-A55 class)
- features include `fp`, `asimd`, `fphp`, `asimdhp`, `asimdrdm`,
  `asimddp`, `aes`, `sha1`, `sha2`, `crc32`, and `atomics`
- CPU frequency policy exposes `1700000` and `2000000` kHz
- current governor is `performance`

Memory:

- total memory: `1962976 kB`
- observed available memory: about `1479 MB`
- no swap
- CMA total: `524288 kB`
- observed CMA free: about `350860 kB`

Implications:

- CPU fallback is a 2-core Cortex-A55 target; start CPU IREE runs with
  `local-task://` and at most two workers.
- Torq/NPU is a real runtime target on this image, but it requires a VMFB
  compiled with the matching Synaptics/Torq compiler backend.
- Do not stage model files on `/`; use `/tmp`, external media, or an explicit
  writable mount.
- Prefer IREE `--parameter_mode=file` and `--module_mode=mmap` for larger
  artifacts when compatible with the module.

Collect a fresh profile with:

```bash
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-board-capabilities.sh
```

## One-Time Setup

Install:

- JDK 21+
- Gradle 9+
- `curl`
- optional: IREE tools (`iree-compile`, `iree-run-module`)
- optional Python baseline: Python 3.10+ and build tools for `llama-cpp-python`

Download models:

```bash
./scripts/download-models.sh
```

## ADB Smoke Run

From the host:

```bash
./gradlew linkReleaseExecutableLinuxArm64
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-board-run.sh help
```

The script pushes the `linuxArm64` executable to `/tmp/skainet-tinyllama`,
building it first only if it is missing. It also creates a local
`/tmp/skainet-tinyllama/libcrypt.so.1` symlink to the board's
`/usr/lib/libcrypt.so.2`. The symlink is intentionally kept under `/tmp` so it
does not mutate the root filesystem.

## Kotlin Native Benchmark

```bash
./gradlew linkReleaseExecutableLinuxArm64
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-board-run.sh eager \
  --model /tmp/skainet-tinyllama/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --tokens 32 \
  --prompt "What is quantization in machine learning?"
```

Repeat with `--model Q8_0`.

The native SKaiNET path streams GGUF through packed/custom tensor storage rather
than expanding the whole model into a dense FP32 image. Runtime memory still
depends on KV cache size, prompt length, backend workspaces, and the exact
GGUF tensor mix, so keep `--ctx` and `--tokens` small for first board runs.

## JVM Benchmark

```bash
./gradlew runJvm --args='eager --model Q4_K_M --tokens 32 --prompt "What is quantization in machine learning?"'
```

## IREE Compile

```bash
./gradlew exportStableHlo
IREE_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
IREE_TARGET_CPU=generic \
./scripts/compile-iree.sh build/stablehlo/tinyllama_step.mlir build/iree/tinyllama_step.vmfb
```

The default script uses a host `iree-compile` binary and produces an `llvm-cpu`
VMFB, suitable for `local-task://`. Tune `IREE_TARGET_CPU` after checking the
exact CPU core exposed by the SL2619 image.

On the Linux host with Dockerized IREE tooling:

```bash
./gradlew exportStableHlo
IREE_DOCKER_IMAGE=<image-with-iree-compile> \
IREE_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
IREE_TARGET_CPU=generic \
./scripts/compile-iree-docker.sh \
  build/stablehlo/tinyllama_step.mlir \
  build/iree/tinyllama_step.vmfb
```

The Docker wrapper mounts the repo at `/work`. Additional compiler flags can be
passed after the output path, which is the intended hook for Synaptics/Torq
toolchain options once the Linux host has the vendor image available.

The board also exposes Synaptics Torq through IREE as `torq://`. A Torq/NPU VMFB
needs the matching vendor IREE compiler backend/toolchain; do not assume the
generic `llvm-cpu` artifact will execute on `torq://`.

Push and run a VMFB:

```bash
ADB_SERIAL=192.168.3.26:5555 ./scripts/adb-iree-run.sh \
  build/iree/tinyllama_step.vmfb \
  --device=local-task:// \
  --function=tinyllama_step \
  --print_statistics=true
```

For a Torq-compiled VMFB, switch `--device=torq://`.
