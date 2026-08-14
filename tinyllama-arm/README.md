# TinyLlama on a 2 GB Cortex-A55, in pure Kotlin

TinyLlama-1.1B (GGUF, Q4_K_M) running on a Synaptics Astra SL2619 — **two in-order Cortex-A55
cores, 1.92 GB of RAM, no swap** — through a from-scratch Kotlin stack: Kotlin Multiplatform,
[SKaiNET](https://github.com/SKaiNET-developers/SKaiNET), hand-written Neon kernels, and an
independent StableHLO→IREE compiled path that cross-checks it.

We benchmarked our Q4_K kernel against **Arm's own KleidiAI micro-kernels on the same board and
lost by 1.95×** — then made that number the published target instead of vendoring the library,
because our format keeps quantization fidelity KleidiAI's discards. Everything here is measured,
tagged, and reproducible on Arm64 hardware you already own.

Licensed MIT — see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## The problem

[Daily-StandAPP](https://github.com/michalharakal/Daily-StandAPP) summarizes your commit history
into a daily standup — locally, because commit history is exactly what you don't send to someone
else's API. Its latency budget is **8 seconds median**; its embedded backend is SKaiNET, already
the fastest local option it measures. At the starting speed of 0.17 tok/s, a 200-token summary
took **twenty minutes**. Push the same workload onto a device you could leave running in a corner
— the SL2619 — and the constraint stops being latency: the first board runs were **OOM-killed
during model load**, at ~3.2 GB peak against 1.92 GB of RAM.

One metric, defined once, never redefined: **tok/s = generated tokens / inference seconds**, load
time excluded and stated separately, greedy decoding, fixed prompt. Every number below has an
annotated git tag, a logbook entry with its mechanism and reproduction command
([`docs/PERF-LOGBOOK.md`](docs/PERF-LOGBOOK.md)), and a row in
[`docs/perf-history.csv`](docs/perf-history.csv).

## What was optimized

**HOST and BOARD are different classes of machine and are never compared to each other** —
llama.cpp itself runs 35× slower on the board than on the host. Each table states its own
yardstick.

### HOST — Apple Silicon, the tier the application runs on

Yardstick: llama.cpp on the same host, 98.9 tok/s @ 1.23 GB.

| Change | Mechanism | Result | Tag |
|---|---|---|---|
| Starting point | canonical loader dequantizes the whole model to FP32 | 0.17 tok/s @ 8.07 GB | `perf/baseline-2026-06-23` |
| Keep weights packed | quantized blocks consumed in place, never expanded | **1.80** @ 5.50 GB (10.6×) | `perf/a1-packed-llama` |
| Right-size the heap | `-Xmx12g → 2g`; the 5.5 GB was JVM headroom, not working set | **2.11** @ 1.94 GB | `perf/a1b-jvm-heap` |
| Fused decode attention | skip GQA concat + SDPA plumbing on the decode path | **3.82** (40-tok) | `perf/a2-fused-decode` |
| Streaming detokenization | per-token detok fix | **4.27** | `streaming-detok-spaces` |
| Upstream engine work | inherited through a version bump, no local change | **10.6** @ ~2.2 GB | `perf/deps-0.34-jvm-win` |

**62× faster at 3.7× less memory** — every step a data-movement decision, not an arithmetic one.
Re-verified 2026-08-11 on SKaiNET 0.39.1: 10.5 tok/s @ 2.05 GB.

### BOARD — SL2619, 2× Cortex-A55, 1.92 GB, no swap

Yardstick: llama.cpp built for and run on this board, 2.8 tok/s @ 0.70 GB
(`board-llamacpp-baseline`).

| Change | Mechanism | Result | Tag |
|---|---|---|---|
| First attempt | raw and packed tensors both resident during load | **OOM-killed** at ~3.2 GB peak | — |
| Fuse load and packing | pack during the streaming load; raw bytes never accumulate | **fits**: 1.48 GB peak, 992 MB steady, correct at 51 s/tok | `board-run-fused-fits` |
| Neon kernels | SDOT integer dot products over packed GGUF blocks, compiled on-board | 51 → **8.1 s/tok** (6.3×) | `board-neon-kernels` |
| Cache-locality reorder | block-outer loop order, weight bytes stream sequentially; bit-identical output | matmul 2.07×; 8.1 → **5.4 s/tok** | `perf/a2b-q4k-cache-locality` |
| Compiled path on-board | int8 quantization to fit: 4.2 GB FP32 → 1.1 GB `.irpa` | correct, ids identical to eager | `iree-int8-board` |

A workload that gets OOM-killed has no tok/s. *Killed → correct* is not a percentage.

We are still **~15× behind llama.cpp on the board** — measured and attributed, not estimated:
~46% quantized Neon matmul, ~54% runtime tail, both profiled into named buckets
(`SKAINET_PROFILE=1`).

## How it works

Two execution paths that share no machinery and must agree:

- **Eager** — Kotlin/Native compiles the runtime straight to a `linuxArm64` executable. GGUF
  weights load **in packed quantized form** (Q4_K/Q6_K blocks are never expanded to FP32) and
  dispatch through a priority-ranked kernel registry: scalar reference (0) → JVM Panama SIMD
  (50) → native Neon (100). A missing kernel degrades, never fails.
- **Compiled** — the same model graph exported from SKaiNET's NN DSL to StableHLO MLIR, compiled
  by IREE for aarch64, run on-device by `iree-run-module`.

Both paths emit the same greedy token ids — `29901, 1724, 338, 278` — and since they share
nothing above the weights, that agreement is the project's numerical ground truth.

## The KleidiAI yardstick

We compiled Arm's KleidiAI dot-product micro-kernels on the board with our exact flags and raced
them against our Q4_K kernel at real TinyLlama decode shapes. Decode at batch 1 is memory-bound,
so the honest unit is effective weight-streaming bandwidth: **ours 1.7 GB/s, KleidiAI 3.3 GB/s —
they win by 1.95×.** That told us two things: the kernels are scheduling-bound, not DRAM-bound
(Arm reaches 3.3 GB/s single-threaded on the same silicon), and the gap is closable without new
hardware. We did not vendor KleidiAI: its `qsi4c32` format throws away the per-32 asymmetric
minima our Q4_K keeps, and it has no Q6_K equivalent. The full decision:
[`docs/KLEIDIAI-EVALUATION.md`](docs/KLEIDIAI-EVALUATION.md). Run the comparison yourself in ~90
seconds: [`benchmarks/kleidiai/`](benchmarks/kleidiai/).

## Try it — the reproduction ladder

Most judges don't have an SL2619. Every claim except the board rows is verifiable on hardware you
already own; each tier below was actually run, on the stated date, before being written here.

**Prerequisites:** JDK 21+. The build needs **only Maven Central** — the fused load+pack that
makes the board fit possible shipped upstream in SKaiNET-transformers 0.36.0, so there are no
sibling checkouts and no composite flags.

### Tier 0 — any machine: a clean clone builds *(verified 2026-08-11, 4m42s)*

```bash
GRADLE_USER_HOME=$(mktemp -d) ./gradlew --no-build-cache --refresh-dependencies \
  :model:jvmTest :eager:compileKotlinJvm :eager:compileKotlinLinuxArm64
./scripts/check-no-leakage.sh
```

### Tier 1 — any Arm64 + JDK 21: real generation, in pure Kotlin *(verified 2026-08-11: 10.5 tok/s)*

Apple Silicon, Graviton, Ampere, Pi 5 — anything AArch64.

```bash
./scripts/download-models.sh   # 668 MB Q4_K_M GGUF
./gradlew :bench:runJvm --args='bench --variants eager-jvm --tokens 40 --ctx 256 \
  --temperature 0.01 --prompt "What is quantization?"'
```

Correct output starts **"Quantization is the process of converting…"** and the summary line
prints tok/s and RSS. Add `--variants python-baseline,eager-jvm` to race the llama.cpp reference
on your own machine (needs `uv`).

**Time to first token** *(verified 2026-08-14: ~1.1 s median, 1.1–1.3 s across 6 runs)* — the
same command with `--tokens 1`: the harness already reports load time separately from inference
time, so with exactly one token to produce, "inference time" *is* time-to-first-token.

```bash
./gradlew :bench:runJvm --args='bench --variants eager-jvm --tokens 1 --ctx 256 \
  --temperature 0.01 --prompt "What is quantization?"'
```

### Tier 1b — Apple Silicon: the Kotlin/Native path *(verified 2026-08-13: 25.6 tok/s, 40-tok)*

The exact binary technology the board runs, no extra toolchain, and — as of SKaiNET 0.40.1 —
the fastest path on this host: a real Neon archive now ships for macosArm64, fixing a silent
scalar-kernel fallback (`perf/apple-neon-macos`).

```bash
./gradlew :eager:linkReleaseExecutableMacosArm64
./eager/build/bin/macosArm64/releaseExecutable/tinyllama-skainet.kexe \
  eager --model Q4_K_M --tokens 40 --ctx 256 --temperature 0.01 --prompt "What is quantization?"
```

### Tier 2 — Arm64 Linux: the board binary *(cross-link from a Maven-only clone verified 2026-08-11)*

```bash
./gradlew :eager:linkReleaseExecutableLinuxArm64
./eager/build/bin/linuxArm64/releaseExecutable/tinyllama-skainet.kexe \
  eager --model Q4_K_M --tokens 8 --ctx 128
```

At startup the kernel registry logs which provider it installed — on an A55-class core the Neon
provider banner is the assertion that you are running the measured configuration.

### Tier 3 — the SL2619 itself: audit the board rows

The board numbers can't be re-run without the hardware; they can be audited. Every board row in
[`docs/perf-history.csv`](docs/perf-history.csv) names its tag; mechanics and logs are in
[`docs/PERF-LOGBOOK.md`](docs/PERF-LOGBOOK.md) and [`docs/BOARD_SL2619.md`](docs/BOARD_SL2619.md).
With a board: `export ADB_SERIAL=<board-ip>:5555`, then `./scripts/adb-board-run.sh help`.

### The compiled path (StableHLO → IREE)

```bash
./gradlew exportStableHlo                 # NN DSL → StableHLO MLIR
./scripts/build-iree-int8.sh              # int8 quantize + iree-compile for aarch64
ADB_SERIAL=<board-ip>:5555 ./scripts/decode-board.sh <vmfb> <irpa> ...
```

Mind the runtime/compiler version trap documented in
[`scripts/decode-board.sh`](scripts/decode-board.sh): a 3.11-compiled vmfb fails on a 3.10
runtime with only `requires [Ch]` as the diagnostic.

## Reusable artifacts

Nine, each usable without the rest of this project — the kernel-vs-KleidiAI harness, the Neon
GGUF kernels (upstreamed), the profiling instrument, the perf-log protocol, the IREE runbook, the
board bring-up notes, and the negative results. One table: [`ARTIFACTS.md`](ARTIFACTS.md).

## What we did not achieve

- **Board tok/s parity.** 5.4 s/token vs llama.cpp's 0.36 — 15×, attributed above. The next
  levers are known and sized: the runtime tail (~54%), threading across both cores (~1.3–1.5×
  bound by Amdahl), and a Q6_K dequant rewrite.
- **Full-model compiled decode.** The compiled path runs a single decoder step with re-prefill,
  not a KV-cached loop.
- Three logged optimizations produced **no win** — they are in the ledger anyway, because a
  benchmark log with no negative results isn't a ledger.

## Updated during the submission window

This project began 2026-06-22 and was significantly updated during the submission period:
dependency work moved the host number 4.27 → 10.6 tok/s (`perf/deps-0.34-jvm-win`), the
build was made reproducible from Maven Central alone (SKaiNET 0.39.x, 2026-08-11), the board
address was factored out of source, and the licensing, attribution, and leak-scrub work landed —
each with its own dated commit and, where a number moved, a CSV row.

## Credits

TinyLlama-1.1B weights by the TinyLlama project, GGUF quantization by TheBloke (Apache-2.0,
attribution in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The Python baseline under
`benchmarks/python/` derives from Arm's TinyLlama edge sample (MIT © 2025 Arm Examples).
KleidiAI and IREE are referenced as tools, never vendored.

*Arm and Neon are registered trademarks or trademarks of Arm Limited (or its subsidiaries or
affiliates) in the US and/or elsewhere.*
