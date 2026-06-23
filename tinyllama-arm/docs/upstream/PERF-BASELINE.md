# Performance baseline — starting point (2026-06-23)

Snapshot to measure the llama.cpp-gap work against. Plan:
`~/.claude/plans/foamy-bubbling-mountain.md`. Review target: **~2 weeks (≈2026-07-07)**.

Model: TinyLlama-1.1B-Chat-v1.0 **Q4_K_M**. Host: Apple Silicon (this Mac). Board: Synaptics
SL2619 (2× Cortex-A55, 2 GB, no swap).

## Numbers (host, prompt "What is quantization?", greedy)

| Variant | tok/s | RSS | load | Correct? | Notes |
|---|---|---|---|---|---|
| **python-baseline** (llama.cpp via uv) | **95.6** | 1.23 GB | ~inst (mmap) | ✅ | reference |
| **eager-jvm** (canonical `fromGguf(FP32)` + OptimizedLLMRuntime) | **0.17** | 8.07 GB | 1.6 s | ✅ matches llama.cpp | FP32 dequant; correctness milestone, not speed |
| eager-native (board, our packed hack) | ~0.0095 | ~0.74 GB | ~3.5 s | ❌ garbage | packed path broken |
| iree-cpu (toy step graph, board) | n/a | tiny | n/a | n/a (toy) | single-step latency only |

Gap: eager-jvm is **~570×** slower and **~6.5×** heavier than llama.cpp.

## Why (root-caused)
1. eager-jvm dequantizes to FP32 (8 GB) instead of keeping weights packed (~0.7 GB) → bandwidth-bound.
2. The packed path it would need (`NATIVE_OPTIMIZED`) is broken for Llama (`gather: unsupported
   input rank 1`) — Gemma's packed path works and is the template.
3. On Kotlin/Native the packed kernels fall back to **scalar** (ARM NEON in-tree but unverified/unwired).
4. No cache-blocked GEMM; no mmap; JVM overhead.

## What's done (this session)
- eager-jvm **correct** via the canonical upstream path (no SKaiNET change needed for host correctness).
- Root cause: tinyllama's own loaders (`loadPackedGgufLlamaWeights`/`mapCompactLlamaRuntimeWeights`/
  `PackedBlockTensorDataView`/`densify`), NOT SKaiNET. Upstream FP32 path matches llama.cpp.
- Composite build (`-PuseLocalSkainet=true`) wired to ../SKaiNET-transformers; parity harness in place.
- IREE: real 22-layer graph exports to StableHLO (1466 nodes, 0 unsupported) + compiles to aarch64 vmfb.

## Targets for the 2-week review (re-run `bench --variants python-baseline,eager-jvm`)
- A1 (TF Llama packed, mirror Gemma): eager packed **correct** + RSS ≤ ~1.5 GB (not 8 GB); uses core packed kernels.
- A2 (core ARM NEON + cache-blocking): tok/s up multi-× toward llama.cpp on the same host/board.
- A3 (core mmap): load ~instant, lower RSS.
- B (IREE): Llama IREE runtime prototyped; quantized `.irpa` so the real model runs on the board.

(Update this table at review time with the new run.)
