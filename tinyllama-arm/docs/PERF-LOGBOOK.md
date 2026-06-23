# Performance logbook — TinyLlama on SKaiNET

Trackable protocol for closing the llama.cpp gap (packed + SIMD) and the IREE path. **Every
relevant improvement = one annotated git tag + one entry here.** Roadmap:
`docs/upstream/PERF-BASELINE.md` (baseline detail) and the plan; phase log:
`docs/upstream/LLAMA-FIX-PROGRESS.md`.

## Conventions
- **Tag:** annotated, prefix `perf/` (e.g. `perf/a1-packed-llama`), in the repo where the change
  is measured end-to-end (this repo for bench-visible wins; upstream repos tagged separately).
  The tag message == the entry's What/Impact so `git tag -n` and this file agree. Cross-repo
  commits add a trailer `Perf-Tag: perf/<id>`.
- **Helper:** `scripts/perf-log.sh <id> "<title>"` runs the bench, appends an entry below + a row
  to `docs/perf-history.csv`, then (on confirm) creates `perf/<id>`.
- **Entry template:**
  ```
  ### perf/<id> — <title>  (<date>)
  - What:   one line.
  - How:    mechanism, repo @ short-sha, key files.
  - Impact: tok/s X→Y, RSS A→B vs baseline; correctness.
  - Run:    bench command + headline result.
  ```

## Latest metrics (host, TinyLlama Q4_K_M, greedy)
| variant | tok/s | RSS | correct? | tag |
|---|---|---|---|---|
| python-baseline (llama.cpp) | 98.9 | 1.23 GB | ✅ | perf/baseline-2026-06-23 |
| eager-jvm (fused decode attn) | **2.77** (16-tok) / **3.82** (40-tok) | 2.1 GB | ✅ matches llama.cpp | perf/a2-fused-decode |
| eager-jvm (packed + `-Xmx2g`) | 2.11 | 1.9 GB | ✅ matches llama.cpp | perf/a1b-jvm-heap |
| eager-jvm (packed, `-Xmx12g`) | 1.80 | 5.5 GB | ✅ matches llama.cpp | perf/a1-packed-llama |
| eager-jvm (dense FP32, was baseline) | 0.17 | 8.07 GB | ✅ | perf/baseline-2026-06-23 |
Trend: `docs/perf-history.csv`.

## Pipeline (and where the gap is)
```mermaid
flowchart LR
  GGUF[Q4_K_M GGUF] --> LD{loader}
  LD -->|DEQUANTIZE_TO_FP32 ~8GB| FP32 --> OPT[OptimizedLLMRuntime]
  LD -->|NATIVE_OPTIMIZED ~0.7GB A1| PK[packed Q-Block] --> OPT
  OPT --> K[CPU kernels: scalar / Panama / NEON A2]
  GGUF --> EXP[DSL to DAG to StableHLO] --> VMFB[iree-compile] --> IREE[iree-run-module B]
```

## Responsibility split + dependencies
```mermaid
flowchart TB
  subgraph TF[skainet-transformers]
    A1[A1 Llama packed loader] --> B1[B1 Llama IREE runtime]
  end
  subgraph CORE[skainet-core]
    A2[A2 ARM NEON + cache-blocking] --> A3[A3 mmap + packed gather]
  end
  A1 --> A2
  B1 --> B2[B2 quantized .irpa]
```

## Schedule (review 2026-07-07)
```mermaid
gantt
  dateFormat YYYY-MM-DD
  section Track A (CPU)
  A1 packed loader (TF)     :a1, 2026-06-24, 2d
  A2 NEON + blocking (core) :a2, after a1, 4d
  A3 mmap (core)            :a3, after a2, 2d
  section Track B (IREE)
  B1 IREE runtime           :b1, 2026-06-24, 3d
  B2 quantized .irpa        :b2, after b1, 3d
  section Review
  2-week review             :milestone, 2026-07-07, 0d
```

---

# Entries (newest first)

### perf/a2-fused-decode — Fused decode-attention fast path  (2026-06-23)
- What:   The decode-path attention (seqQ==1) now runs as one buffer-direct kernel instead of
  the generic op chain. Acts on the A2 profile (matmul was never the bottleneck — attention was).
- How:    SKaiNET-transformers @ 3791f88 (`MultiHeadAttention.kt`, transformer-core). New
  `fusedDecodeAttention`: scores→softmax→GQA-weighted-V straight from the cached K/V buffers,
  emitting merged `[1, qDim]`. Bypasses `repeatKVHeads` concat (rebuilt every token×layer), the
  `unsqueeze→SDPA→squeeze→permute` chain, and the intermediate allocations those create. Same
  max-stable softmax + head→kv-head mapping as the general path → bit-for-bit-equivalent output.
  Guarded to self-attn, no sliding window; prefill (seqLen>1) keeps the general SDPA path.
- Impact: eager-jvm **2.11 → 2.77 tok/s** (16-tok headline) / **2.57 → 3.82 tok/s** (40-tok
  sustained, ~1.5×). RSS ~2.1 GB unchanged. Output identical + coherent. Gain compounds with
  generation length (KV grows; plumbing cost was per-token×layer).
- Run:    `./gradlew -PuseLocalSkainet=true :bench:runJvm --args='bench --variants eager-jvm --tokens 40 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'` → 3.82 tok/s.

### a2-profile — Bottleneck is attention, not matmul  (2026-06-23, investigation)
- What:   Profiled the eager-jvm packed decode path before optimising. **The quantized matmul
  is NOT the bottleneck** — parallelising it buys only ~20%. Full write-up: `docs/upstream/A2-PROFILE.md`.
- How:    `KERNEL_DIAG=1` (kernel registry), `top -l 2` (CPU%), `jstack` ×12 (hot frames).
  Findings: (a) native-ffm Q4_K kernel (serial C, priority 100) is bundled on macOS arm64 and
  outranks the parallel Panama kernel → decode pins to ~0.8 core; (b) even parallel Panama only
  reaches ~1.1 cores; (c) jstack hot frames are attention — `MultiHeadAttention.attentionImpl`,
  `repeatKVHeads` (GQA concat/token/layer), generic per-element `DenseFloatArrayTensorData.get →
  calcFlatIndex`, and intermediate `DenseTensorDataFactory.init` allocations. No matmul frame appears.
- Impact: Re-scopes A2: original "NEON + cache-blocked GEMM" was matmul-centric. Real levers now
  ranked — (1) fused decode SDPA + buffer-direct access [core], (2) GQA without concat
  [transformer-core], (3) parallelise/de-rank serial native kernel [core]. NEON stays board-only.
- Tooling: added `KERNEL_DIAG=1` (eager) + `-PexcludeNativeCpu=true` (bench) for repeatable diagnosis.

### perf/a1b-jvm-heap — Right-size the JVM heap (RSS 5.5 → 1.9 GB)  (2026-06-23)
- What:   The packed path's 5.5 GB RSS was JVM heap *headroom* from `-Xmx12g`, not working set.
  Capping the heap closes the memory half of the llama.cpp gap.
- How:    tinyllama `bench/build.gradle.kts` `-Xmx12g → -Xmx2g` (both `application` defaults and
  the `runJvm` task). Floor probed: `-Xmx2g` runs clean, `-Xmx1536m` OOMs → ~1.6 GB true working
  set (≈0.7 GB packed weights + 0.26 GB FP32 embedding + activations/KV + JVM overhead).
- Impact: eager-jvm **RSS 5.5 → 1.9 GB** (vs llama.cpp 1.23 GB — gap mostly closed) and
  **1.80 → 2.11 tok/s** (tighter heap = less GC). Still coherent, top-10 logits == llama.cpp.
- Run:    `./gradlew -PuseLocalSkainet=true :bench:runJvm --args='bench --variants eager-jvm --tokens 16 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'` → 2.11 tok/s, 1938 MB.

### perf/a1-packed-llama — Llama NATIVE_OPTIMIZED packed path (mirror Gemma)  (2026-06-23)
- What:   Packed-quant Llama eager: weights stay packed (Q4_K/Q6_K) + run the in-kernel matmul
  instead of dequantizing everything to FP32.
- How:    SKaiNET-transformers @ ccbd87e — new `LlamaQuantLayout.kt` + `LlamaPackedWeights.kt`
  (`convertLlamaWeightsPacked`: token_embd→FP32 gather, matrices→packed `Q*BlockTensorData`),
  wired into `LlamaNetworkLoader.load()` on NATIVE_OPTIMIZED. tinyllama @ 1116f9e: eager-jvm
  defaults to NATIVE_OPTIMIZED + composite `dependencySubstitution` fix (POM_ARTIFACT_ID).
- Impact: eager-jvm **0.17 → 1.80 tok/s (~10.6×)**, **8.07 → 5.5 GB**, still coherent + top-10
  logits == llama.cpp. (python-baseline 98.9 tok/s @ 1.23 GB.)
- Run:    `./gradlew -PuseLocalSkainet=true :bench:runJvm --args='bench --variants python-baseline,eager-jvm --tokens 16 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'`

### perf/baseline-2026-06-23 — Starting point  (2026-06-23)
- What:   First correct SKaiNET TinyLlama inference (eager-jvm) + full measurement baseline.
- How:    eager-jvm switched to the canonical `LlamaNetworkLoader.fromGguf(DEQUANTIZE_TO_FP32).load
  + OptimizedLLMRuntime` path (matches llama.cpp). skainet-tinyllama-iree @ 9433267. Bug was our
  custom loaders, not SKaiNET. IREE: real 22-layer graph exports (1466 nodes, 0 unsupported) + vmfb.
- Impact: eager-jvm 0.17 tok/s @ 8.07 GB (correct); python-baseline 95.6 tok/s @ 1.23 GB. Gap ~570×
  — pure FP32 dequant, no packed/SIMD yet. Correctness milestone, not speed.
- Run:    `./gradlew -PuseLocalSkainet=true :bench:runJvm --args='bench --variants python-baseline,eager-jvm --tokens 16 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'`
