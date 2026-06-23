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
| python-baseline (llama.cpp) | 95.6 | 1.23 GB | ✅ | perf/baseline-2026-06-23 |
| eager-jvm (canonical FP32) | 0.17 | 8.07 GB | ✅ matches llama.cpp | perf/baseline-2026-06-23 |
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

### perf/baseline-2026-06-23 — Starting point  (2026-06-23)
- What:   First correct SKaiNET TinyLlama inference (eager-jvm) + full measurement baseline.
- How:    eager-jvm switched to the canonical `LlamaNetworkLoader.fromGguf(DEQUANTIZE_TO_FP32).load
  + OptimizedLLMRuntime` path (matches llama.cpp). skainet-tinyllama-iree @ 9433267. Bug was our
  custom loaders, not SKaiNET. IREE: real 22-layer graph exports (1466 nodes, 0 unsupported) + vmfb.
- Impact: eager-jvm 0.17 tok/s @ 8.07 GB (correct); python-baseline 95.6 tok/s @ 1.23 GB. Gap ~570×
  — pure FP32 dequant, no packed/SIMD yet. Correctness milestone, not speed.
- Run:    `./gradlew -PuseLocalSkainet=true :bench:runJvm --args='bench --variants python-baseline,eager-jvm --tokens 16 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'`
