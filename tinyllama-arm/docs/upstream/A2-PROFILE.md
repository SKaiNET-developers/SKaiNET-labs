# A2 profile — the host bottleneck is attention, NOT the quantized matmul

**Date:** 2026-06-23 · **Host:** macOS arm64, 14 cores, JDK 21 + Panama Vector ·
**Subject:** eager-jvm, TinyLlama Q4_K_M, packed NATIVE_OPTIMIZED path (after `perf/a1b-jvm-heap`).

A2 was originally scoped as *"ARM NEON + cache-blocked GEMM"* — i.e. make the **matmul** faster.
Profiling says that's the wrong target. **The matmul is not the bottleneck.**

## Evidence

### 1. Which kernel actually runs (KERNEL_DIAG=1)
```
[kernel-diag] cores=14 providers=3
[kernel-diag]   native-ffm     priority=100 available=true q4k=true   <- WINS
[kernel-diag]   panama-vector  priority=50  available=true q4k=true
[kernel-diag]   scalar         priority=0   available=true q4k=true
```
Surprise: the native FFM `libskainet_kernels.dylib` **is** bundled on macOS arm64 (the
Explore pass assumed board-only). So the packed Q4_K matmul dispatches to the **native-ffm**
kernel at priority 100 — a **serial C `for o` loop with no threading** (`q4k_matmul.c`).
The Panama Kotlin kernel (priority 50) *does* thread via `parallelChunks`, but loses the
priority race.

### 2. CPU utilisation during decode (14 cores ⇒ 1400% = full)
| kernel | tok/s | CPU% | cores busy |
|---|---|---|---|
| native-ffm (serial C, default) | 2.11 | ~84% | ~0.8 |
| panama-vector (`-PexcludeNativeCpu=true`) | ~2.5 | ~115% | ~1.1 |

Even the *parallel* kernel only reaches ~1.1 cores, and serial→parallel matmul buys just **~20%**.
If matmul were the bottleneck, the parallel kernel on 14 cores would be many× faster. It isn't.

### 3. Stack-sample profile (jstack ×12 during inference, `sk.ainet` leaf frames)
```
MultiHeadAttention.attentionImpl / attentionImpl
MultiHeadAttention.repeatKVHeads          <- GQA expand via concat(32 slices)/token/layer
DenseTensorDataFactory.init               <- intermediate-tensor allocation
DenseFloatArrayTensorData.get → calcFlatIndex → access   <- GENERIC per-element indexing
DefaultCpuOpsBase.reshape / concat / unsqueeze           <- KV-cache + head reshapes
```
**No Q4_K matmul frame appears at all.** The time is in the attention path: generic
per-element tensor access (multi-dim stride math per element), intermediate allocations, the
GQA `repeatKVHeads` concat, and KV-cache reshapes/concat.

## Re-scoped A2 (the real levers, ranked)

1. **Fused decode-path SDPA + buffer-direct access** — *(core: ../SKaiNET backend-cpu)*.
   Replace the generic `get`/`calcFlatIndex` element access and intermediate tensor
   allocations in the seqLen==1 attention path with a single FloatArray-direct kernel
   (scores → softmax → weighted-V in one pass). Biggest lever — this is where the samples land.
2. **GQA without `concat`** — *(transformer-core: SKaiNET-transformers, in composite)*.
   `repeatKVHeads` builds `[nHeads, seq, headDim]` by `concat`-ing 32 narrow slices every
   token every layer. Make SDPA `n_rep`-aware (broadcast KV heads) so the expansion never
   materialises.
3. **Kernel priority on many-core hosts / parallelise native FFM** — *(core)*. Either give
   the native FFM C kernel a `startO/count` API so `parallelChunks` can fan it across cores,
   or de-rank serial native kernels below parallel SIMD when `cores >> 1`. Worth ~20% alone,
   but only after (1)/(2) since matmul isn't dominant.

NEON/cache-blocking (the original A2 wording) stays relevant **only for the board** (few cores,
no Panama, native NEON is the right kernel) — verify on SL2619/QEMU, not measurable on host.

## Repro
```
# which kernel + correctness
KERNEL_DIAG=1 ./gradlew -PuseLocalSkainet=true :bench:runJvm \
  --args='bench --variants eager-jvm --tokens 16 --ctx 256 --temperature 0.01 --prompt "What is quantization?"'
# force the parallel Panama kernel (drop serial native-ffm)
./gradlew -PuseLocalSkainet=true -PexcludeNativeCpu=true :bench:runJvm --args='bench --variants eager-jvm ...'
# CPU sampling: top -l 2 -pid <worker> -stats cpu ; profile: jstack <worker>
```
Build note: the nested composite (`includeBuild ../SKaiNET-transformers`, which itself includes
`../SKaiNET`) substitutes **both** transformers and core from source — so all three levers are
editable here and recompile via the composite. `transformer-core` (lever 2) is the lowest-risk.
