# KleidiAI evaluation — reusing Arm's micro-kernels for the quantized eager path

**Date:** 2026-07-09 · **Question:** instead of maintaining hand-written NEON kernels in
`skainet-backend-native-cpu`, can we reuse [Arm KleidiAI](https://github.com/ARM-software/kleidiai)
(Apache-2.0) micro-kernels — and what would it buy on the SL2619 board?

## TL;DR

- KleidiAI's dotprod GEMV micro-kernels (`matmul_clamp_f32_qai8dxp_qsi4c32p`) are directly
  comparable to our `skainet_q4k_matmul`: f32 activations in, int8-dynamic-quant + int-dot inside,
  f32 out, ~4.5 bits/weight streamed. They run on the A55 (dotprod, no i8mm needed).
- Host microbenchmark (M4 Pro, single thread, **A55-parity flags** `-march=armv8.2-a+fp16+dotprod`,
  ≥96 MB rotating working set): **KleidiAI ≈ 2.1–2.3× faster than `skainet_q4k`** at every
  TinyLlama-1.1B decode shape, ~2.9× vs `q6k`, ~5–6× vs `q8_0`.
- Board microbenchmark (SL2619 Cortex-A55, same sources compiled on-board with gcc 13.3,
  same flags): **the win holds — ~1.95–2.0× vs `skainet_q4k`, ~3.7× vs `skainet_q6k`,
  ~7.5× vs `skainet_q8_0`**. Summed over one decode token, the matmul slice drops from
  ~445 ms to ~178 ms single-thread (**2.5×**, because the Q6_K tensors gain the most).
- Cost of adoption: KleidiAI does not speak ggml block formats. Weights must be **repacked at
  load** into its `qsi4c32p` layout — exact for Q4_0, **lossy for Q4_K** (drops the per-32
  asymmetric min), and **no 6-bit equivalent for Q6_K** (nearest option: int8 `qsi8cxp`,
  +30% bytes on those tensors, or 4-bit with real quality loss on `output`/`ffn_down`/`attn_v`).
- The `KernelProvider` registry seam (priority 100 NEON / 0 scalar) makes this a low-risk,
  per-format, A/B-able integration; the fused load+pack hook (SKaiNET PR #220) is where the
  repack would live.

## Why KleidiAI (vs ggml / XNNPACK / BLAS)

| option | license | fits `KernelProvider` seam | format fit | notes |
|---|---|---|---|---|
| **KleidiAI** | Apache-2.0 | ✅ single-thread n-splittable micro-kernels | ⚠️ repack required (`qsi4c32p`), Q4_K lossy, no Q6_K | built for embedding; stable releases; asm scheduled by Arm; SME2-ready for future silicon |
| **ggml (llama.cpp)** | MIT | ✅ `vec_dot` row primitives | ✅ **bit-identical** Q4_K/Q5_K/Q6_K/Q8_0 | fastest path to a win; no stable API — vendor a snapshot |
| XNNPACK | Apache-2.0 | ❌ owns graph + threadpool | ❌ | adopt-as-backend only |
| OpenBLAS/BLIS | BSD | ✅ | fp32/bf16 only | decode is quantized-matmul-bound; irrelevant |

ggml remains the zero-repack, zero-accuracy-risk alternative; this document quantifies the
KleidiAI side. (The two are not exclusive: ggml for K-quants, KleidiAI where its layout wins.)

## What was measured

Single-thread m=1 GEMV at TinyLlama-1.1B decode shapes (hidden 2048, GQA kv 256, ffn 5632,
vocab 32000). Timed per call for **both** sides: activation→int8 quantization + matmul
(SKaiNET kernels quantize activations internally; KleidiAI's `kai_run_lhs_quant_pack_qai8dxp_f32`
is inside the timed region). Weight packing excluded on both sides (GGUF arrives packed; KleidiAI
RHS packing is a load-time step). Weights rotate over ≥96 MB so nothing lives in cache — the
real decode regime where all ~640 MB of weights stream per token.

Harness: `benchmarks/kleidiai/bench.c` (also staged on the board at
`/home/skainet-tinyllama/kleidiai-bench/` with sources, binary, and `bench.log`); KleidiAI @
`main` (2026-07-09 shallow clone); SKaiNET kernels from `skainet-backend-native-cpu/native/src`
(the board-neon-kernels sources). Build (host or board):

```sh
cc -O3 -ffast-math -march=armv8.2-a+fp16+dotprod -DNDEBUG \
   -I <skainet>/native/include -I <kleidiai> \
   bench.c <skainet>/native/src/{q4k,q6k,q8_0}_matmul.c \
   <kleidiai>/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/*neon_dotprod{.c,_asm.S} \
   <kleidiai>/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c \
   <kleidiai>/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c -o bench -lm
``` Weight bytes are random with sane fp16/bf16 scales — these
kernels are branch-free on data, so values don't affect timing.

### Host — Apple M4 Pro, dotprod-only kernel selection (i8mm not used)

| shape (n×k) | skainet q4k | skainet q6k | skainet q8_0 | best KleidiAI dotprod GEMV | q4k speedup |
|---|---|---|---|---|---|
| attn q/o 2048×2048 | 96.3 µs | 129.7 µs | 273.4 µs | **45.2 µs** (1x4_4x4) | **2.13×** |
| attn kv 256×2048 | 12.7 µs | 16.8 µs | 33.7 µs | **5.8 µs** | **2.19×** |
| ffn gate/up 5632×2048 | 296.6 µs | 374.5 µs | 755.6 µs | **121.4 µs** | **2.44×** |
| ffn down 2048×5632 | 310.8 µs | 387.8 µs | 1018.6 µs | **159.1 µs** | **1.95×** |
| lm_head 32000×2048 | 1611.3 µs | 2483.3 µs | 4213.4 µs | **692.0 µs** | **2.33×** |

Effective weight-streaming rate: skainet q4k ≈ 21–24 GB/s; KleidiAI ≈ 41–54 GB/s — same
bytes/weight, so the gap is instruction scheduling and packed-layout locality, not bandwidth.
Of the three GEMV variants, `qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod` won every shape on this
core; `…8x8_1x8x32` was within ~10%.

### Board — SL2619 Cortex-A55 @ ~2 GHz, 1 thread (same sources, gcc 13.3, same flags)

Run 2026-07-09 on the board itself (`/home/skainet-tinyllama/kleidiai-bench/bench.log`).
On the A55 the best variant flips to `qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod`
(the 1x4 that won on M4 is ~10% behind here).

| shape (n×k) | skainet q4k | skainet q6k | skainet q8_0 | best KleidiAI dotprod GEMV | q4k speedup | q6k speedup |
|---|---|---|---|---|---|---|
| attn q/o 2048×2048 | 1407.7 µs | 2653.5 µs | 5354.9 µs | **718.0 µs** | **1.96×** | 3.70× |
| attn kv 256×2048 | 194.9 µs | 352.4 µs | 676.7 µs | **96.5 µs** | **2.02×** | 3.65× |
| ffn gate/up 5632×2048 | 3855.6 µs | 7261.3 µs | 14803.1 µs | **1971.9 µs** | **1.96×** | 3.68× |
| ffn down 2048×5632 | 3870.3 µs | 7303.2 µs | 14807.1 µs | **1987.5 µs** | **1.95×** | 3.67× |
| lm_head 32000×2048 | 21640.8 µs | 41146.2 µs | 84582.8 µs | **11221.9 µs** | **1.93×** | 3.67× |

Effective weight streaming: skainet q4k **1.7 GB/s**, q6k 1.3 GB/s, q8_0 0.83 GB/s;
KleidiAI **3.3 GB/s** — single-core, above the ~1.8 GB/s that llama.cpp's whole decode
achieves with 2 threads (2.8 tok/s × 640 MB), so the A55's DRAM is not the ceiling for
the current kernels; instruction scheduling is.

**Per-token projection (decode, single thread, matmul slice only).** Using the Q4_K_M
tensor mix (q/o/gate/up/k = Q4_K; v/down/lm_head = Q6_K), 22 layers + lm_head:

| kernels | per layer | + lm_head | total/token | matmul-only ceiling |
|---|---|---|---|---|
| skainet (current) | 18.38 ms | 41.1 ms | **445 ms** | 2.2 tok/s |
| KleidiAI dotprod | 7.56 ms | 11.2 ms | **178 ms** | 5.6 tok/s |

(The Q6_K rows assume requantization to 4-bit `qsi4c32p`; with the safer int8 `qsi8cxp`
mapping those three tensor groups stream ~30% more bytes, landing between the two rows.)
This is the kernel slice only — the board profile attributes ~54% of wall time to the
non-matmul tail (K/N alloc/GC), which this change does not touch.

## Format mapping (the real cost)

TinyLlama Q4_K_M tensor mix (from the 2026-06-29 logbook entry): 135× Q4_K, 21× Q6_K
(10× `ffn_down` [5632,2048], 10× `attn_v` [2048,256], 1× `output` [2048,32000]), rest F32.

| GGUF format | KleidiAI target | repack | accuracy |
|---|---|---|---|
| Q4_0 (32-elem, fp16 scale, symmetric) | `qsi4c32p` (32-elem, bf16 scale, symmetric) | nibble reorder + fp16→bf16 | **exact** (bf16 scale round-trip is the only delta) |
| Q4_K (256-superblock, 6-bit sub-scales **+ mins**) | `qsi4c32p` | requantize: fold min into symmetric per-32 scale | **lossy** — drops the asymmetric min that is Q4_K's edge over Q4_0 |
| Q6_K (6-bit) | none — nearest `qsi8cxp` (int8 per-channel) | requantize 6→8 bit (upcast, near-lossless) | +2 bpw ≈ **+30% bytes** on 21 hot tensors (~+18 MB) |
| Q8_0 | `qsi8cxp` / `qsi8d32p` | scale conversion | near-exact |

So the clean KleidiAI story is a **Q4_0 model** (llama.cpp's KleidiAI backend made the same
choice). For our Q4_K_M model the honest options are (a) requantize-and-measure perplexity,
(b) switch the board artifact to Q4_0 GGUF, or (c) keep ggml-layout kernels for K-quants and
use KleidiAI only where the mapping is exact.

## Integration sketch

1. Vendor the needed KleidiAI files (~10 .c/.S/.h, Apache-2.0 headers intact) into
   `skainet-backend-native-cpu/native/`, alongside the existing sources — they compile with the
   exact board flags already in use.
2. Add a `KleidiKernelProvider` (priority 200) registering matmul for the formats it serves;
   ggml-layout NEON (100) and scalar (0) remain fallbacks. Per-format, A/B-able at runtime.
3. Hook the repack into the fused load+pack path (SKaiNET PR #220): GGUF block →
   `kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0` at load, one pass, no extra resident copy.
4. Threading: KleidiAI GEMV kernels are pure single-thread primitives with an `n_step` split
   granularity — they drop into the existing 2-core row-split unchanged.
5. Prefill (m>1) can use the same RHS packing with GEMM variants (`4x4_16x4` dotprod on A55);
   on i8mm hosts the `4x8x32_neon_i8mm` variants apply — one packed layout serves both.

## The learning — KleidiAI as kernel yardstick, not replacement

First, read the result from the other side: our kernels are only **2× off Arm's own
hand-scheduled assembly**, written by the team with the best knowledge of this
micro-architecture that exists. ~1,000 lines of portable C with intrinsics, written inside a
Kotlin project, landing within 2× of that reference — after already taking the board from
51 s/token (scalar) to 5.4 s/token — is a strong validation of the in-house kernel approach,
not an indictment of it. The gap that remains is enumerable technique, not magic.

**Decision (2026-07-09): adopt KleidiAI as the kernel-level benchmark baseline** — the
per-GEMV analogue of what llama.cpp is end-to-end — **and beat it in our own kernels** by
applying the techniques it demonstrates. This is not a contradiction of the project story;
it sharpens it. The bench proves the gap is *not* physics: same core, same DRAM, same
bits-per-weight, same dotprod ISA — 2× is pure implementation technique. And our formats are
strictly *richer* (Q4_K keeps per-32 asymmetric mins that KleidiAI's symmetric `qsi4c32`
throws away), so matching its speed at our fidelity is a strictly stronger result than
adopting it.

What KleidiAI does that our kernels don't (transferable, in expected-impact order):

1. **nr-wide interleaved RHS panels.** Weights for 4–8 output rows are interleaved along k
   in kr-chunks, so one *linear* weight stream feeds 4–8 accumulators and every loaded
   activation vector is reused nr times. Our kernels walk one output row per pass. Nothing
   forces RAM layout to mirror GGUF's on-disk block layout — the fused load+pack hook
   (SKaiNET PR #220) is exactly where a panel repack belongs, keeping ggml semantics
   (scales + mins) while changing storage order.
2. **Scales resident next to codes.** Panel-adjacent scales stream in the same cache lines
   as the codes. Q4_K's 12-byte packed 6-bit scale header is decoded *in the hot loop* —
   hoist it at load into panel-adjacent fp16 arrays.
3. **Stay in the int domain.** The Q6_K kernel materializes a 256-float scratch via scalar
   6-bit unpack (the documented [[q6k-reorder-no-win]] bottleneck); KleidiAI never leaves
   int8 until the per-block epilogue. The measured 3.7× on Q6_K shapes is the bound on that
   rewrite — sdot-based 6-bit path, no float scratch.
4. **Instruction scheduling for the in-order A55.** KleidiAI's inner loops are hand-scheduled
   asm; ours trust gcc with intrinsics. The 2026-06-29 loop-reorder already proved this core
   punishes scheduling mistakes 2× — inspect the emitted inner loop, and keep asm as the
   last-resort option.

Success criterion: `skainet_q4k_matmul` ≥ 3.3 GB/s effective on the A55 (parity with
KleidiAI's dotprod GEMV at 4.5 bpw), measured by `benchmarks/kleidiai/bench.c`, while
staying bit-exact on ggml Q4_K semantics. Same harness, same board, no excuses.

## Caveats

- Host ratios did not fully transfer (2.1–2.4× on M4 vs 1.95× on A55, and a different
  winning variant) — always decide on board numbers. Both runs agree on the conclusion.
- Board eager-native is ~54% non-matmul tail (K/N alloc/GC) — kernel wins compound with,
  but don't replace, that work. Amdahl: a 2.5× matmul win alone roughly doubles decode
  speed only if the tail shrinks proportionally too.
- KleidiAI needs `k % 32 == 0` per block — all TinyLlama dims qualify.
- The 0.184 tok/s board best vs llama.cpp's 2.8 tok/s on the same board says most of the gap
  is not in any single kernel; this evaluation bounds what the kernel slice can contribute.
