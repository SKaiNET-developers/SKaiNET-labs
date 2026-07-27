# Arm kernel yardstick — your quantized GEMV vs Arm KleidiAI

**What this is.** A single-file microbenchmark that times in-house NEON kernels against
[Arm KleidiAI](https://github.com/ARM-software/kleidiai)'s `matmul_clamp_f32_qai8dxp_qsi4c32p`
dotprod micro-kernels at real TinyLlama-1.1B decode GEMV shapes (m=1, single thread).

**Who it's for.** Anyone writing quantized matmul kernels for AArch64 who wants to know how far
off they are from Arm's own hand-scheduled assembly — on their own hardware, not on a slide.
It runs on any AArch64 CPU with `asimddp` (dotprod): Cortex-A55 class boards, Raspberry Pi 5,
Graviton, Apple Silicon.

**Why it exists.** It is easy to declare a kernel "optimized". This measures the gap against a
reference maintained by the people who designed the microarchitecture. Our result and what we
did about it: [`docs/KLEIDIAI-EVALUATION.md`](../../docs/KLEIDIAI-EVALUATION.md).

## Fairness rules (why the numbers are comparable)

- Weight packing happens **once, outside** the timed loop on both sides — SKaiNET weights arrive
  pre-packed from GGUF, and KleidiAI RHS packing is a load-time step in a real integration.
- Per-token activation int8 quantization **is** timed on both sides: the SKaiNET kernels quantize
  internally per call, and KleidiAI's `kai_run_lhs_quant_pack_qai8dxp_f32` is inside the timed
  region.
- Weights rotate across a **≥96 MB** working set, so nothing lives in cache — the real decode
  regime, where every layer's weights stream from DRAM each token.
- Weight bytes are pseudo-random with sane fp16/bf16 scales, from a deterministic xorshift. These
  kernels are branch-free on data, so values don't affect timing.

## Build

Needs two checkouts alongside this repo: [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET)
(for the kernel sources and `skainet_kernels.h`) and KleidiAI.

```sh
git clone --depth 1 https://github.com/ARM-software/kleidiai       # Apache-2.0, not vendored
git clone --depth 1 https://github.com/SKaiNET-developers/SKaiNET  # MIT

cc -O3 -ffast-math -march=armv8.2-a+fp16+dotprod -DNDEBUG \
   -I SKaiNET/skainet-backends/skainet-backend-native-cpu/native/include -I kleidiai \
   bench.c \
   SKaiNET/skainet-backends/skainet-backend-native-cpu/native/src/{q4k,q6k,q8_0}_matmul.c \
   kleidiai/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/*neon_dotprod{.c,_asm.S} \
   kleidiai/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c \
   kleidiai/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c \
   -o bench -lm
./bench
```

`-march=armv8.2-a+fp16+dotprod` is Cortex-A55 parity. **Do not add `+i8mm`** — it is Armv8.6 and
will SIGILL on an A55. Confirm your target first: `grep Features /proc/cpuinfo` must list
`asimddp`.

To build on an x86 host instead, cross-compile with `aarch64-linux-gnu-gcc` and the same flags.
Host ratios do not fully transfer between microarchitectures — decide on target-hardware numbers.

## Output

One row per decode shape, µs per call, best-of-N, for each SKaiNET kernel and each KleidiAI GEMV
variant, plus effective weight-streaming GB/s. Shapes are TinyLlama-1.1B: attn q/o 2048×2048,
attn kv 256×2048, ffn gate/up 5632×2048, ffn down 2048×5632, lm_head 32000×2048.

Which KleidiAI variant wins is microarchitecture-dependent — `1x4_4x4` won on an M4 Pro,
`1x8x32` won on the A55. The harness times all of them.

## License

`bench.c` is MIT (see `../../LICENSE`). KleidiAI is Apache-2.0 and is **referenced, not
vendored** — its sources are cloned separately and no KleidiAI code is redistributed here. See
[`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
