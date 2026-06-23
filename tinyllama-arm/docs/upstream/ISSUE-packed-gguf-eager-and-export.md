# Upstream issue draft: packed GGUF weights in the eager / graph-capture path

> Ready to file against the SKaiNET upstream. The StableHLO **lowering** is fine (core);
> the gap is in the **model→graph capture / eager runtime** (`skainet-transformers`,
> `inference-llama` / `apps.llm`). Versions used: transformers `0.31.1`, core `0.32.0`
> (also reproduced on core `0.31.2`).

## Title

`fromWeights` + `OptimizedLLMRuntime` (and tape-based StableHLO export) cannot consume packed GGUF weights

## Summary

The StableHLO lowering of a full TinyLlama-1.1B graph works perfectly — **1466 nodes, 0
unsupported markers** (`toStableHlo` / the 0.32.0 `StableHloGraphExporter`). But you can
only get that graph by hand-preprocessing the weights, because the standard path
(`LlamaNetworkLoader.fromWeights(weights)` driven by `OptimizedLLMRuntime` *or* recorded via
`DefaultGraphExecutionContext.tape`) fails on real packed GGUF weights. The same failure
makes the **JVM eager** runtime unusable on real GGUF — only the native
`LlamaRuntime` + `mapCompactLlamaRuntimeWeights` path works today.

## What works

- Native `LlamaRuntime` + `CpuAttentionBackend` + `mapCompactLlamaRuntimeWeights` runs real
  Q4_K_M inference (packed kernels; `chooseQuantizedMatmulHeap` dispatches `Q4_K`/`Q6_K`).
- `LlamaNetworkLoader.fromWeights` on a **synthetic dense** graph tapes and lowers cleanly.
- Real-weights lowering, once the weights are shape-corrected dense skeletons, lowers with
  **0 unsupported ops**.

## The gap (two failures, in order)

1. **Embedding gather on packed storage** — `Embedding.forward` → `gather` →
   `DenseTensorDataFactory.init` throws `class java.lang.Byte cannot be cast to class
   java.lang.Float`. The gather has no packed (Q6_K) path. This is also exactly why JVM
   eager (`OptimizedLLMRuntime` DIRECT) fails on real GGUF.

2. **Weight orientation** — the generic transformer layers'
   `linearProject` does `x @ w.t()` and expect `[out, in]`; GGUF projections load `[in, out]`
   (e.g. `attn_k [2048, 256]`, `ffn_gate [2048, 5632]`), and the token embedding loads
   `[dim, vocab]` while `Embedding` needs `[vocab, dim]`. The native `LlamaRuntime.linearProject`
   adapts (`if (wRows == xCols) x@w else x@w.t()`); the generic layers do not.

## Reproduce

```kotlin
val ctx = DirectCpuExecutionContext()
val weights = loadPackedGgufLlamaWeights("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", ctx) // packed
val model = LlamaNetworkLoader.fromWeights(weights)
model.forward(tokenTensor(1), DefaultGraphExecutionContext.tape(ctx.ops)) // -> Byte!=Float, then shape mismatch
```

## Ask

Make the eager / graph-capture path consume packed GGUF directly, e.g. one of:

- a packed-aware `gather` (dequantize-on-gather) so `Embedding` works on `PackedBlockStorage`;
- have `LlamaNetworkLoader.fromWeights` apply the same weight mapping the native runtime
  does (`mapCompactLlamaRuntimeWeights`: embedding reshape + orientation handling), so
  `fromWeights` is consistent across eager-JVM, eager-native, and export;
- or document the required preprocessing and expose a helper.

## Workaround we used (export only)

Because `embedConstants = false` makes weights external graph inputs (values irrelevant to
the exported structure), we replaced weights with **zero FP32 skeletons at transposed
shapes** purely for graph capture — no dequant, no large heap — which lowers to StableHLO
with 0 unsupported. This does **not** fix eager-JVM (which needs correct values + a packed
gather).

## UPDATE — deeper correctness bug (both backends, not just fromWeights)

Getting eager-jvm to *run* (via the compact `LlamaRuntime` path, same as native) revealed that
**SKaiNET inference is numerically wrong for TinyLlama-1.1B Q4_K_M on every path**, while
`llama.cpp` on the same GGUF is coherent:

- `python-baseline` (llama.cpp), greedy: "Quantization is the process of converting ..." ✓
- SKaiNET `LlamaRuntime` (native board AND JVM), greedy `--temperature 0.01`:
  `lstlstlstlstlst...` (degenerate single-token repetition) ✗
- SKaiNET `fromWeights`/`OptimizedLLMRuntime` (with the densify workaround): mixed-script
  gibberish ✗

### Localization (it is skainet-transformers, NOT core)

Bisected on JVM (`DEBUG_TOKENS`, `DEQUANT_PROJ` toggles in `:eager`):

1. **Tokenizer / decode — correct (transformers, but fine).** SKaiNET prompt IDs equal
   llama.cpp's exactly: `Question: What is quantization?\n\nAnswer:` →
   `[5462,303,291,29901,1724,338,4323,2133,29973,13,13,22550,29901]` (llama.cpp adds BOS=1,
   which LlamaRuntime prepends internally). Generated id `20155` decodes to a real token "lst".
2. **Core kernels — not the cause.** Dequantizing all non-embedding weights to dense FP32 and
   running the same runtime (`DEQUANT_PROJ=1`, bypassing the packed matmul kernels) still
   collapses (`19444…`). So core `matmul`/`dequant`/`gather` are functional.
3. **Forward ignores context — attention bug in the transformers runtime.** Greedy output is
   the SAME constant token for every prompt (`"The capital of France is"` and `"Once upon a
   time"` both → `20155…`); it depends only on the last token (`"Answer:"`). The model behaves
   like there is no attention/context mixing.

**Attribution: `skainet-transformers`, not `sk.ainet.core`.** Two distinct runtime defects:
- `runtime-kllama` `LlamaRuntime` + `CpuAttentionBackend` (the deprecated path): attention does
  not mix context → constant-token collapse (prompt-independent).
- `inference-llama` `OptimizedLLMRuntime` + `fromWeights` (the intended path): can't load packed
  GGUF (embedding gather), and even with the dequant+reorient workaround its output is *varied*
  (so its attention DOES mix context) but wrong → weight orientation / `WeightMapper` mapping.
  This is the better path to a correct eager result once orientation is fixed.

Core is clean. The fix(es) live in `skainet-transformers` (`runtime-kllama` attention and/or
`inference-llama` weight mapping) → warrants a `skainet-transformers` release.

Repro: `bench --variants python-baseline,eager-jvm --tokens 24 --temperature 0.01 --prompt "What is quantization?"`;
`DEBUG_TOKENS=1` prints prompt/generated ids; `DEQUANT_PROJ=1` forces dense weights.

## Related

- A genuine inference (not just export) fix likely lands in `skainet-transformers`
  (`inference-llama` / `apps.llm`) and would warrant a `skainet-transformers` release
  (it is still at `0.31.1` while core is `0.32.0`).
