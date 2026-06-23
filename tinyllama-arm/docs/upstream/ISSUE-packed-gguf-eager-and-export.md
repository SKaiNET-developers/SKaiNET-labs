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

## Related

- A genuine inference (not just export) fix likely lands in `skainet-transformers`
  (`inference-llama` / `apps.llm`) and would warrant a `skainet-transformers` release
  (it is still at `0.31.1` while core is `0.32.0`).
