# Llama real-GGUF inference fix — progress log

**Goal:** make SKaiNET produce *coherent* TinyLlama output (matching llama.cpp) by fixing the
real-GGUF Llama weight loading/orientation in `../SKaiNET-transformers`, mirroring the working
Gemma path. Verified end-to-end via the composite build in this repo
(`-PuseLocalSkainet=true`).

Plan: `~/.claude/plans/foamy-bubbling-mountain.md`. Issue: `ISSUE-packed-gguf-eager-and-export.md`.

## Root cause (confirmed)
Llama real-GGUF path is untested + buggy upstream; Gemma is the validated template:
- `DecoderGgufWeights` lacks `logicalShapes` (Gemma has it) → matrix dims lost for packed weights.
- No commonMain packed converter for Llama (Gemma: `convertGemmaWeightsPacked`); jvmMain MemSeg
  converter double-transposes K-quants.
- No real-GGUF Llama parity test.
Verified clean: tokenizer, `DecoderRuntime.generate`, `CpuAttentionBackend`, `HeapKvCache`, core
matmul (dense test). It's weight loading/orientation.

## Status

| Phase | Status | Notes |
|---|---|---|
| 0. Composite build wired | ✅ done | `-PuseLocalSkainet=true` builds tinyllama vs local TF source (commit 969cf73). |
| 1. Parity harness | ✅ done | `parity_ref.py` (uv) + `parityDump` (PARITY=1). Divergence confirmed (below). |
| 2. Fix orientation + logicalShapes (Tier A) | ▶ in progress | DecoderGgufWeightLoader + LlamaNetworkLoader in TF; mirror Gemma. |
| 3. Verify coherent eager-jvm | ⬜ | bench python-baseline vs eager-jvm coherent; remove "OUTPUT NOT CORRECT" flags. |
| 4. Upstream real-GGUF Llama parity test | ⬜ | RealTinyLlamaQ4KParityTest mirroring GemmaQ5KPackedParityTest. |
| 5. Release + drop workaround (follow-on) | ⬜ | TF CHANGELOG/version bump; Tier B packed board path separate. |

## Key upstream references (Gemma template)
- `llm-inference/gemma/.../Gemma4RuntimeWeights.kt:78` (logicalShapes field)
- `llm-inference/gemma/.../Gemma4WeightLoader.kt:229,527` (reversed-shape tracking)
- `llm-inference/gemma/.../GemmaPackedWeights.kt:31` (`convertGemmaWeightsPacked`, orientation)
- `llm-inference/gemma/src/jvmTest/.../GemmaQ5KPackedParityTest.kt` (parity test template)
- Llama (to fix): `llm-inference/llama/.../DecoderGgufWeightLoader.kt:43-47`,
  `.../LlamaNetworkLoader.kt`, jvmMain `DecoderGgufMemSegConverter.kt:114-124` (double-transpose).

## Parity reference (target for the fix)
Prompt `Question: What is quantization?\n\nAnswer:`, tokens `[1,5462,303,291,29901,1724,338,4323,2133,29973,13,13,22550,29901]`:
- **llama.cpp** top next token: `' Quant'` (logprob −0.11), then ` `, `\n`, ` In`, ` A`, ` The` — coherent.
- **SKaiNET** (LlamaRuntime, same tokens) top: `'lst'`(8.66), `ṛ`, ` Shaw`, `வ`, `selected` — garbage.
- => identical input, wrong logits → forward/weight-loading bug. Fix target: SKaiNET top-1 == `' Quant'`.

## Log
- 2026-06-23: composite build wired + verified; root cause localized (untested Llama path);
  plan approved; progress log created.
- 2026-06-23: Phase 1 done — parity harness (`benchmarks/python/parity_ref.py` + `PARITY=1`
  `parityDump`). llama.cpp top=`' Quant'`; SKaiNET top=`'lst'` for identical tokens. Reference
  captured. Starting Phase 2 (upstream orientation + logicalShapes fix).
