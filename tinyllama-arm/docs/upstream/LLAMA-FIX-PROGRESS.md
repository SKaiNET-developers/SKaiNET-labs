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
| 2. Fix (REVISED: our code, not upstream) | ✅ done | eager-jvm switched to canonical `fromGguf(DEQUANTIZE_TO_FP32).load + OptimizedLLMRuntime`. No upstream change. |
| 3. Verify coherent eager-jvm | ✅ done | bench: eager-jvm → "Quantization is the process of converting a digital signal…" (matches llama.cpp). 0.17 tok/s, 8 GB FP32. |
| 4. Native (host arm64) canonical path | ✅ done | Native binary's **default** is now `fromGguf(NATIVE_OPTIMIZED) packed + OptimizedLLMRuntime` → **coherent on macosArm64** ("Quantization is the process of converting a digital signal…"), 0.97 tok/s (2026-06-26). `EAGER_NATIVE_FP32=1` keeps the FP32 parity path; the bespoke collapse path is now `EAGER_NATIVE_LEGACY=1` (debug only). |
| 5. Board packed path (upstream NATIVE_OPTIMIZED) | ✅ host / ⬜ board | RESOLVED on host: `fromGguf(NATIVE_OPTIMIZED) + OptimizedLLMRuntime` runs coherently on macosArm64 (the old `gather: unsupported input rank 1` is fixed by the upstream packed-Llama path, transformers ≥0.32.0). Board-fittable (packed, low-memory). Remaining: link the `linuxArm64` binary + measure on the 2 GB board. |
| 6. Upstream real-GGUF Llama parity test (optional) | ⬜ | RealTinyLlamaQ4KParityTest; lower priority since FP32 path works. |

## Native LlamaRuntime "attention bug" — DEBUGGED on host arm64 (2026-06-25)
Repro'd the `lstlstlst…` collapse on the Apple-Silicon native binary and localized it with an
in-process probe (`PROBE=1`, since removed):
- **Single-token `forward` IS input-dependent** (distinct argmax per token) → embedding/weights/output
  are not producing constant output.
- The `CpuAttentionBackend` code is **correct** (stores K/V at `position`, attends `0..pos`, proper
  GQA/softmax/RoPE) — read line-by-line.
- Tested the `linearProject` square-matrix-orientation hypothesis (always-transpose) → made it WORSE
  (empty/shape-error), so the square wq/wo are NOT the issue.
- **Decisive:** wiring the native binary to the canonical `fromGguf(DEQUANTIZE_TO_FP32) +
  OptimizedLLMRuntime` (commonMain, runs on K/N) → **coherent output**. So the K/N runtime + Accelerate
  kernels are correct; the defect is isolated to tinyllama's custom packed stack
  (`loadPackedGgufLlamaWeights` + `mapCompactLlamaRuntimeWeights` + deprecated `LlamaRuntime`).
- **Host fix shipped + generalized (2026-06-26):** the canonical fix now works with the **packed
  `NATIVE_OPTIMIZED`** policy too (not just FP32) — `fromGguf(NATIVE_OPTIMIZED) + OptimizedLLMRuntime`
  is coherent on macosArm64 and is now the native **default** (board-fittable, no 4.4 GB FP32 blowup).
  The bespoke `loadPackedGgufLlamaWeights`/`LlamaRuntime` collapse path is demoted to
  `EAGER_NATIVE_LEGACY=1` (debug only). Board link + measure is all that remains (Phase 5).

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

## ⚡ BREAKTHROUGH (revises root cause)
The **canonical upstream path is CORRECT**, the bug was in OUR repo's custom loaders:
- `LlamaNetworkLoader.fromGguf(DEQUANTIZE_TO_FP32).load() + OptimizedLLMRuntime` → top token
  `'Quant'` (16.84) and top-10 **exactly matches llama.cpp**. ✓ CORRECT.
- The garbage came from tinyllama's own `loadPackedGgufLlamaWeights` + `mapCompactLlamaRuntimeWeights`
  + `PackedBlockTensorDataView` + `densifyGgufForDsl` + deprecated `LlamaRuntime` — NOT skainet.
- `fromGguf(NATIVE_OPTIMIZED)` (packed) → runtime error `gather: unsupported input rank 1` → the
  upstream PACKED path IS broken (board/Tier B), but the FP32 path is fine.

**Revised fix:**
- HOST eager-jvm (this repo): switch to the canonical `fromGguf(DEQUANTIZE_TO_FP32).load()` path
  → correct (~4.4 GB FP32, fine on host). Drop the custom hacks. No upstream change.
- BOARD native eager (packed, 2 GB): still needs the upstream `NATIVE_OPTIMIZED` gather fix
  (rank-1 packed embedding) — that's the real upstream item (Tier B). Our `mapCompact`/`LlamaRuntime`
  packed path is also wrong; prefer fixing upstream packed.

## Log
- 2026-06-23: composite build wired + verified; root cause localized (untested Llama path);
  plan approved; progress log created.
- 2026-06-23: Phase 1 done — parity harness. llama.cpp top=`' Quant'`; SKaiNET (our LlamaRuntime
  path) top=`'lst'`.
- 2026-06-23: BREAKTHROUGH — canonical `fromGguf(DEQUANTIZE_TO_FP32).load + OptimizedLLMRuntime`
  parity top=`'Quant'`, top-10 == llama.cpp. The bug was tinyllama's custom loaders, not SKaiNET.
  Switched eager-jvm to the canonical path → coherent output matching llama.cpp (Phases 2+3 done,
  host). Remaining: upstream packed `NATIVE_OPTIMIZED` gather (board); board's real path is IREE.
