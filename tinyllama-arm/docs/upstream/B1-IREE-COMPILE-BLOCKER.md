# B1 IREE compile blocker — iree-compile crash in constant folding (seqLen > 1)

**Date:** 2026-06-24 · **Status:** export + .irpa work; `iree-compile` crashes on the real graph.

## What works
- `:export-hlo:exportLlamaIree` → value-correct StableHLO MLIR (245 weights as `util.global`
  external params, func takes only the token input) + weights safetensors. 1467 nodes, **0
  unsupported**, full attention wired (op census in PERF-LOGBOOK `b1-iree-export`).
- `iree-convert-parameters --parameters=...safetensors --output=tinyllama.irpa` → 4.4 GB `.irpa`. ✅
- Synthetic 1-layer smoke graph (`exportStableHlo`, **seqLen=1**) → `iree-compile` → **24 KB vmfb, exit 0**. ✅

## The crash
`iree-compile build/iree/tinyllama_iree.mlir` (the real seqLen=8 graph) → **exit 245, no diagnostic**.
With `--mlir-disable-threading` the backtrace surfaces:
```
mlir::ElementsAttr::getType() const + 4          <-- SIGSEGV (null deref)
... mlir::Operation::fold(...)
... mlir::applyPatternsGreedily(...)             <-- greedy canonicalize/fold
ireeCompilerInvocationPipeline
```
- Stage isolation: `--compile-to=input` succeeds (parse OK, 882 KB normalized IR);
  `--compile-to=flow` crashes. So it's an **input→flow canonicalization fold**, not parsing.
- Reproduces identically on IREE **3.7.0** (skainet-iree:local), **3.10.0**
  (skainet-iree-compile:3.10.0, board bytecode target), and **3.11.0** (gemma-fc-iree). Version-independent.

## What's been ruled out
- **Not the external-param construct**: a minimal `util.global #flow.parameter.named` + dot_general compiles fine.
- **Not the causal-mask pattern alone**: a minimal `iota GE iota → select(pred, 0.0, X) → add` graph
  compiles with **both** `X = -inf (0xFF800000)` and `X = -1e30`. So neither -inf nor the select
  pattern is the trigger in isolation.
- **Correlates with seqLen > 1**: smoke seqLen=1 compiles; the real seqLen=8 graph (which adds the
  `tensor<1x32x8x8xf32>` attention-score constants + causal mask + softmax reduce) crashes. The
  trigger is an **interaction in the fuller multi-position attention subgraph**, not any single op.

## Observations / leads
- The exported graph returns **dangling per-layer intermediates** (22× `tensor<32x8x64xf32>` +
  22× `tensor<4x8x64xf32>`, the post-RoPE q/k) alongside the final `tensor<1x8x32000xf32>` logits.
  These create dead-ish subgraphs; the greedy folder + DCE interplay on them is a prime suspect.
  `toComputeGraph` has no explicit-outputs param (only synthesizeExternalInputs / inputTensorIds /
  embedConstants), so pruning to logits-only needs ComputeGraph post-processing or converter work.

## Next actions (ranked)
1. **Prune graph outputs to just logits** (in `LlamaIreeExport`/converter). Removes the dead
   subgraphs; most likely to dodge the fold crash and is correct anyway (decode only reads logits).
2. **Bisect up** from the smoke graph: grow seqLen 1→2, add layers, until the crash appears, to get
   a minimal real repro, then reduce with `iree-reduce`.
3. **Try the Torq-fork iree-compile** the board/gemma path used (g165e12a) — gemma-iree notes it's
   required for the board anyway; it may also sidestep this fold.
4. **File an IREE issue** with the reduced repro (null-deref in `ElementsAttr::getType` during
   greedy fold is a compiler bug regardless of our IR).

## Repro
```
./gradlew -PuseLocalSkainet=true :export-hlo:exportLlamaIree -Pseq=8   # MLIR + weights
docker run --rm -m18g -v "$PWD":/work -w /work --entrypoint sh skainet-iree-compile:3.10.0 -c '
  iree-convert-parameters --parameters=build/iree/tinyllama_weights.safetensors --output=build/iree/tinyllama.irpa
  iree-compile build/iree/tinyllama_iree.mlir --compile-to=flow -o /tmp/b.mlir --mlir-disable-threading'  # crashes
```
