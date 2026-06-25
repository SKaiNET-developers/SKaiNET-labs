# B1 IREE compile blocker — iree-compile crash in constant folding (seqLen > 1)

**Date:** 2026-06-24 · **Status: ✅ RESOLVED 2026-06-25** — root-caused to non-traceable
interleaved RoPE freezing Q/K as constants; fixed in SKaiNET-transformers `RoPE.kt`. The real
TinyLlama graph now compiles to an aarch64 `.vmfb` at seq=2 and seq=8.

## ✅ RESOLUTION (2026-06-25)
**Fix:** `transformer-core/.../RoPE.kt` — `applyRoPEInterleaved` dropped to raw float arrays
(`input.data.copyToFloatArray()` → scalar rotate → `ctx.fromFloatArray`), which under graph
tracing records the rotated Q/K as a **disconnected constant**, severing them from the projection
weights. Post-GQA-broadcast that lowered to the `insert_slice`-into-`tensor.empty()` const cascade
that crashed `iree-dispatch-creation-convert-tensor-to-flow`. Added `applyRoPEInterleavedOps` — a
pure-tensor-op interleaved rotation (reshape `[headDim]→[halfRotary,2]`, narrow even/odd, rotate
with cos/sin tables, re-interleave), numerically identical to the raw path. Gated on
`input.ops is KspTensorOps` (tracing only; eager uses real CPU ops → raw fast path unchanged).
**Verified:** seq=2 & seq=8 → aarch64 `.vmfb` (exit 0); eager-jvm still coherent ("Quantization is
the process of converting a digital signal…", 3.24 tok/s, matches llama.cpp); `LlamaDslPipelineTest`
(traces Llama) green. Export now: 2171 nodes, 1 raw output (was 45), 289 ext params (+88 = cos/sin
tables now recorded as ops not frozen).

### ✅ LOGIT PARITY CONFIRMED (2026-06-25) — bit-identical to eager
Ran the compiled vmfb (IREE 3.11, host llvm-cpu) with the FP32 `.irpa` on a fixed 8-token prefill
`[1,5462,303,291,29901,1724,338,4323]` and compared top-10 last-position logits to eager
`fromGguf(DEQUANTIZE_TO_FP32) + OptimizedLLMRuntime` (the path that matches llama.cpp). **All 10 token
ids match in order, logits identical to 3 decimals** (id 23378=13.098, 29874=12.313, 2164=11.654, …,
28677=9.583). So the exported+compiled graph is numerically correct, not just compilable. Repro:
`scripts/parity-iree-eager.sh`. Tooling: `iree-run-module --parameters=model=…irpa` (gemma-fc-iree);
eager via `PARITY_CANON=1 PARITY_TOKENS=… -Pxmx=12g` (dense FP32 needs ~4.4 GB heap).

### ✅ B2 — int8-quantized `.irpa` fits the board (2026-06-25)
Weight-only int8 post-quant (`scripts/quantize-irpa.py`): the 156 large rank-2 matmul weights →
i8 globals + per-row F32 scales, with a dequant (`convert` + `broadcast_in_dim` scale + `multiply`)
spliced at each `util.global.load` site; the 133 small tensors (RoPE tables, RMSNorm gains) stay F32.
**`.irpa` 4.2 GB → 1.1 GB** (fits 1.96 GB board); int8 graph compiles host + aarch64 (~222 KB vmfb).
**Parity (same 8-token prefill vs FP32): top-1/2/3 ids+logits exact, all top-10 ids preserved, logits
within ~0.16, only sub-0.04-logit ties reorder.** Pipeline: export → `quantize-irpa.py` →
`iree-convert-parameters` int8 safetensors → `.irpa` → `iree-compile`.

### ✅ Decode loop — built + host-validated; board script ready (2026-06-25)
Greedy decode over the fixed-seq prefill graph (no KV cache): re-run the prefill on the growing
token window padded to L, argmax the last real position (causal mask ⇒ pos k depends only on
tokens 0..k, so right-padding is safe), append, repeat.
- **Host** (`scripts/decode-iree.py`, `iree.runtime` — loads vmfb + `.irpa` ONCE, loops in-process):
  prompt `[1,5462,303,291]` → generates `29901,1724,338,278` ("Question: What is the") — coherent.
  **int8 and FP32 generate the IDENTICAL sequence**, and FP32==eager (bit-identical, B1) ⇒ int8
  decode == eager greedy == llama.cpp-quality.
- **Board** (`scripts/decode-board.sh`, adb-orchestrated): pushes vmfb + 1.1 GB `.irpa` to `/home`
  (not tmpfs), loops `iree-run-module --device=local-task://` per step, host does argmax. Correct
  but slow (params reload + full L-forward per token on the 2-core A55); KV-cache decode is the perf
  follow-up. **Ready to run on a board-connected machine** — needs board `iree-run-module` **3.11**.

### Tools updated to IREE 3.11 ("update to latest 3.11 tools all")
`docker/iree-compile.Dockerfile` 3.10→3.11; `scripts/*` use `skainet-iree:3.11.0`. **The board
runtime must also be refreshed to 3.11** — a 3.11 vmfb needs module feature `[Ch]`, which the board's
old 3.10 `iree-run-module` lacked; updating the board runtime to 3.11 satisfies it.

---
### Original investigation (kept for the record)
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

## Tried: output pruning — DID NOT fix the crash (2026-06-24)
Implemented action #1: `ComputeGraph.prunedToOutputs(logits)` in `LlamaIreeExport` (via
`skainet-compile-opt`). Export confirms it works — **45 outputs → 1 (logits only)**, nodes
**1467 → 1247** (drops the 44 dangling per-layer q/k subgraphs; ext params 245 → 201, 3.8 GB).
But `iree-compile --compile-to=flow` on the pruned graph **crashes identically**: same
`mlir::ElementsAttr::getType()` SIGSEGV in `Operation::fold` under greedy canonicalize. So the
dead subgraphs were **not** the trigger; the fold bug lives in the surviving multi-position
attention path that feeds logits. Pruning is kept anyway (correct: decode only reads logits).
→ Move to action #2/#3.

## seqLen bisect — threshold is EXACTLY seq=1 (OK) → seq=2 (crash) (2026-06-24)
Ran `scripts/bisect-seqlen.sh` (export the **real** graph at each seqLen, `iree-compile
--compile-to=flow`, host-target since `flow` is target-independent). **First time the real graph
compiled at any seqLen:**

| seqLen | nodes | ext params | --compile-to=flow |
|---|---|---|---|
| **1** | 565  | 135 | ✅ **exit 0 (compiles)** |
| 2 | 1247 | 201 | ❌ exit 245 (crash) |
| 3 | 1247 | 201 | ❌ crash |
| 4 | 1247 | 201 | ❌ crash |

seq≥2 are structurally identical (1247 nodes; only constant *shapes* scale with seq), so **seq=2
is the minimal crashing real graph**. Per-op census diff seq1→seq2 — the ops that appear ONLY at
seq≥2 (i.e. that fold away at single-position) are the entire **multi-position attention block**:
- `compare`(22) + `select`(22) + `subtract`(22) + `maximum`(22) → causal mask (iota≥iota → select)
  + stable-softmax (sub max / max-reduce). Trivial/folded at 1×1, live at 2×2.
- `iota`(44), `concatenate`(44), `slice`(176) → RoPE (position iota, rotate-half concat/split).
- `constant` jumps 91 → 267; the crash is in constant *folding*, so the suspect is a fold pattern
  on one of these new multi-position constructs (causal-mask select / softmax-max / RoPE concat).

This matches the earlier "ruled out in isolation" findings: causal-mask alone and select(-inf) alone
both compile → it's an **interaction** among mask + softmax + RoPE that only exists at seqLen ≥ 2.
MLIRs kept: `build/iree/bisect/seq{1,2,3,4}.mlir`.

## ROOT CAUSE — exact pass + mechanism (2026-06-24)
`iree-reduce` isn't shipped in any local image (dev tool); instead used MLIR's crash reproducer
(`--mlir-pass-pipeline-local-reproducer --mlir-pass-pipeline-crash-reproducer=…`). It names the
crashing pass authoritatively and emits a self-contained module:

- **Crashing pass:** `iree-dispatch-creation-convert-tensor-to-flow`, running inside a
  **`util.initializer`** (global-initializer, not the main func). Repro: `build/iree/bisect/crash-local.mlir`.
- **The IR it dies on** (a hoisted global initializer):
  ```mlir
  %0 = flow.tensor.constant #flow.parameter.named<"model"::"t27"> : tensor<4x2x64xf32>  // RoPE table, FROZEN as a const param
  %s = tensor.extract_slice %0[0,0,0][1,2,64][1,1,1] : tensor<4x2x64xf32> to tensor<2x64xf32>
  %e = tensor.empty() : tensor<32x2x64xf32>
  %i = tensor.insert_slice %s into %e[0,0,0][1,2,64][1,1,1] : tensor<2x64xf32> into tensor<32x2x64xf32>
  ```
  `convert-tensor-to-flow` tries to **const-fold** `insert_slice(extract_slice(const), tensor.empty())`
  and null-derefs in `ElementsAttr::getType()` (the `tensor.empty` has no `ElementsAttr`).
- **Scale:** the seq=2 graph has **1408** `insert_slice 2x64 → 32x2x64` + **528** `extract_slice
  4x2x64 → 2x64`. That's the **GQA head expansion (4 kv-heads → 32 q-heads)** lowered as a mountain
  of per-head slice/insert into `tensor.empty()`, seeded by RoPE tables that the export **froze into
  constant params** (`t27 = 4x2x64`, seq baked into dim 1).

### Two distinct defects
1. **IREE compiler bug** — `convert-tensor-to-flow` should not segfault folding `insert_slice` into
   a `tensor.empty()`. Fileable. (Minimal hand-repro not yet isolated: a *single* const slice/insert
   into empty compiles — it needs the fuller cascade; `crash-local.mlir` is the working repro.)
2. **Export pathology = the real unblock lever.** Two things our export does wrong:
   (a) freezes the **RoPE rotation tables as constant external params** (they're position functions,
   not weights — matches the `FREEZE_DIAG` "activation-shaped constants carrying the seq dim"); and
   (b) emits **GQA head broadcast as 1408 slice/insert-into-empty** ops instead of a clean broadcast.
   **Decisive evidence it's the const-ness:** `build/iree/bisect/mini.mlir` — the identical
   `extract_slice/insert_slice/empty` pattern with a **runtime `%arg0` source compiles fine**; only a
   **constant** source triggers the fold → crash. So if RoPE tables were runtime-computed ops (or
   non-const inputs) rather than frozen const params, IREE wouldn't fold the cascade and wouldn't crash.

### Recommended direction
- **Unblock (export side):** stop materializing RoPE cos/sin as constant params — keep them as
  traced ops (iota→sin/cos), and/or express GQA head expansion as a single broadcast. Likely lands in
  SKaiNET-transformers RoPE/`repeatKVHeads` under `VoidTensorOps` tracing, or in how `LlamaIreeExport`
  sets `embedConstants`/materialization for seq-shaped tensors. This is real work, not a flag.
- **Also file IREE issue** with `crash-local.mlir` (compiler robustness — fold should bail, not segfault).
- Tried & failed as quick workarounds: output pruning (above); `--iree-opt-const-eval=false`,
  `--iree-opt-const-expr-hoisting=false`, `--iree-opt-numeric-precision-reduction=false` (all still crash).

## Next actions (ranked)
1. ~~Prune graph outputs to just logits.~~ **DONE, did not help** (see above).
2. ~~Bisect up seqLen for a minimal real repro.~~ **DONE → seq=2 is minimal**; root-caused to
   `convert-tensor-to-flow` folding RoPE/GQA `insert_slice`-into-`empty` (see ROOT CAUSE above).
3. **[NOW THE LEAD] Export fix: don't freeze RoPE tables as constant params.** Keep RoPE cos/sin as
   traced ops and/or emit GQA head-expansion as a broadcast, so IREE has no constant slice-cascade to
   fold. Evidence (`mini.mlir`) says a non-const source compiles. SKaiNET-transformers RoPE/`repeatKVHeads`
   and/or `LlamaIreeExport` materialization policy.
4. **Try the Torq-fork iree-compile** (g165e12a) — required for the board anyway; may lower this differently.
5. **File IREE issue** with `build/iree/bisect/crash-local.mlir` (segfault in fold is a compiler bug).
4. **File an IREE issue** with the reduced repro (null-deref in `ElementsAttr::getType` during
   greedy fold is a compiler bug regardless of our IR).

## Repro
```
./gradlew -PuseLocalSkainet=true :export-hlo:exportLlamaIree -Pseq=8   # MLIR + weights
docker run --rm -m18g -v "$PWD":/work -w /work --entrypoint sh skainet-iree-compile:3.10.0 -c '
  iree-convert-parameters --parameters=build/iree/tinyllama_weights.safetensors --output=build/iree/tinyllama.irpa
  iree-compile build/iree/tinyllama_iree.mlir --compile-to=flow -o /tmp/b.mlir --mlir-disable-threading'  # crashes
```
