# Memory layout & relayout: SKaiNET vs llama.cpp (ggml) vs ZML

Analysis prompted by the board OOM ([[board-oom-load]]): the NATIVE_OPTIMIZED Llama load peaks ~3.2 GB
for a 668 MB model, and the row-major→block-major **relayout is a physical copy**. Question from the
user: *can SKaiNET already do what llama.cpp does with its built-in tensor data-layout capabilities —
or do we need to go deeper (ZML-style, buffers as first-class architecture)?*

## TL;DR

SKaiNET has the **right bones** — a clean `Tensor` (logical) / `TensorData` (storage) / ops separation,
zero-copy **views** (`TensorView`/`IndexMapper`), a **buffer-handle + mmap substrate**
(`BufferHandle`, `MemoryChunk`, `TensorStorage` with `Placement`), and even a compile/IREE lowering
path. But three things are **missing or unwired**, and they are exactly what forces the relayout copy
and the heap-resident weights:

1. **No layout/stride descriptor on tensors** (no ggml `nb[]`). `Shape` is dims-only; layout is implicit.
2. **Kernels are layout-fixed, not stride-parameterized** — they read a raw `ByteArray` at one hardcoded
   contiguous layout, so they cannot consume a view, a stride, or an mmap region.
3. **The zero-copy/mmap substrate is not plumbed end-to-end** (load → TensorData → kernel) and is
   **JVM-only** (no Kotlin/Native mmap).

So SKaiNET *cannot* yet match llama.cpp's "mmap + run, no copy" — not for lack of abstractions, but
because the layout abstraction (views) and the storage abstraction (buffers) are **parallel and
disconnected**, and the performance kernels ignore both. The fix is to **unify** them, not to rebuild.

## How llama.cpp (ggml) does it

`ggml_tensor` = `ne[4]` (dims) + **`nb[4]` (byte strides)** + `type` (quant block format) + `data`
(pointer, typically into an **mmap'd** file). Consequences:

- **Layout is explicit** via `nb[]`. Every op reads strides, so views / slices / permutes / transposes
  are *metadata* — zero copy. A "transpose" just swaps `ne`/`nb`; the kernel still reads correctly.
- The **block quant format IS the on-disk layout**. GGUF stores Q4_K/Q6_K superblocks row-major; ggml's
  matmul iterates *per output row over that row's blocks* — i.e. it consumes the GGUF layout **directly,
  in place**. No row-major→block-major relayout because the kernel's iteration order matches storage.
- Weights are **mmap'd**: `data` points into the file's page cache. RSS ≈ the model, file-backed,
  shared, evictable. Load is ~instant (no parse-into-heap), peak ≈ resident model (~1.2 GB for our case).

The whole trick: **one memory image, many layouts via strides, kernels read strides, storage is mmap'd.**

## What SKaiNET has today (grounded)

- **Separation (good):** `Tensor<T,V>` (shape + ops) → `TensorData<T,V>` (storage, indexed accessor)
  → ops. `skainet-lang-core/.../tensor/`.
- **Views (zero-copy, but slow path only):** `TensorView`/`SlicedTensorView` + `IndexMapper`
  (`.../tensor/SlicedTensorView.kt`, `IndexMapper.kt`) — logical coordinate remap, no data copy. BUT
  consumed only through the per-element `get()` accessor; heavy ops `CopyMaterializationStrategy`
  **materialize (copy)** a view before compute. So views are for *correctness*, not *performance*.
- **Storage / buffer substrate (real, but unwired + JVM-only):** `BufferHandle` =
  `Owned | Borrowed | Aliased | FileBacked` (`.../tensor/storage/BufferHandle.kt`); `TensorStorage`
  carries `encoding` + `Placement` (`CPU_HEAP | MMAP_WEIGHTS`); `MemoryChunk.slice()` (zero-copy
  windows); JVM `MappedRandomAccessSource`/`MmapTensorData`/`Q4KMemSegMatmulKernel`. This is a genuine
  zero-copy/mmap foundation — but `TensorStorageFactory.extractBytes()` doesn't even handle `Aliased`,
  and **none of it is on the Kotlin/Native (board) path** (`PosixPreadRandomAccessSource` = pread into a
  fresh `ByteArray`; no mmap).
- **`Shape` = `dimensions: IntArray` only — NO strides** (`.../tensor/Shape.kt`). No `nb[]` equivalent.
- **Fast kernels are layout-fixed:** e.g. `ScalarQ6_KMatmulKernel` /
  `NativeKnQ6KMatmulKernel` take `(weight: ByteArray, weightByteOffset)` and index
  `(blockIdx * outputDim + o) * BYTES_PER_BLOCK` — a hardcoded **block-major** layout, no stride
  parameters. They can't read GGUF-native row-major blocks, a view, or an mmap region directly.

### Why the relayout copy exists

GGUF stores Q4_K row-major; SKaiNET's matmul iterates *per input-block over all output rows*
(block-major) for cache locality, so `LlamaQuantLayout.relayoutKSeriesRowMajorToBlockMajor` physically
reorders the bytes. ggml avoids this only because its kernel iterates in the storage order and reads
`nb[]`. **The relayout is a missing-stride-descriptor symptom**, not a fundamental need.

## What's missing (the unification)

| capability | llama.cpp | SKaiNET today | gap |
|---|---|---|---|
| stride/layout descriptor (`nb[]`) | ✓ on every tensor | ✗ `Shape` dims-only | **add LayoutSpec** |
| views consumable by fast kernels | ✓ (kernels read `nb[]`) | views exist but materialize for compute | **stride-aware kernels** |
| quant block read in-place (no relayout) | ✓ | ✗ relayout copy | follows from strides |
| mmap'd file-backed weights | ✓ everywhere | JVM substrate only, unwired; **none on K/N** | **K/N mmap + plumb** |
| buffer as first-class (logical≠physical) | implicit (data ptr) | `BufferHandle` exists, unplumbed | **wire end-to-end** |

## Is the architecture efficient, or go deeper (ZML)?

**ZML's model:** tensors are *pure logical* (shape+dtype, no data); `Buffer` = explicit device memory;
layout/aliasing/donation handled by the **MLIR/compiler**; ops are traced into a graph the compiler
places. "Buffer as first-class" = strict logical/physical split with a lowering layer that owns memory.

**SKaiNET is already partway there** and — importantly — *already has a compile/lowering path*
(`skainet-compile`, StableHLO export, IREE). That is the ZML-equivalent, and it's the right place for
full compiler-managed buffers. **For the COMPILED path, "go deeper toward ZML" = invest in the IREE
path** (currently parked).

**For the EAGER CPU path (the live board goal), full ZML is the wrong depth** — you don't want an
MLIR compiler in the hot loop. The right depth is the **ggml model**: a layout/stride descriptor +
layout-aware kernels + mmap. That is lighter than ZML, matches llama.cpp's proven CPU efficiency, and
reuses SKaiNET's existing `BufferHandle`/`MemoryChunk` substrate.

### Recommended direction (eager CPU)

1. **LayoutSpec on TensorData/buffer** — generalize `Shape` with a layout (strides or a small enum:
   `RowMajor | BlockMajorKQuant | GgufNative`). Logical tensor = (shape, dtype, layout, BufferHandle).
2. **Make the GGUF-native block layout directly consumable** by the packed kernels (add a
   GGUF-row-major variant, or parameterize the block stride) → **eliminate the relayout copy** entirely.
   This is the single biggest win: removes the only irreducible copy AND unlocks mmap (the kernel reads
   the file layout in place).
3. **Plumb `BufferHandle`/mmap end-to-end** + add a **Kotlin/Native `mmap` `MemoryChunk`** so weights are
   file-backed on the board. Wire `Aliased` through `TensorStorageFactory.extractBytes`.
4. Result target: eager CPU at llama.cpp-class memory — mmap'd, zero-copy, ~1.2 GB, instant load — on
   the 2 GB board.

### Tactical vs strategic

- **Tactical (now):** the fused load+pack patch ([[fused-load-pack-plan]]) lowers the load *peak* so the
  board *runs* — but it still relayout-copies and keeps packed weights in heap. Worth shipping to unblock
  a first board number; it is NOT the architecture.
- **Strategic (the real fix):** items 1–3 above — the ggml-style layout-descriptor + mmap unification.
  This is the user's instinct ("SKaiNET can already do this with built-in layout capabilities") and it's
  *almost* true: the pieces exist; they need a stride descriptor and to be wired into the kernels.

## Note on measurement integrity

The fused-fix measurements to date are **invalid** — the downstream `-PuseLocalSkainet` composite
substitutes core but **not** transformers for native targets (`skainet-transformers-inference-llama`
resolves to Maven `0.32.1`). To measure any transformers-side change (fused fix, or the layout work)
**publish transformers to mavenLocal with a dev version + bump the downstream pin**, or fix the composite
KMP-native substitution. Until then, board-fit numbers can't be trusted.
