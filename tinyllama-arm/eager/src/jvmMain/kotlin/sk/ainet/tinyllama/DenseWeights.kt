package sk.ainet.tinyllama

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaTensorNames

/**
 * Dequantize + reorient GGUF weights so `LlamaNetworkLoader.fromWeights`' dense `[out, in]`
 * DSL params accept them. The generic transformer layers do `x @ w.t()` (expect `[out, in]`)
 * and can't gather packed storage; GGUF loads packed `[in, out]` (embedding `[dim, vocab]`).
 * The native `LlamaRuntime` adapts to packed `[in, out]` itself, so it doesn't need this.
 *
 * HOST-ONLY workaround (~4.4 GB FP32) until the upstream fix lands — see
 * docs/upstream/ISSUE-packed-gguf-eager-and-export.md. The packed native board path is untouched.
 */
fun densifyGgufForDsl(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: DirectCpuExecutionContext,
): DecoderGgufWeights<FP32, Float> {
    val vocab = weights.metadata.vocabSize
    val dim = weights.metadata.embeddingLength
    // Tied embedding/output: stored vocab-major, so the [dim,vocab] label is just wrong —
    // relabel to [vocab,dim] (no data move). Other rank-2 weights are W^T ([in,out]) and need
    // a real data transpose to W ([out,in]).
    val tied = setOf(LlamaTensorNames.TOKEN_EMBEDDINGS, LlamaTensorNames.OUTPUT_WEIGHT)
    val out = linkedMapOf<String, Tensor<FP32, Float>>()
    for ((name, t) in weights.tensors) {
        val floats = when (val d = t.data) {
            is FloatArrayTensorData<*> -> d.copyToFloatArray()
            is PackedBlockStorage -> d.toFloatArray()
            else -> d.copyToFloatArray()
        }
        if (System.getenv("DEQUANT_DEBUG") == "1" &&
            (name == LlamaTensorNames.TOKEN_EMBEDDINGS || name == LlamaTensorNames.attnQ(0) || name == LlamaTensorNames.ffnGate(0))
        ) {
            var mn = Float.MAX_VALUE; var mx = -Float.MAX_VALUE; var nan = 0
            for (f in floats) { if (f.isNaN()) nan++ else { if (f < mn) mn = f; if (f > mx) mx = f } }
            System.err.println("[dequant] $name shape=${t.shape.dimensions.toList()} n=${floats.size} min=$mn max=$mx nan=$nan first=${floats.take(5)}")
        }
        val dims = t.shape.dimensions
        out[name] = when {
            name in tied -> ctx.fromFloatArray(Shape(vocab, dim), FP32::class, floats)
            dims.size == 2 -> ctx.fromFloatArray(Shape(dims[1], dims[0]), FP32::class, transpose2d(floats, dims[0], dims[1]))
            else -> ctx.fromFloatArray(t.shape, FP32::class, floats)
        }
    }
    return weights.copy(tensors = out)
}

/**
 * Dequantize packed weights to dense FP32 **keeping the GGUF layout** (no transpose, no
 * reshape) — for the LlamaRuntime path, which adapts to `[in,out]` itself. Leaves the token
 * embedding / output weight packed so `tokenEmbeddingForRuntime` still wraps them correctly.
 * Used to test whether the packed matmul kernels (core) are the source of wrong logits.
 */
fun dequantizeNonEmbedding(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: DirectCpuExecutionContext,
): DecoderGgufWeights<FP32, Float> {
    val keepPacked = setOf(LlamaTensorNames.TOKEN_EMBEDDINGS, LlamaTensorNames.OUTPUT_WEIGHT)
    val out = linkedMapOf<String, Tensor<FP32, Float>>()
    for ((name, t) in weights.tensors) {
        out[name] = when (val d = t.data) {
            is PackedBlockStorage -> if (name in keepPacked) t
                else ctx.fromFloatArray(t.shape, FP32::class, d.toFloatArray())
            else -> t
        }
    }
    return weights.copy(tensors = out)
}

/** Transpose a [rows] x [cols] row-major matrix to [cols] x [rows] row-major. */
private fun transpose2d(src: FloatArray, rows: Int, cols: Int): FloatArray {
    val dst = FloatArray(src.size)
    for (i in 0 until rows) {
        val base = i * cols
        for (j in 0 until cols) dst[j * rows + i] = src[base + j]
    }
    return dst
}
