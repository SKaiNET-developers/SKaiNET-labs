package sk.ainet.tinyllama

import java.io.File
import kotlin.random.Random
import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.llama.LlamaTensorNames

/** Summary of an export run, for the CLI / comparison harness. */
data class ExportSummary(
    val outFile: String,
    val nodes: Int,
    val edges: Int,
    val mlirLines: Int,
    val unsupported: Int,
)

/**
 * Export a Llama-shaped SKaiNET DSL graph to StableHLO MLIR.
 *
 * If [modelPath] is null, exports a small synthetic graph ([tinyLlamaSmokeWeights]) that
 * validates the DSL -> DAG -> StableHLO path. If [modelPath] points at a GGUF file, exports
 * the REAL TinyLlama architecture with the loaded weights, capped to [contextLength] (Phase C).
 * Weights stay as external graph inputs (embedConstants = false) to avoid baking ~GBs of
 * constants into the MLIR.
 */
fun exportStableHlo(out: String, modelPath: String? = null, contextLength: Int = 32): ExportSummary {
    val ctx = DirectCpuExecutionContext()
    val weights = if (modelPath == null) {
        println("Export: synthetic smoke graph (no --model)")
        tinyLlamaSmokeWeights(ctx)
    } else {
        println("Export: REAL TinyLlama from $modelPath (ctx cap $contextLength)")
        val loaded = runBlocking { loadPackedGgufLlamaWeights(modelPath, ctx) }
        val capped = capLlamaContext(loaded, contextLength)
        println("Tensor storage: ${summarizeTensorStorage(capped)}")
        println("Arch: dim=${capped.metadata.embeddingLength}, layers=${capped.metadata.blockCount}, vocab=${capped.metadata.vocabSize}, heads=${capped.metadata.headCount}")
        // Weights become external graph inputs (embedConstants=false), so only their shapes
        // matter for the export. Emit zero FP32 skeletons with corrected (transposed) shapes.
        println("Building FP32 weight skeletons (zeros, transposed shapes) for graph capture ...")
        if (System.getenv("EXPORT_DEBUG_SHAPES") == "1") {
            val names = listOf(
                LlamaTensorNames.TOKEN_EMBEDDINGS, LlamaTensorNames.OUTPUT_WEIGHT, LlamaTensorNames.OUTPUT_NORM,
                LlamaTensorNames.attnNorm(0), LlamaTensorNames.attnQ(0), LlamaTensorNames.attnK(0),
                LlamaTensorNames.attnV(0), LlamaTensorNames.attnOut(0), LlamaTensorNames.ffnNorm(0),
                LlamaTensorNames.ffnGate(0), LlamaTensorNames.ffnUp(0), LlamaTensorNames.ffnDown(0),
            )
            names.forEach { n -> println("  shape[$n] = ${capped.tensors[n]?.shape?.dimensions?.toList()}") }
        }
        skeletonizeWeights(capped, ctx)
    }
    val model = LlamaNetworkLoader.fromWeights(weights)
    val tapingCtx = DefaultGraphExecutionContext.tape(baseOps = ctx.ops)
    val input = tokenTensor(1)

    tapingCtx.startRecording()
    model.forward(input, tapingCtx)
    val tape = tapingCtx.stopRecording() as DefaultExecutionTape
    val graph = tape.toComputeGraph(
        synthesizeExternalInputs = true,
        inputTensorIds = emptySet(),
        embedConstants = false,
    )

    val mlir = toStableHlo(graph, "tinyllama_step").content
    val outFile = File(out)
    outFile.parentFile?.mkdirs()
    outFile.writeText(mlir)

    val unsupportedLines = mlir.lines().filter { it.contains("Unsupported", ignoreCase = true) }
    val summary = ExportSummary(outFile.absolutePath, graph.nodes.size, graph.edges.size, mlir.lines().size, unsupportedLines.size)
    println("WROTE_MLIR ${summary.outFile}")
    println("Graph: ${summary.nodes} nodes, ${summary.edges} edges")
    println("MLIR: ${summary.mlirLines} lines, unsupported markers=${summary.unsupported}")
    if (unsupportedLines.isNotEmpty()) {
        println("UNSUPPORTED OPS (first 20 of ${unsupportedLines.size}):")
        unsupportedLines.take(20).forEach { println("  ${it.trim()}") }
    }
    return summary
}

/**
 * Build dense FP32 weight *skeletons* for graph capture.
 *
 * The export uses `embedConstants = false`, so weights become external graph inputs — their
 * VALUES never appear in the MLIR, only their shapes. So we don't dequantize the packed GGUF
 * (cheap + no multi-GB heap); we emit zero tensors with the shapes the generic SKaiNET
 * transformer layers expect.
 *
 * GGUF projection weights load as `[in, out]` (e.g. attn_k `[2048, 256]`, ffn_gate
 * `[2048, 5632]`), but the generic `MultiHeadAttention`/FFN `linearProject` computes
 * `x @ w.t()` and expects `[out, in]`; the token embedding loads `[dim, vocab]` but the
 * `Embedding` layer needs `[vocab, dim]`. Both are just a dimension swap, so we transpose
 * the shape of every rank-2 weight. (The native LlamaRuntime adapts to `[in, out]` itself,
 * which is why it doesn't need this.)
 */
private fun skeletonizeWeights(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: DirectCpuExecutionContext,
): DecoderGgufWeights<FP32, Float> {
    val out = linkedMapOf<String, Tensor<FP32, Float>>()
    for ((name, t) in weights.tensors) {
        val dims = t.shape.dimensions
        val shape = if (dims.size == 2) Shape(dims[1], dims[0]) else t.shape
        out[name] = ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume))
    }
    return weights.copy(tensors = out)
}

private fun tinyLlamaSmokeWeights(ctx: DirectCpuExecutionContext): DecoderGgufWeights<FP32, Float> {
    val dim = 8
    val ffDim = 16
    val vocabSize = 16
    val nHeads = 2
    val kvHeads = 2
    val headDim = dim / nHeads
    val seqLen = 32
    val metadata = LlamaModelMetadata(
        architecture = "llama",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        feedForwardLength = ffDim,
        ropeDimensionCount = headDim,
        vocabSize = vocabSize,
    )
    val tensors = linkedMapOf(
        LlamaTensorNames.TOKEN_EMBEDDINGS to randn(ctx, Shape(vocabSize, dim), 10),
        LlamaTensorNames.OUTPUT_NORM to ones(ctx, Shape(dim)),
        LlamaTensorNames.OUTPUT_WEIGHT to randn(ctx, Shape(vocabSize, dim), 11),
        LlamaTensorNames.attnNorm(0) to ones(ctx, Shape(dim)),
        LlamaTensorNames.attnQ(0) to randn(ctx, Shape(dim, dim), 1),
        LlamaTensorNames.attnK(0) to randn(ctx, Shape(dim, dim), 2),
        LlamaTensorNames.attnV(0) to randn(ctx, Shape(dim, dim), 3),
        LlamaTensorNames.attnOut(0) to randn(ctx, Shape(dim, dim), 4),
        LlamaTensorNames.ffnNorm(0) to ones(ctx, Shape(dim)),
        LlamaTensorNames.ffnGate(0) to randn(ctx, Shape(ffDim, dim), 5),
        LlamaTensorNames.ffnDown(0) to randn(ctx, Shape(dim, ffDim), 6),
        LlamaTensorNames.ffnUp(0) to randn(ctx, Shape(ffDim, dim), 7),
    )
    return DecoderGgufWeights(metadata, tensors)
}

private fun randn(ctx: DirectCpuExecutionContext, shape: Shape, seed: Int): Tensor<FP32, Float> {
    val rng = Random(seed)
    val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
    return ctx.fromFloatArray(shape, FP32::class, values)
}

private fun ones(ctx: DirectCpuExecutionContext, shape: Shape): Tensor<FP32, Float> =
    ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume) { 1.0f })

private fun tokenTensor(tokenId: Int): Tensor<FP32, Float> {
    val shape = Shape(intArrayOf(1))
    val data = DenseFloatArrayTensorData<FP32>(shape, floatArrayOf(tokenId.toFloat()))
    return VoidOpsTensor(data = data, dtype = FP32::class)
}
