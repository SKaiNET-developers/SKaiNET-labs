package com.dtag.skainet.tinyllama

import java.io.File
import kotlin.random.Random
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
 * NOTE: currently uses a small synthetic graph ([tinyLlamaSmokeWeights]) to validate
 * the DSL -> DAG -> StableHLO path. Phase C swaps this for real GGUF weights and the
 * full 22-layer architecture.
 */
fun exportStableHlo(out: String): ExportSummary {
    val ctx = DirectCpuExecutionContext()
    val weights = tinyLlamaSmokeWeights(ctx)
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

    val unsupported = mlir.lines().count { it.contains("Unsupported", ignoreCase = true) }
    val summary = ExportSummary(outFile.absolutePath, graph.nodes.size, graph.edges.size, mlir.lines().size, unsupported)
    println("WROTE_MLIR ${summary.outFile}")
    println("Graph: ${summary.nodes} nodes, ${summary.edges} edges")
    println("MLIR: ${summary.mlirLines} lines, unsupported markers=${summary.unsupported}")
    if (unsupported > 0) {
        println("Note: this records the current compiler gap count for the Llama-shaped graph.")
    }
    return summary
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
