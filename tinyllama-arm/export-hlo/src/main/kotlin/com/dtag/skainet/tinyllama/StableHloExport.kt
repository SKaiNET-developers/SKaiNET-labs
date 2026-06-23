package com.dtag.skainet.tinyllama

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
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
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
        // Dequantize packed weights to dense FP32 for graph capture. StableHLO is dense-float,
        // and ops like the embedding gather can't read packed (Byte) storage. Weights become
        // external FP32 graph inputs. Host-side only (needs a large heap), not a board path.
        println("Densifying packed weights to FP32 for graph capture ...")
        densifyWeights(capped, ctx)
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

/** Replace any packed/quantized weight tensors with dense FP32 tensors (logical shapes preserved). */
private fun densifyWeights(
    weights: DecoderGgufWeights<FP32, Float>,
    ctx: DirectCpuExecutionContext,
): DecoderGgufWeights<FP32, Float> {
    val vocab = weights.metadata.vocabSize
    val dim = weights.metadata.embeddingLength
    val dense = linkedMapOf<String, Tensor<FP32, Float>>()
    for ((name, t) in weights.tensors) {
        val floats = when (val d = t.data) {
            is FloatArrayTensorData<*> -> { dense[name] = t; continue }
            // PackedBlockStorage.toFloatArray() dequantizes block-by-block in logical order.
            // (The generic TensorData.copyToFloatArray() is broken for 2D packed tensors.)
            is PackedBlockStorage -> d.toFloatArray()
            else -> d.copyToFloatArray()
        }
        // The token embedding loads in ggml ne-order ([dim, vocab]); the Embedding layer
        // gathers rows of length dim, so it must be [vocab, dim]. Its data is already in
        // [vocab, dim] row-major order, so only the shape label changes. Other weights keep
        // their loaded shape — the runtime's linearProject transposes them as needed.
        val shape = if (name == LlamaTensorNames.TOKEN_EMBEDDINGS) Shape(vocab, dim) else t.shape
        dense[name] = ctx.fromFloatArray(shape, FP32::class, floats)
    }
    return weights.copy(tensors = dense)
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
