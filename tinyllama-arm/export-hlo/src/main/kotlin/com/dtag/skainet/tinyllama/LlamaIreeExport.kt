package com.dtag.skainet.tinyllama

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.ConstantMaterializationPolicy
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.graph.DefaultExecutionTape
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.tape.Execution

/** Result of a Llama IREE export (MLIR + external-parameter weights). */
data class IreeExportSummary(
    val mlirFile: String,
    val weightsFile: String,
    val nodes: Int,
    val externalParams: Int,
    val weightBytes: Long,
)

/**
 * Export TinyLlama to a self-contained IREE artifact pair: a StableHLO MLIR whose weights are
 * lifted to `util.global` external parameters, plus a flat safetensors holding the real weight
 * bytes. `iree-compile --iree-opt-import-parameters` + `iree-run-module --parameters=model=...`
 * then run it with only the token input — no per-weight argument plumbing.
 *
 * Two deliberate choices, both mirroring the proven Gemma bake (RealGemmaBakeIrpaTest):
 *  - **Trace the known-correct model.** We build via `fromGguf(DEQUANTIZE_TO_FP32).load()` — the
 *    same topology + weight orientation as the coherent eager path — NOT the `fromWeights`
 *    densify route (still numerically buggy for Llama, see runEagerJvmDsl). embedConstants=true
 *    then resolves those correct bound weights into the graph.
 *  - **Fixed-seq prefill, no KV cache.** A single pass over `seqLen` positions computes K/V fresh
 *    and stays traceable; KVCache.update() does a non-traceable copyToFloatArray. The decode loop
 *    pads tokens to `seqLen` and reads the argmax at the last real position (gemma-iree design).
 *
 * Weights are emitted as safetensors (1-D F32, size-equivalent, keyed by tensor name) — the
 * working bridge to `.irpa` via `iree-convert-parameters`; SKaiNET's IrpaWriter header is not
 * IREE-v0 compatible yet. The `util.global` carries the real shape, so 1-D storage is fine.
 */
fun exportLlamaIree(
    mlirOut: String,
    weightsOut: String,
    modelPath: String,
    seqLen: Int = 32,
): IreeExportSummary {
    val ctx = DirectCpuExecutionContext()
    println("Llama IREE export: $modelPath (seqLen=$seqLen)")
    println("Building known-correct model via fromGguf(DEQUANTIZE_TO_FP32).load() ...")
    val model = runBlocking {
        LlamaNetworkLoader.fromGguf(
            randomAccessProvider = {
                createRandomAccessSource(modelPath) ?: error("no RandomAccessSource: $modelPath")
            },
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
        ).load<FP32, Float>(ctx)
    }

    // Strip KV caches so the prefill pass stays fully traceable (see kdoc).
    stripKvCache(model)

    // Token input [1, seqLen] under VoidTensorOps: record graph structure only; the bound
    // weights resolve at finalize (embedConstants=true), so no GBs of matmul actually run.
    val input = VoidOpsTensor(
        object : TensorData<FP32, Float> {
            override val shape = Shape(1, seqLen)
            override fun get(vararg indices: Int): Float = 0f
            override fun set(vararg indices: Int, value: Float) {}
        },
        FP32::class,
    )
    val tapeCtx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val tape = tapeCtx.record {
        val ct = (this as DefaultGraphExecutionContext).currentTape ?: error("no tape")
        Execution.tapeStack.pushTape(ct)
        try {
            model.forward(input, this as ExecutionContext)
        } finally {
            Execution.tapeStack.popTape()
        }
    }.first

    val graph = (tape as DefaultExecutionTape).toComputeGraph(
        synthesizeExternalInputs = true,
        embedConstants = true,
    )
    val module = StableHloConverterFactory
        .createBasic(ConstantMaterializationPolicy.ExternalAlways(scope = "model"))
        .convert(graph, "tinyllama")

    File(mlirOut).apply { parentFile?.mkdirs() }.writeText(module.content)

    val ext = module.externalParameters
    val totalBytes = ext.sumOf { it.source.sizeInBytes }
    val globals = module.content.lineSequence().count { it.trimStart().startsWith("util.global ") }
    val funcArgs = module.content.lineSequence().firstOrNull { it.contains("func.func @tinyllama(") }
        ?.let { Regex("%arg\\d+").findAll(it).count() } ?: -1
    println("Graph: ${graph.nodes.size} nodes")
    println("EXTPARAMS ${ext.size} totalMiB=${totalBytes / (1024 * 1024)} UTILGLOBALS $globals FUNCARGS $funcArgs")

    writeSafetensors(weightsOut, ext)

    println("WROTE_MLIR ${File(mlirOut).absolutePath}")
    println("WROTE_WEIGHTS ${File(weightsOut).absolutePath} (sizeMiB=${File(weightsOut).length() / (1024 * 1024)})")
    println("Next: iree-convert-parameters --parameters=$weightsOut --output=<model.irpa>, then iree-compile.")
    return IreeExportSummary(File(mlirOut).absolutePath, File(weightsOut).absolutePath, graph.nodes.size, ext.size, totalBytes)
}

/** Recursively detach KV caches (non-traceable under VoidTensorOps). */
private fun stripKvCache(m: Module<*, *>) {
    if (m is MultiHeadAttention<*, *>) m.kvCache = null
    m.modules.forEach { stripKvCache(it) }
}

/**
 * Flat safetensors: each external param as 1-D F32, size-equivalent, keyed by tensor name.
 * Header is a JSON map of name -> {dtype, shape:[count], data_offsets:[start,end]} followed by
 * the raw little-endian float bytes, in the same order. Consumed by `iree-convert-parameters`.
 */
private fun writeSafetensors(path: String, ext: List<ExternalParameterRef>) {
    var off = 0L
    val hdr = StringBuilder("{")
    ext.forEachIndexed { i, e ->
        val len = e.source.sizeInBytes
        if (i > 0) hdr.append(",")
        hdr.append("\"${e.key}\":{\"dtype\":\"F32\",\"shape\":[${len / 4}],\"data_offsets\":[$off,${off + len}]}")
        off += len
    }
    hdr.append("}")
    val headerBytes = hdr.toString().encodeToByteArray()
    val f = File(path).apply { parentFile?.mkdirs() }
    BufferedOutputStream(FileOutputStream(f), 1 shl 20).use { os ->
        os.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.size.toLong()).array())
        os.write(headerBytes)
        for (e in ext) {
            val src = e.source
            require(src is BufferHandle.Owned) {
                "expected materialized BufferHandle.Owned for '${e.key}', got ${src::class.simpleName}"
            }
            os.write(src.data, src.offset, src.sizeInBytes.toInt())
        }
    }
}
