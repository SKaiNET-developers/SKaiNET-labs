package com.dtag.skainet.tinyllama

import java.io.File
import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random
import kotlin.time.measureTime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.apps.llm.generate
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.llama.LlamaRuntime

/**
 * Run TinyLlama eager on the JVM via the compact `LlamaRuntime` path (same as the native board
 * path: `mapCompactLlamaRuntimeWeights` + `CpuAttentionBackend`, packed weights). This is the
 * known-correct route; the `fromWeights`/`OptimizedLLMRuntime` route can't consume packed GGUF
 * (see [runEagerJvmDsl] and docs/upstream/ISSUE-packed-gguf-eager-and-export.md).
 */
suspend fun runEagerJvm(options: EagerOptions): BenchmarkResult {
    val modelPath = Path.of(resolveTinyLlamaModelPath(options.model)).toAbsolutePath().normalize()
    require(Files.exists(modelPath)) { "Model not found: $modelPath. Run scripts/download-models.sh first." }

    val prompt = options.prompt ?: readPrompt(options.promptFile, options.promptIndex)
    val formattedPrompt = formatQuestionPrompt(prompt)
    val ctx = DirectCpuExecutionContext()

    println("TinyLlama SKaiNET eager benchmark (JVM, LlamaRuntime + packed GGUF)")
    println("Model: $modelPath")
    println("Prompt: $prompt")
    println("Tokens: ${options.tokens}, Context: ${options.context}, Temperature: ${options.temperature}")
    println("-".repeat(64))

    val beforeLoad = residentMb()
    var bosToken = 1
    lateinit var runtime: LlamaRuntime<FP32>
    lateinit var tokenizer: GGUFTokenizer
    val loadTime = measureTime {
        val loaded = loadPackedGgufLlamaWeights(modelPath.toString(), ctx)
        var weights = capLlamaContext(loaded, options.context)
        bosToken = weights.metadata.bosTokenId
        if (System.getenv("DEQUANT_PROJ") == "1") {
            println("DEQUANT_PROJ=1: dequantizing non-embedding weights to dense FP32 (core-vs-transformers test)")
            weights = dequantizeNonEmbedding(weights, ctx)
        }
        println("Tensor storage: ${summarizeTensorStorage(weights)}")
        println("Model context: ${loaded.metadata.contextLength}, eager context cap: ${weights.metadata.contextLength}")
        val rw = mapCompactLlamaRuntimeWeights(weights, ctx)
        val attention = CpuAttentionBackend(
            ctx = ctx,
            weights = rw,
            dtype = FP32::class,
            ropeFreqBase = rw.metadata.ropeFreqBase,
            maxContextLength = rw.metadata.contextLength,
        )
        @Suppress("DEPRECATION")
        runtime = LlamaRuntime(
            ctx = ctx, weights = rw, attentionBackend = attention,
            dtype = FP32::class, eps = rw.metadata.rmsNormEps, random = Random(0),
        )
        tokenizer = GGUFTokenizer.fromRandomAccessSource(
            createRandomAccessSource(modelPath.toString())
                ?: error("RandomAccessSource not available for $modelPath"),
        )
    }
    val afterLoad = residentMb()

    val promptTokens = tokenizer.encode(formattedPrompt)
    require(promptTokens.size <= options.context) {
        "Prompt token count ${promptTokens.size} exceeds --ctx ${options.context}"
    }
    val dbg = System.getenv("DEBUG_TOKENS") == "1"
    if (dbg) System.err.println("[tok] formatted=${formattedPrompt.replace("\n", "\\n")}\n[tok] promptIds=${promptTokens.toList()}")

    if (System.getenv("PARITY") == "1") {
        parityDump("llama-runtime", runtime, tokenizer::decode, bosToken, promptTokens)
        return BenchmarkResult(Variant.EagerJvm, options.model, options.tokens, options.context, loadTime.inWholeMilliseconds, 0.0, notes = "parity dump")
    }

    val genIds = mutableListOf<Int>()
    val response = StringBuilder()
    val inferenceTime = measureTime {
        runtime.reset()
        runtime.generate(promptTokens, options.tokens, options.temperature) { tokenId ->
            if (dbg) genIds.add(tokenId)
            response.append(tokenizer.decode(tokenId))
        }
    }
    if (dbg) System.err.println("[tok] genIds=$genIds")
    val afterInference = residentMb()
    val seconds = inferenceTime.inWholeNanoseconds / 1_000_000_000.0
    val result = BenchmarkResult(
        variant = Variant.EagerJvm, model = options.model, tokens = options.tokens,
        context = options.context, loadMs = loadTime.inWholeMilliseconds, inferenceSeconds = seconds,
        peakRssMb = afterInference.toLong(), response = response.toString().trim(),
        notes = "host LlamaRuntime+SIMD, packed; OUTPUT NOT CORRECT (upstream transformers attn bug)",
    )

    println()
    println("Model Response:")
    println("-".repeat(64))
    println(result.response)
    println("-".repeat(64))
    println("Load time: ${result.loadMs} ms")
    println("Inference time: ${roundTo(seconds, 1)} s")
    println("Speed: ${roundTo(result.tokensPerSecond, 1)} tokens/sec")
    println("RSS after inference: ${roundTo(afterInference, 1)} MB (delta load ${roundTo(afterLoad - beforeLoad, 1)} MB)")
    return result
}

/**
 * EXPERIMENTAL / WIP — the `fromWeights` + `OptimizedLLMRuntime` route. Currently produces
 * incoherent output on real GGUF even after [densifyGgufForDsl] (dequant + reorient): shapes
 * are correct and dequant values are sane, so the remaining bug is in WeightMapper/DSL
 * semantics (the upstream issue). Kept to debug + drive the upstream fix; not used by default.
 */
suspend fun runEagerJvmDsl(options: EagerOptions): BenchmarkResult {
    val modelPath = Path.of(resolveTinyLlamaModelPath(options.model)).toAbsolutePath().normalize()
    require(Files.exists(modelPath)) { "Model not found: $modelPath." }

    val prompt = options.prompt ?: readPrompt(options.promptFile, options.promptIndex)
    val formattedPrompt = formatQuestionPrompt(prompt)
    val ctx = DirectCpuExecutionContext()
    println("TinyLlama SKaiNET eager (JVM, fromWeights/OptimizedLLMRuntime — WIP)")

    val loadedWeights: DecoderGgufWeights<FP32, Float>
    val loadTime = measureTime { loadedWeights = loadPackedGgufLlamaWeights(modelPath.toString(), ctx) }
    val weights = capLlamaContext(loadedWeights, options.context)
    println("Tensor storage: ${summarizeTensorStorage(weights)}")

    println("Dequantizing + reorienting weights for the DSL network ...")
    val dslWeights = densifyGgufForDsl(weights, ctx)
    val model = LlamaNetworkLoader.fromWeights(dslWeights)
    val runtime = OptimizedLLMRuntime(
        model = model, ctx = ctx, mode = OptimizedLLMMode.DIRECT,
        dtype = FP32::class, bos = weights.metadata.bosTokenId, random = Random(0),
    )
    val tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { TokenizerFactory.fromGgufSource(it) }
    val promptTokens = tokenizer.encode(formattedPrompt)

    val response = StringBuilder()
    val inferenceTime = measureTime {
        runtime.reset()
        runtime.generate(prompt = promptTokens, steps = options.tokens, temperature = options.temperature) { id ->
            response.append(tokenizer.decode(id))
        }
    }
    val seconds = inferenceTime.inWholeNanoseconds / 1_000_000_000.0
    println("Model Response (WIP, may be incoherent):")
    println(response.toString().trim())
    return BenchmarkResult(
        variant = Variant.EagerJvm, model = options.model, tokens = options.tokens, context = options.context,
        loadMs = loadTime.inWholeMilliseconds, inferenceSeconds = seconds, response = response.toString().trim(),
        notes = "WIP fromWeights/OptimizedLLMRuntime (output not yet coherent)",
    )
}

/**
 * Parity probe: feed [prompt] (BOS-prepended) through [runtime] one token at a time and dump the
 * top-10 next-token logits at the last position — to compare against llama.cpp for the same token
 * IDs (see benchmarks/python/parity_ref.py). Pinpoints where SKaiNET diverges.
 */
internal fun parityDump(tag: String, runtime: InferenceRuntime<FP32>, decode: (Int) -> String, bos: Int, prompt: IntArray) {
    runtime.reset()
    val toks = if (prompt.firstOrNull() == bos) prompt else intArrayOf(bos) + prompt
    var logits: sk.ainet.lang.tensor.Tensor<FP32, Float>? = null
    for (t in toks) logits = runtime.forward(t)
    val data = logits!!.data
    val buf = if (data is FloatArrayTensorData<*>) data.buffer else data.copyToFloatArray()
    val top = buf.indices.sortedByDescending { buf[it] }.take(10)
    println("[parity:$tag] tokens=${toks.toList()}")
    top.forEach { id -> println("[parity:$tag] id=$id logit=${roundTo(buf[id].toDouble(), 3)} tok='${decode(id)}'") }
}

internal fun readPrompt(path: String, index: Int): String {
    val file = File(path)
    if (!file.exists()) return "What is quantization in machine learning?"
    val prompts = file.readText()
        .split(Regex("\\n\\s*\\n|\\n"))
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }
    if (prompts.isEmpty()) return "What is quantization in machine learning?"
    return prompts.getOrElse(index.coerceAtLeast(0) % prompts.size) {
        "What is quantization in machine learning?"
    }
}

internal fun residentMb(): Double {
    val pid = ProcessHandle.current().pid().toString()
    val fromPs = runCatching {
        val proc = ProcessBuilder("ps", "-o", "rss=", "-p", pid).start()
        val text = proc.inputStream.bufferedReader().readText().trim()
        proc.waitFor()
        text.toDouble() / 1024.0
    }.getOrNull()
    if (fromPs != null && fromPs > 0.0) return fromPs

    val heap = ManagementFactory.getMemoryMXBean().heapMemoryUsage.used
    val nonHeap = ManagementFactory.getMemoryMXBean().nonHeapMemoryUsage.used
    return (heap + nonHeap).toDouble() / (1024.0 * 1024.0)
}
