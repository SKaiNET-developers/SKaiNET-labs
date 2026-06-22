package com.dtag.skainet.tinyllama

import java.io.File
import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import kotlin.random.Random
import kotlin.time.measureTime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaNetworkLoader

/** Run TinyLlama eager on the JVM (OptimizedLLMRuntime DIRECT), printing and returning a [BenchmarkResult]. */
suspend fun runEagerJvm(options: EagerOptions): BenchmarkResult {
    val modelPath = Path.of(resolveTinyLlamaModelPath(options.model)).toAbsolutePath().normalize()
    require(Files.exists(modelPath)) {
        "Model not found: $modelPath. Run scripts/download-models.sh first."
    }

    val prompt = options.prompt ?: readPrompt(options.promptFile, options.promptIndex)
    val formattedPrompt = formatQuestionPrompt(prompt)
    val ctx = DirectCpuExecutionContext()

    println("TinyLlama SKaiNET eager benchmark")
    println("Model: $modelPath")
    println("Prompt: $prompt")
    println("Tokens: ${options.tokens}, Context: ${options.context}, Temperature: ${options.temperature}")
    println("Threads requested: ${options.threads} (accepted for parity with the Arm Python sample; SKaiNET chooses backend threading internally)")
    println("Weight path: streaming GGUF with packed SKaiNET quantized tensor storage")
    println("-".repeat(64))

    val beforeLoad = residentMb()
    val loadedWeights: DecoderGgufWeights<FP32, Float>
    val loadTime = measureTime {
        loadedWeights = loadPackedGgufLlamaWeights(modelPath.toString(), ctx)
    }
    val afterLoad = residentMb()
    val weights = capLlamaContext(loadedWeights, options.context)
    println("Tensor storage: ${summarizeTensorStorage(weights)}")
    println("Model context: ${loadedWeights.metadata.contextLength}, eager context cap: ${weights.metadata.contextLength}")

    val model = LlamaNetworkLoader.fromWeights(weights)
    val runtime = OptimizedLLMRuntime(
        model = model,
        ctx = ctx,
        mode = OptimizedLLMMode.DIRECT,
        dtype = FP32::class,
        bos = weights.metadata.bosTokenId,
        random = Random(0),
    )

    val tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
        TokenizerFactory.fromGgufSource(source)
    }
    val promptTokens = tokenizer.encode(formattedPrompt)
    require(promptTokens.size <= options.context) {
        "Prompt token count ${promptTokens.size} exceeds --ctx ${options.context}"
    }

    val response = StringBuilder()
    val inferenceTime = measureTime {
        runtime.reset()
        runtime.generate(
            prompt = promptTokens,
            steps = options.tokens,
            temperature = options.temperature,
        ) { tokenId ->
            response.append(tokenizer.decode(tokenId))
        }
    }
    val afterInference = residentMb()
    val seconds = inferenceTime.inWholeNanoseconds / 1_000_000_000.0
    val result = BenchmarkResult(
        variant = Variant.EagerJvm,
        model = options.model,
        tokens = options.tokens,
        context = options.context,
        loadMs = loadTime.inWholeMilliseconds,
        inferenceSeconds = seconds,
        peakRssMb = afterInference.toLong(),
        response = response.toString().trim(),
        notes = "SIMD kernels via backend-native-cpu when available",
    )

    println()
    println("Model Response:")
    println("-".repeat(64))
    println(result.response)
    println("-".repeat(64))
    println()
    println("Performance Results:")
    println("Load time: ${result.loadMs} ms")
    println("Inference time: ${roundTo(seconds, 1)} s")
    println("Speed: ${roundTo(result.tokensPerSecond, 1)} tokens/sec")
    println("Throughput: ${roundTo(result.tokensPerSecond * 60, 0)} tokens/min")
    println()
    println("Memory Usage:")
    println("Before load: ${roundTo(beforeLoad, 1)} MB")
    println("After load: ${roundTo(afterLoad, 1)} MB")
    println("After inference: ${roundTo(afterInference, 1)} MB")
    println("Model loading delta: ${roundTo(afterLoad - beforeLoad, 1)} MB")
    println("Inference delta: ${roundTo(afterInference - afterLoad, 1)} MB")
    return result
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
