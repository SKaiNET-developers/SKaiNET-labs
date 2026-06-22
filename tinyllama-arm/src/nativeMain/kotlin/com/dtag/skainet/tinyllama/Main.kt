package com.dtag.skainet.tinyllama

import kotlin.random.Random
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.readString
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaRuntime

fun main(args: Array<String>) = runBlocking {
    try {
        when (val command = parseCommand(args)) {
            is Command.Eager -> runNativeEager(command.options)
            is Command.Export -> println("StableHLO export is JVM-only in this repo. Run ./gradlew exportStableHlo.")
            is Command.Inspect -> println("GGUF inspect is JVM-only in this repo. Run ./gradlew runJvm --args='inspect --model ${command.model}'.")
            Command.Help -> println(usage())
        }
    } catch (t: Throwable) {
        println("error: ${t.message}")
        exitProcess(1)
    }
}

/** Resident set size of this process in MB, read from /proc/self/statm. -1 if unavailable. */
private fun rssMb(): Long = try {
    val text = SystemFileSystem.source(Path("/proc/self/statm")).buffered().use { it.readString() }
    text.trim().split(" ")[1].toLong() * 4096L / (1024L * 1024L)
} catch (t: Throwable) {
    -1L
}

private suspend fun runNativeEager(options: EagerOptions) {
    val modelPathString = resolveTinyLlamaModelPath(options.model)
    val modelPath = Path(modelPathString)
    require(SystemFileSystem.exists(modelPath)) {
        "Model not found: $modelPathString. Run scripts/download-models.sh first."
    }

    val prompt = options.prompt ?: "What is quantization in machine learning?"
    val formattedPrompt = formatQuestionPrompt(prompt)
    val ctx = DirectCpuExecutionContext()

    println("TinyLlama SKaiNET native ARM64 eager benchmark")
    println("Model: $modelPathString")
    println("Prompt: $prompt")
    println("Tokens: ${options.tokens}, Context: ${options.context}, Temperature: ${options.temperature}")
    println("Native path: compact SKaiNET eager runtime with packed GGUF quantized tensor storage")
    println("-".repeat(64))

    lateinit var runtime: LlamaRuntime<FP32>
    lateinit var tokenizer: GGUFTokenizer
    val loadTime = measureTime {
        val loadedWeights = loadPackedGgufLlamaWeights(modelPathString, ctx)
        val weights = capLlamaContext(loadedWeights, options.context)
        println("Tensor storage: ${summarizeTensorStorage(weights)}")
        println("Model context: ${loadedWeights.metadata.contextLength}, eager context cap: ${weights.metadata.contextLength}")
        val runtimeWeights = mapCompactLlamaRuntimeWeights(weights, ctx)
        val attention = CpuAttentionBackend(
            ctx = ctx,
            weights = runtimeWeights,
            dtype = FP32::class,
            ropeFreqBase = runtimeWeights.metadata.ropeFreqBase,
            maxContextLength = runtimeWeights.metadata.contextLength,
        )
        @Suppress("DEPRECATION")
        runtime = LlamaRuntime(
            ctx = ctx,
            weights = runtimeWeights,
            attentionBackend = attention,
            dtype = FP32::class,
            eps = runtimeWeights.metadata.rmsNormEps,
            random = Random(0),
        )
        // Use the random-access (pread-based) GGUF reader for the tokenizer. The
        // sequential GGUFTokenizer.fromSource path pulls far more than metadata into
        // memory and, on top of the ~650 MB of resident packed weights, OOM-kills
        // the 2 GB board. fromRandomAccessSource parses metadata only (~1 MB).
        tokenizer = GGUFTokenizer.fromRandomAccessSource(
            createRandomAccessSource(modelPathString)
                ?: error("RandomAccessSource is not available for this platform/path: $modelPathString"),
        )
    }

    val promptTokens = tokenizer.encode(formattedPrompt)
    require(promptTokens.size <= options.context) {
        "Prompt token count ${promptTokens.size} exceeds --ctx ${options.context}"
    }

    val response = StringBuilder()
    val inferenceTime = measureTime {
        runtime.reset()
        runtime.generate(promptTokens, options.tokens, options.temperature) { tokenId ->
            response.append(tokenizer.decode(tokenId))
        }
    }

    val seconds = inferenceTime.inWholeNanoseconds / 1_000_000_000.0
    val tokPerSec = options.tokens / seconds

    println()
    println("Model Response:")
    println("-".repeat(64))
    println(response.toString().trim())
    println("-".repeat(64))
    val secPerTok = seconds / options.tokens
    println("Load time: ${loadTime.inWholeMilliseconds} ms")
    println("Inference time: ${format1(seconds)} s")
    // tokens/sec rounds to 0.0 on the slow native scalar-kernel path, so also
    // report s/token, which stays informative at sub-0.05 tok/s rates.
    println("Speed: ${format3(tokPerSec)} tokens/sec (${format1(secPerTok)} s/token)")
    println("Process RSS: ${rssMb()} MB")
}

private fun format1(value: Double): String {
    val rounded = kotlin.math.round(value * 10.0) / 10.0
    return rounded.toString()
}

private fun format3(value: Double): String {
    val rounded = kotlin.math.round(value * 1000.0) / 1000.0
    return rounded.toString()
}
