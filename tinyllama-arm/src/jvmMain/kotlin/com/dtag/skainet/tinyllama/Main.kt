package com.dtag.skainet.tinyllama

import java.io.File
import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.round
import kotlin.random.Random
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.compile.hlo.toStableHlo
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
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

fun main(args: Array<String>) = runBlocking {
    try {
        when (val command = parseCommand(args)) {
            is Command.Eager -> runEager(command.options)
            is Command.Export -> exportStableHlo(command.out)
            is Command.Inspect -> inspectGguf(command.model)
            Command.Help -> println(usage())
        }
    } catch (t: Throwable) {
        System.err.println("error: ${t.message}")
        exitProcess(1)
    }
}

private suspend fun runEager(options: EagerOptions) {
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
    val tokPerSec = options.tokens / seconds

    println()
    println("Model Response:")
    println("-".repeat(64))
    println(response.toString().trim())
    println("-".repeat(64))
    println()
    println("Performance Results:")
    println("Load time: ${loadTime.inWholeMilliseconds} ms")
    println("Inference time: ${format1(seconds)} s")
    println("Speed: ${format1(tokPerSec)} tokens/sec")
    println("Throughput: ${format0(tokPerSec * 60)} tokens/min")
    println()
    println("Memory Usage:")
    println("Before load: ${format1(beforeLoad)} MB")
    println("After load: ${format1(afterLoad)} MB")
    println("After inference: ${format1(afterInference)} MB")
    println("Model loading delta: ${format1(afterLoad - beforeLoad)} MB")
    println("Inference delta: ${format1(afterInference - afterLoad)} MB")
}

private fun inspectGguf(model: String) {
    val modelPath = Path.of(resolveTinyLlamaModelPath(model)).toAbsolutePath().normalize()
    require(Files.exists(modelPath)) { "Model not found: $modelPath" }
    JvmRandomAccessSource.open(modelPath.toString()).use { source ->
        StreamingGGUFReader.open(source).use { reader ->
            val fields = reader.fields
            println("Model: $modelPath")
            println("Architecture: ${fields["general.architecture"] ?: "unknown"}")
            println("Name: ${fields["general.name"] ?: "unknown"}")
            println("Tensors: ${reader.tensorCount}")
            println("Context length: ${fields["llama.context_length"] ?: fields["mistral.context_length"] ?: "unknown"}")
            println("Embedding length: ${fields["llama.embedding_length"] ?: fields["mistral.embedding_length"] ?: "unknown"}")
            println("Blocks: ${fields["llama.block_count"] ?: fields["mistral.block_count"] ?: "unknown"}")
            println("EOS token: ${fields["tokenizer.ggml.eos_token_id"] ?: "unknown"}")
        }
    }
}

private fun exportStableHlo(out: String) {
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
    println("WROTE_MLIR ${outFile.absolutePath}")
    println("Graph: ${graph.nodes.size} nodes, ${graph.edges.size} edges")
    println("MLIR: ${mlir.lines().size} lines, unsupported markers=$unsupported")
    if (unsupported > 0) {
        println("Note: this records the current compiler gap count for the Llama-shaped graph.")
    }
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

private fun readPrompt(path: String, index: Int): String {
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

private fun residentMb(): Double {
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

private fun format0(value: Double): String = round(value).toLong().toString()
private fun format1(value: Double): String = (round(value * 10.0) / 10.0).toString()
