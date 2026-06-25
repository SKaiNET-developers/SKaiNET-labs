package com.dtag.skainet.tinyllama

import java.io.File

private const val ADB_SERIAL = "192.168.3.26:5555"
private const val BOARD_MODEL_DIR = "/home/skainet-tinyllama/models"
private const val TOY_MLIR = "build/stablehlo/tinyllama_step.mlir"
private const val TOY_VMFB = "build/iree/tinyllama_step.vmfb"

/** Run each requested [variants] entry, collect a [BenchmarkResult], and print a comparison table. */
suspend fun runComparison(variants: List<Variant>, options: EagerOptions): List<BenchmarkResult> {
    val results = variants.map { v ->
        println()
        println("========== variant: ${v.id}  (${v.label}) ==========")
        runCatching {
            when (v) {
                Variant.EagerJvm ->
                    if (System.getenv("EAGER_JVM_DSL") == "1") runEagerJvmDsl(options) else runEagerJvm(options)
                Variant.EagerNative -> runEagerNativeOnBoard(options)
                Variant.EagerNativeHost -> runEagerNativeHost(options)
                Variant.IreeCpu -> runIreeStepOnBoard(options, "local-task://")
                Variant.IreeTorq -> runIreeStepOnBoard(options, "torq://")
                Variant.PythonBaseline -> runPythonBaseline(options)
            }
        }.getOrElse { e ->
            println("variant ${v.id} FAILED: ${e.message}")
            BenchmarkResult(
                variant = v, model = options.model, tokens = options.tokens, context = options.context,
                loadMs = 0L, inferenceSeconds = 0.0, notes = "FAILED: ${(e.message ?: "").take(80)}",
            )
        }
    }
    println()
    println("================= COMPARISON =================")
    println(formatComparisonTable(results))
    println()
    println("Notes:")
    println(" - eager-* run end-to-end but SKaiNET output is NOT yet numerically correct vs")
    println("   llama.cpp (upstream skainet-transformers attention bug); python-baseline is the")
    println("   only correct reference today. Tokens/sec are still meaningful.")
    println(" - iree-* measure a single decoder-step graph (the real-model graph compiles but its")
    println("   FP32 weights don't fit the 2 GB board). Units are not directly comparable.")
    return results
}

/** eager on the board (native binary via adb), parsed from its stdout. */
private fun runEagerNativeOnBoard(options: EagerOptions): BenchmarkResult {
    val args = listOf(
        "bash", "scripts/adb-board-run.sh", "eager",
        "--model", boardModelPath(options.model),
        "--tokens", options.tokens.toString(),
        "--ctx", options.context.toString(),
        "--temperature", options.temperature.toString(),
        "--prompt", options.prompt ?: ".",
    )
    val out = runProcess(args, mapOf("ADB_SERIAL" to ADB_SERIAL, "FORCE_BUILD" to "0"))
        .requireSuccess("eager-native (board)").output
    fun num(re: String): Double? = Regex(re).find(out)?.groupValues?.get(1)?.toDoubleOrNull()
    val loadMs = num("""Load time:\s*([0-9]+)\s*ms""")?.toLong() ?: 0L
    val infS = num("""Inference time:\s*([0-9.]+)\s*s""") ?: 0.0
    val rss = num("""Process RSS:\s*([0-9]+)\s*MB""")?.toLong() ?: -1L
    return BenchmarkResult(
        variant = Variant.EagerNative, model = options.model, tokens = options.tokens,
        context = options.context, loadMs = loadMs, inferenceSeconds = infS, peakRssMb = rss,
        notes = "board CPU, scalar packed kernels",
    )
}

/**
 * eager on the HOST via the macosArm64 Kotlin/Native binary. Same native runtime path as the
 * board (`LlamaRuntime`), but on Apple Silicon it dispatches to Accelerate (NEON + AMX), so the
 * tok/s is NOT board-comparable — it's a same-ISA (arm64), no-adb datapoint for the eager path.
 * Reproduces the native runtime's correctness bug on the host (debuggable without the board).
 */
private fun runEagerNativeHost(options: EagerOptions): BenchmarkResult {
    val bin = File("eager/build/bin/macosArm64/releaseExecutable/tinyllama-skainet.kexe")
    require(bin.exists()) {
        "Missing $bin — run ./gradlew :eager:linkReleaseExecutableMacosArm64 first."
    }
    val args = listOf(
        bin.absolutePath, "eager",
        "--model", hostModelPath(options.model),
        "--tokens", options.tokens.toString(),
        "--ctx", options.context.toString(),
        "--temperature", options.temperature.toString(),
        "--prompt", options.prompt ?: ".",
    )
    val out = runProcess(args).requireSuccess("eager-native-host").output
    fun num(re: String): Double? = Regex(re).find(out)?.groupValues?.get(1)?.toDoubleOrNull()
    val loadMs = num("""Load time:\s*([0-9]+)\s*ms""")?.toLong() ?: 0L
    val infS = num("""Inference time:\s*([0-9.]+)\s*s""") ?: 0.0
    val rss = num("""Process RSS:\s*([0-9]+)\s*MB""")?.toLong() ?: -1L
    return BenchmarkResult(
        variant = Variant.EagerNativeHost, model = options.model, tokens = options.tokens,
        context = options.context, loadMs = loadMs, inferenceSeconds = infS, peakRssMb = rss,
        notes = "host arm64 Kotlin/Native (Accelerate NEON+AMX); not board-comparable",
    )
}

/** Single decoder-step IREE graph on the board (the toy step graph; the real one won't fit). */
private fun runIreeStepOnBoard(options: EagerOptions, device: String): BenchmarkResult {
    val mlir = File(TOY_MLIR)
    require(mlir.exists()) { "Missing $TOY_MLIR — run ./gradlew exportStableHlo first." }
    val vmfb = File(TOY_VMFB)
    if (!vmfb.exists()) IreeCompile.compileViaDocker(mlir, vmfb, IreeCompile.DEFAULT_IMAGE)
    val fn = IreeInputs.functionName(mlir) ?: "tinyllama_step"
    val inputs = IreeInputs.zeroInputs(mlir)
    return IreeRun.runOnBoard(vmfb, fn, device, ADB_SERIAL, options.model, inputs)
}

/** llama.cpp host baseline via uv (scripts/python-baseline.sh), parsed from its stdout. */
private fun runPythonBaseline(options: EagerOptions): BenchmarkResult {
    val args = listOf(
        "bash", "scripts/python-baseline.sh",
        "--model", hostModelPath(options.model),
        "--tokens", options.tokens.toString(),
        "--ctx", options.context.toString(),
        "--threads", options.threads.toString(),
        "--prompt", options.prompt ?: "What is quantization in machine learning?",
    )
    val out = runProcess(args).requireSuccess("python-baseline (uv)").output
    fun num(re: String): Double? = Regex(re).find(out)?.groupValues?.get(1)?.toDoubleOrNull()
    // Derive seconds from the script's reported tokens/sec so the table matches it exactly
    // (its "Inference time" line is only 2 decimals, which skews a recomputed rate).
    val speed = num("""Speed:\s*([0-9.]+)\s*tokens/sec""")
    val infS = if (speed != null && speed > 0.0) options.tokens / speed
        else num("""Inference time:\s*([0-9.]+)\s*s""") ?: 0.0
    val rss = num("""Total usage:\s*([0-9.]+)\s*MB""")?.toLong() ?: -1L
    return BenchmarkResult(
        variant = Variant.PythonBaseline, model = options.model, tokens = options.tokens,
        context = options.context, loadMs = 0L, inferenceSeconds = infS, peakRssMb = rss,
        notes = "host llama.cpp (uv); reference, not the board",
    )
}

private fun hostModelPath(model: String): String =
    if (model.endsWith(".gguf", ignoreCase = true) || model.contains('/')) model
    else java.io.File("models/tinyllama-1.1b-chat-v1.0.$model.gguf").absolutePath

private fun boardModelPath(model: String): String =
    if (model.endsWith(".gguf", ignoreCase = true) || model.contains('/')) model
    else "$BOARD_MODEL_DIR/tinyllama-1.1b-chat-v1.0.$model.gguf"
