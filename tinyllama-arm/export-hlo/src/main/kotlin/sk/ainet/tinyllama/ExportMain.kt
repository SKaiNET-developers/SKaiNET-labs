package sk.ainet.tinyllama

/**
 * Entry point for the export Gradle tasks.
 *   --out <path>     output MLIR (default build/stablehlo/tinyllama_step.mlir)
 *   --model <id>     if set, export the REAL TinyLlama graph from this GGUF (Q4_K_M | path.gguf)
 *   --ctx <n>        context length cap for the real export (default 32)
 */
fun main(args: Array<String>) {
    // --iree mode: bake the value-correct IREE artifact pair (MLIR with external params + weights
    // safetensors). Requires --model. --weights sets the safetensors path; --seq the prefill len.
    if (args.contains("--iree")) {
        val model = resolveTinyLlamaModelPath(opt(args, "--model") ?: "Q4_K_M")
        val mlir = opt(args, "--out") ?: "build/iree/tinyllama_iree.mlir"
        val weights = opt(args, "--weights") ?: "build/iree/tinyllama_weights.safetensors"
        val seq = opt(args, "--seq")?.toInt() ?: 32
        exportLlamaIree(mlir, weights, model, seq)
        return
    }
    val out = opt(args, "--out") ?: "build/stablehlo/tinyllama_step.mlir"
    val model = opt(args, "--model")?.let { resolveTinyLlamaModelPath(it) }
    val ctx = opt(args, "--ctx")?.toInt() ?: 32
    exportStableHlo(out, model, ctx)
}

private fun opt(args: Array<String>, name: String): String? {
    args.forEachIndexed { i, a ->
        if (a == name) return args.getOrNull(i + 1)
        if (a.startsWith("$name=")) return a.substringAfter("=")
    }
    return null
}
