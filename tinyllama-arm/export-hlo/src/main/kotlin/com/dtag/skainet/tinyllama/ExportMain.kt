package com.dtag.skainet.tinyllama

/** Entry point for the `exportStableHlo` Gradle task. */
fun main(args: Array<String>) {
    val out = parseOut(args) ?: "build/stablehlo/tinyllama_step.mlir"
    exportStableHlo(out)
}

private fun parseOut(args: Array<String>): String? {
    args.forEachIndexed { i, a ->
        if (a == "--out") return args.getOrNull(i + 1)
        if (a.startsWith("--out=")) return a.substringAfter("=")
    }
    return null
}
