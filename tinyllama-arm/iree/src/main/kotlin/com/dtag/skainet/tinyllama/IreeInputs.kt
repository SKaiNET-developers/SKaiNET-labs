package com.dtag.skainet.tinyllama

import java.io.File

/**
 * Derives `iree-run-module` `--input` flags from an exported MLIR function signature.
 *
 * The exported graph takes its weights/activations as external inputs (e.g.
 * `func.func @tinyllama_step(%arg0: tensor<16x8xf32>, %arg1: tensor<1xi32>, ...)`).
 * For a smoke/latency run we splat each input with a constant fill value, e.g.
 * `tensor<16x8xf32>` -> `--input=16x8xf32=0`.
 */
object IreeInputs {
    private val funcSig = Regex("""func\.func\s+@(\w+)\s*\(([^)]*)\)""")
    private val tensorType = Regex("""tensor<([^>]+)>""")

    /** Name of the first exported function, or null. */
    fun functionName(mlir: File): String? = funcSig.find(mlir.readText())?.groupValues?.get(1)

    /** `--input=...` flags (zero-filled by default) for every argument of the first function. */
    fun zeroInputs(mlir: File, fill: String = "0"): List<String> {
        val sig = funcSig.find(mlir.readText()) ?: return emptyList()
        val argsBlob = sig.groupValues[2]
        return tensorType.findAll(argsBlob).map { m ->
            "--input=${m.groupValues[1]}=$fill"
        }.toList()
    }
}
