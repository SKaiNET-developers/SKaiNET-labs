package com.dtag.skainet.tinyllama

import java.io.File

/** Drives `iree-compile` inside a Docker image (no host iree-compile available). */
object IreeCompile {
    const val DEFAULT_TARGET_TRIPLE = "aarch64-unknown-linux-gnu"
    const val DEFAULT_TARGET_CPU = "generic"

    // Compiler whose bytecode (16.0) matches the SL2619 board's iree-run-module.
    // Build with: docker build -t skainet-iree-compile:3.10.0 -f docker/iree-compile.Dockerfile .
    const val DEFAULT_IMAGE = "skainet-iree-compile:3.10.0"

    /**
     * Compile [mlir] to a `.vmfb` at [outVmfb] via scripts/compile-iree-docker.sh using [image].
     * [image] must contain an `iree-compile` binary (see docker/iree-compile.Dockerfile).
     */
    fun compileViaDocker(
        mlir: File,
        outVmfb: File,
        image: String,
        targetTriple: String = DEFAULT_TARGET_TRIPLE,
        targetCpu: String = DEFAULT_TARGET_CPU,
        halBackends: String = "llvm-cpu",
        scriptsDir: File = File("scripts"),
    ): File {
        require(mlir.exists()) { "MLIR not found: ${mlir.path}" }
        outVmfb.parentFile?.mkdirs()
        val script = File(scriptsDir, "compile-iree-docker.sh")
        require(script.exists()) { "Missing ${script.path}" }
        val env = mapOf(
            "IREE_DOCKER_IMAGE" to image,
            "IREE_TARGET_TRIPLE" to targetTriple,
            "IREE_TARGET_CPU" to targetCpu,
            "IREE_HAL_BACKENDS" to halBackends,
        )
        runProcess(listOf("bash", script.path, mlir.path, outVmfb.path), env)
            .requireSuccess("iree-compile (docker)")
        require(outVmfb.exists()) { "iree-compile did not produce ${outVmfb.path}" }
        return outVmfb
    }
}
