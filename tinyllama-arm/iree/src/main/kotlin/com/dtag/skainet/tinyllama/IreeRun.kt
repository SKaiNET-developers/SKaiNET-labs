package com.dtag.skainet.tinyllama

import java.io.File

/** Drives `iree-run-module` on the board over adb (the board has the runtime, not the compiler). */
object IreeRun {
    /**
     * Push [vmfb] to the board and run [function] on [device] (`local-task://` CPU or `torq://` NPU),
     * via scripts/adb-iree-run.sh. Returns a [BenchmarkResult] whose inferenceSeconds is the measured
     * per-invocation latency (tokens=1; this is a single decoder-step graph, not full generation).
     */
    fun runOnBoard(
        vmfb: File,
        function: String,
        device: String = "local-task://",
        // Default arguments are evaluated only when omitted, so host callers never hit this.
        adbSerial: String = System.getenv("ADB_SERIAL")
            ?: error("ADB_SERIAL is not set. Board runs need e.g. ADB_SERIAL=<board-host>:5555."),
        model: String = "tinyllama_step",
        extraArgs: List<String> = emptyList(),
        scriptsDir: File = File("scripts"),
    ): BenchmarkResult {
        require(vmfb.exists()) { "VMFB not found: ${vmfb.path}" }
        val script = File(scriptsDir, "adb-iree-run.sh")
        require(script.exists()) { "Missing ${script.path}" }
        val variant = if (device.startsWith("torq")) Variant.IreeTorq else Variant.IreeCpu

        val cmd = buildList {
            add("bash"); add(script.path); add(vmfb.path)
            add("--device=$device")
            add("--function=$function")
            add("--print_statistics=true")
            addAll(extraArgs)
        }
        // This iree-run-module build has no timing flag, so measure wall-clock of the call
        // (includes vmfb push + adb/process startup). For tiny graphs that startup dominates.
        val startNs = System.nanoTime()
        val result = runProcess(cmd, mapOf("ADB_SERIAL" to adbSerial)).requireSuccess("iree-run-module (board)")
        val elapsedS = (System.nanoTime() - startNs) / 1_000_000_000.0
        val parsed = parseLatencySeconds(result.output)
        return BenchmarkResult(
            variant = variant,
            model = model,
            tokens = 1,
            context = 0,
            loadMs = 0L,
            inferenceSeconds = if (parsed > 0.0) parsed else elapsedS,
            response = "",
            notes = "single-step graph; wall-clock incl. push+startup on $device",
        )
    }

    /**
     * Best-effort parse of iree-run-module timing. iree prints lines like
     * "...  real_time ... 12.3 ms ..."; fall back to 0.0 if not found.
     */
    internal fun parseLatencySeconds(output: String): Double {
        val msRegex = Regex("""([0-9]+(?:\.[0-9]+)?)\s*ms""")
        val usRegex = Regex("""([0-9]+(?:\.[0-9]+)?)\s*us""")
        val ms = output.lineSequence()
            .filter { it.contains("time", ignoreCase = true) || it.contains("elapsed", ignoreCase = true) }
            .mapNotNull { msRegex.find(it)?.groupValues?.get(1)?.toDoubleOrNull() }
            .maxOrNull()
        if (ms != null) return ms / 1000.0
        val us = output.lineSequence()
            .mapNotNull { usRegex.find(it)?.groupValues?.get(1)?.toDoubleOrNull() }
            .maxOrNull()
        return if (us != null) us / 1_000_000.0 else 0.0
    }
}
