package com.dtag.skainet.tinyllama

import kotlin.math.round

/**
 * One measured run of a [Variant]. Units differ by variant: eager variants report
 * end-to-end token generation; an IREE single-step graph reports per-invocation
 * latency (tokens=1, inferenceSeconds = step latency). [notes] carries that nuance.
 */
data class BenchmarkResult(
    val variant: Variant,
    val model: String,
    val tokens: Int,
    val context: Int,
    val loadMs: Long,
    val inferenceSeconds: Double,
    val peakRssMb: Long = -1L,
    val response: String = "",
    val notes: String = "",
) {
    val tokensPerSecond: Double get() = if (inferenceSeconds > 0.0) tokens / inferenceSeconds else 0.0
    val secondsPerToken: Double get() = if (tokens > 0) inferenceSeconds / tokens else 0.0
}

/** Round to [digits] decimals as a string, platform-independently. */
fun roundTo(value: Double, digits: Int): String {
    var factor = 1.0
    repeat(digits) { factor *= 10.0 }
    return (round(value * factor) / factor).toString()
}

/** Render results as an aligned plain-text comparison table. */
fun formatComparisonTable(results: List<BenchmarkResult>): String {
    val headers = listOf("variant", "tok/s", "s/tok", "load(ms)", "infer(s)", "RSS(MB)", "notes")
    val rows = results.map { r ->
        listOf(
            r.variant.id,
            roundTo(r.tokensPerSecond, 4),
            roundTo(r.secondsPerToken, 2),
            r.loadMs.toString(),
            roundTo(r.inferenceSeconds, 1),
            if (r.peakRssMb >= 0) r.peakRssMb.toString() else "-",
            r.notes,
        )
    }
    val widths = headers.indices.map { col ->
        (rows.map { it[col].length } + headers[col].length).max()
    }
    fun line(cells: List<String>) =
        cells.mapIndexed { i, c -> c.padEnd(widths[i]) }.joinToString("  ").trimEnd()
    return buildString {
        appendLine(line(headers))
        appendLine(line(widths.map { "-".repeat(it) }))
        rows.forEach { appendLine(line(it)) }
    }.trimEnd()
}
