package com.dtag.skainet.tinyllama

/** An execution backend that can run (a step or full generation of) the model. */
enum class Variant(val id: String, val label: String) {
    EagerJvm("eager-jvm", "Eager / JVM (OptimizedLLMRuntime)"),
    EagerNative("eager-native", "Eager / Kotlin-Native (LlamaRuntime)"),
    IreeCpu("iree-cpu", "IREE / local-task:// (CPU)"),
    IreeTorq("iree-torq", "IREE / torq:// (NPU)"),
    ;

    companion object {
        fun fromId(id: String): Variant =
            entries.firstOrNull { it.id == id }
                ?: error("Unknown variant '$id'. Known: ${entries.joinToString(", ") { it.id }}")

        fun parseList(csv: String): List<Variant> =
            csv.split(",").map { it.trim() }.filter { it.isNotEmpty() }.map { fromId(it) }
    }
}
