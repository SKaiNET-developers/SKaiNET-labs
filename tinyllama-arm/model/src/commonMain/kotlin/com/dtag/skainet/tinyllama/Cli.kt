package com.dtag.skainet.tinyllama

sealed interface Command {
    data class Eager(val options: EagerOptions) : Command
    data class Export(val out: String) : Command
    data class Inspect(val model: String) : Command
    data class Bench(val variants: List<Variant>, val options: EagerOptions) : Command
    data object Help : Command
}

data class EagerOptions(
    val model: String = "Q4_K_M",
    val prompt: String? = null,
    val promptFile: String = "prompts/prompts.txt",
    val promptIndex: Int = 0,
    val tokens: Int = 128,
    val temperature: Float = 0.8f,
    val context: Int = 512,
    val threads: Int = 4,
)

fun parseCommand(args: Array<String>): Command {
    if (args.isEmpty()) return Command.Help
    return when (args[0]) {
        "eager" -> Command.Eager(parseEager(args.drop(1)))
        "export" -> Command.Export(option(args.drop(1), "--out") ?: "build/stablehlo/tinyllama_step.mlir")
        "inspect" -> Command.Inspect(option(args.drop(1), "--model") ?: "Q4_K_M")
        "bench" -> {
            val rest = args.drop(1)
            val variants = Variant.parseList(option(rest, "--variants") ?: "eager-native,iree-cpu")
            Command.Bench(variants, parseEager(stripOption(rest, "--variants")))
        }
        "-h", "--help", "help" -> Command.Help
        else -> Command.Help
    }
}

/** Remove a `--name value` / `--name=value` pair from an argument list. */
private fun stripOption(args: List<String>, name: String): List<String> {
    val out = mutableListOf<String>()
    var i = 0
    while (i < args.size) {
        val a = args[i]
        when {
            a == name -> i++ // skip the following value too
            a.startsWith("$name=") -> {} // drop
            else -> out.add(a)
        }
        i++
    }
    return out
}

private fun parseEager(args: List<String>): EagerOptions {
    var opts = EagerOptions()
    var i = 0
    while (i < args.size) {
        val key = args[i]
        fun value(): String {
            require(i + 1 < args.size) { "Missing value for $key" }
            return args[++i]
        }
        when {
            key == "--model" -> opts = opts.copy(model = value())
            key.startsWith("--model=") -> opts = opts.copy(model = key.substringAfter("="))
            key == "--prompt" -> opts = opts.copy(prompt = value())
            key.startsWith("--prompt=") -> opts = opts.copy(prompt = key.substringAfter("="))
            key == "--prompt-file" -> opts = opts.copy(promptFile = value())
            key.startsWith("--prompt-file=") -> opts = opts.copy(promptFile = key.substringAfter("="))
            key == "--prompt-index" -> opts = opts.copy(promptIndex = value().toInt())
            key.startsWith("--prompt-index=") -> opts = opts.copy(promptIndex = key.substringAfter("=").toInt())
            key == "--tokens" -> opts = opts.copy(tokens = value().toInt())
            key.startsWith("--tokens=") -> opts = opts.copy(tokens = key.substringAfter("=").toInt())
            key == "--temperature" -> opts = opts.copy(temperature = value().toFloat())
            key.startsWith("--temperature=") -> opts = opts.copy(temperature = key.substringAfter("=").toFloat())
            key == "--ctx" -> opts = opts.copy(context = value().toInt())
            key.startsWith("--ctx=") -> opts = opts.copy(context = key.substringAfter("=").toInt())
            key == "--threads" -> opts = opts.copy(threads = value().toInt())
            key.startsWith("--threads=") -> opts = opts.copy(threads = key.substringAfter("=").toInt())
            else -> error("Unknown eager option: $key")
        }
        i++
    }
    require(opts.tokens > 0) { "--tokens must be positive" }
    require(opts.context > 0) { "--ctx must be positive" }
    return opts
}

private fun option(args: List<String>, name: String): String? {
    args.forEachIndexed { idx, arg ->
        if (arg == name) return args.getOrNull(idx + 1)
        if (arg.startsWith("$name=")) return arg.substringAfter("=")
    }
    return null
}

fun resolveTinyLlamaModelPath(model: String): String =
    if (model.endsWith(".gguf", ignoreCase = true) || model.contains('/')) {
        model
    } else {
        "models/tinyllama-1.1b-chat-v1.0.$model.gguf"
    }

fun formatQuestionPrompt(prompt: String): String = "Question: $prompt\n\nAnswer:"

fun usage(): String = """
TinyLlama SKaiNET / StableHLO / IREE

Commands:
  eager   --model <Q4_K_M|Q8_0|path.gguf> [--prompt TEXT] [--tokens 128] [--temperature 0.8]
  inspect --model <Q4_K_M|Q8_0|path.gguf>
  export  --out build/stablehlo/tinyllama_step.mlir
  bench   --variants eager-native,eager-jvm,iree-cpu [--model ...] [--tokens ...] [--ctx ...]

Variants: ${Variant.entries.joinToString(", ") { it.id }}

Examples:
  ./gradlew runJvm --args='eager --model Q4_K_M --tokens 64 --prompt "What is quantization?"'
  ./gradlew :bench:runJvm --args='bench --variants eager-native,iree-cpu --tokens 4 --ctx 64'
  # host arm64 native (Apple Silicon, no board) — build first, then bench:
  ./gradlew :eager:linkReleaseExecutableMacosArm64
  ./gradlew :bench:runJvm --args='bench --variants eager-native-host,eager-jvm --tokens 8 --ctx 128'
  ./gradlew exportStableHlo
""".trimIndent()

