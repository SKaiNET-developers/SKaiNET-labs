package com.dtag.skainet.tinyllama

import java.nio.file.Files
import java.nio.file.Path
import kotlin.system.exitProcess
import kotlinx.coroutines.runBlocking
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

fun main(args: Array<String>) {
    runBlocking {
        try {
            when (val command = parseCommand(args)) {
                is Command.Eager -> runEagerJvm(command.options)
                is Command.Export -> exportStableHlo(command.out)
                is Command.Inspect -> inspectGguf(command.model)
                is Command.Bench -> runComparison(command.variants, command.options)
                Command.Help -> println(usage())
            }
        } catch (t: Throwable) {
            System.err.println("error: ${t.message}")
            exitProcess(1)
        }
    }
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
