package com.dtag.skainet.tinyllama

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertIs

class CliTest {
    @Test
    fun resolvesArmSampleModelVariants() {
        assertEquals(
            "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            resolveTinyLlamaModelPath("Q4_K_M"),
        )
        assertEquals("/tmp/model.gguf", resolveTinyLlamaModelPath("/tmp/model.gguf"))
    }

    @Test
    fun parsesEagerCommand() {
        val command = parseCommand(arrayOf("eager", "--model", "Q8_0", "--tokens", "16", "--prompt", "Hi"))
        val eager = assertIs<Command.Eager>(command)
        assertEquals("Q8_0", eager.options.model)
        assertEquals(16, eager.options.tokens)
        assertEquals("Hi", eager.options.prompt)
    }

    @Test
    fun eagerPathDoesNotClaimWholeGgufFp32Dequantization() {
        val root = Path.of("").toAbsolutePath()
        val checkedFiles = listOf(
            "README.md",
            "docs/BOARD_SL2619.md",
            "src/jvmMain/kotlin/com/dtag/skainet/tinyllama/Main.kt",
            "src/nativeMain/kotlin/com/dtag/skainet/tinyllama/Main.kt",
        )
        val combinedText = checkedFiles.joinToString("\n") { relativePath ->
            root.resolve(relativePath).toFile().readText()
        }

        assertFalse("DEQUANTIZE_TO_FP32" in combinedText)
        assertFalse("dequantized to FP32" in combinedText)
        assertFalse("dequantizes Llama GGUF weights to FP32" in combinedText)
    }
}
