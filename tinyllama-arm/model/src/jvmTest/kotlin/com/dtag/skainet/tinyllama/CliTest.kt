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
        val root = repoRoot()
        val checkedFiles = listOf(
            "README.md",
            "docs/BOARD_SL2619.md",
            "eager/src/jvmMain/kotlin/com/dtag/skainet/tinyllama/EagerJvm.kt",
            "eager/src/nativeMain/kotlin/com/dtag/skainet/tinyllama/EagerNative.kt",
        )
        val combinedText = checkedFiles
            .map { root.resolve(it).toFile() }
            .filter { it.exists() }
            .joinToString("\n") { it.readText() }

        assertFalse("DEQUANTIZE_TO_FP32" in combinedText)
        assertFalse("dequantized to FP32" in combinedText)
        assertFalse("dequantizes Llama GGUF weights to FP32" in combinedText)
    }

    /** Walk up from the test working directory to the repository root (holds settings.gradle.kts). */
    private fun repoRoot(): Path {
        var dir: Path? = Path.of("").toAbsolutePath()
        while (dir != null) {
            if (dir.resolve("settings.gradle.kts").toFile().exists()) return dir
            dir = dir.parent
        }
        return Path.of("").toAbsolutePath()
    }
}
