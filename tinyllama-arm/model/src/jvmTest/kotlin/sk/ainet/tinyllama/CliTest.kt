package sk.ainet.tinyllama

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContains
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
    fun nativeBoardPathPreservesPackedStorage() {
        // The NATIVE board path must keep GGUF weights packed BY DEFAULT (the 2 GB board can't
        // hold the ~4.4 GB FP32 expansion). An FP32 parity path exists but must stay behind the
        // host-only EAGER_NATIVE_FP32 env gate — it must never become the default policy.
        val root = repoRoot()
        val native = root.resolve("eager/src/nativeMain/kotlin/sk/ainet/tinyllama/EagerNative.kt").toFile()
        if (!native.exists()) return
        val text = native.readText()
        assertContains(text, "if (fp32) QuantPolicy.DEQUANTIZE_TO_FP32 else QuantPolicy.NATIVE_OPTIMIZED")
        assertContains(text, "EAGER_NATIVE_FP32")
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
