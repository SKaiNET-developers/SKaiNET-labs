package sk.ainet.tinyllama

import java.io.File

/** Result of running an external process: exit code and captured stdout+stderr. */
data class ProcResult(val exitCode: Int, val output: String) {
    fun requireSuccess(what: String): ProcResult {
        check(exitCode == 0) { "$what failed (exit $exitCode):\n$output" }
        return this
    }
}

/** Run [command] (optionally with extra [env] and [workingDir]), tee output to stdout, and capture it. */
fun runProcess(
    command: List<String>,
    env: Map<String, String> = emptyMap(),
    workingDir: File? = null,
): ProcResult {
    val pb = ProcessBuilder(command).redirectErrorStream(true)
    if (workingDir != null) pb.directory(workingDir)
    pb.environment().putAll(env)
    val proc = pb.start()
    val captured = StringBuilder()
    proc.inputStream.bufferedReader().useLines { lines ->
        lines.forEach { line ->
            println(line)
            captured.appendLine(line)
        }
    }
    val code = proc.waitFor()
    return ProcResult(code, captured.toString())
}
