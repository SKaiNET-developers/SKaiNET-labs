import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

dependencies {
    // Orchestrates iree-compile (Docker) and iree-run-module (board, via adb).
    // No SKaiNET deps: it drives the scripts/ tooling and reports BenchmarkResult.
    api(project(":model"))
}
