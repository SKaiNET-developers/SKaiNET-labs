import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
}

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }
    linuxArm64 {
        binaries {
            executable {
                entryPoint = "com.dtag.skainet.tinyllama.main"
                baseName = "tinyllama-skainet"
            }
        }
        // skainet-backend-native-cpu wires libskainet_kernels.a (NEON) only into its OWN
        // binaries (binaries.all { linkerOpts }), which does not propagate to a consumer's
        // executable link — so a downstream consumer must add the archive itself or the
        // K/N link fails with "undefined symbol: skainet_q4k_matmul". Composite-only: the
        // archive lives in the sibling core build dir (built natively on the board, then
        // pulled to that path — see docs board-neon-kernels entry).
        if (providers.gradleProperty("useLocalSkainet").orNull == "true") {
            binaries.all {
                val a = rootProject.file(
                    "../SKaiNET/skainet-backends/skainet-backend-native-cpu/build/native/cmake-build-arm64/libskainet_kernels.a"
                )
                linkerOpts(a.absolutePath)
            }
        }
    }
    // Handy for local development on Apple Silicon. The SL2619 board target is linuxArm64.
    macosArm64 {
        binaries {
            executable {
                entryPoint = "com.dtag.skainet.tinyllama.main"
                baseName = "tinyllama-skainet"
            }
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(project.dependencies.platform(libs.skainet.bom))
            api(project(":model"))
            // Compact KLLama runtime (CpuAttentionBackend, GGUFTokenizer, LlamaRuntime) — used
            // by both the native board path and the JVM eager path.
            implementation(libs.skainet.transformers.runtime.kllama)
            implementation(libs.kotlinx.coroutines.core)
            implementation(libs.kotlinx.io.core)
            // KernelProfile (matmul-path timing) for the opt-in SKAINET_PROFILE breakdown.
            implementation(libs.skainet.backend.cpu)
        }
        jvmMain.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(project.dependencies.platform(libs.skainet.bom))
            // SIMD-accelerated CPU kernels, auto-discovered on supported JVM hosts.
            implementation(libs.skainet.backend.native.cpu)
        }
        // Board (Cortex-A55) NEON kernels. skainet-backend-native-cpu targets only
        // jvm/linuxX64/linuxArm64 — NOT macosArm64 — so it goes in linuxArm64Main, not
        // nativeMain. The K/N cinterop links libskainet_kernels.a (aarch64, NEON) at final
        // link; the provider is registered at startup via installPlatformKernels()
        // (expect/actual: real on linuxArm64, no-op on macosArm64 which uses Accelerate).
        val linuxArm64Main by getting {
            dependencies {
                implementation(project.dependencies.platform(libs.skainet.bom))
                implementation(libs.skainet.backend.native.cpu)
            }
        }
    }
}
