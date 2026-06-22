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
            implementation(libs.kotlinx.coroutines.core)
            implementation(libs.kotlinx.io.core)
        }
        jvmMain.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(project.dependencies.platform(libs.skainet.bom))
            // SIMD-accelerated CPU kernels, auto-discovered on supported JVM hosts.
            implementation(libs.skainet.backend.native.cpu)
        }
        nativeMain.dependencies {
            implementation(project.dependencies.platform(libs.skainet.transformers.bom))
            implementation(project.dependencies.platform(libs.skainet.bom))
            // Compact KLLama runtime (CpuAttentionBackend, GGUFTokenizer) for the board.
            implementation(libs.skainet.transformers.runtime.kllama)
        }
    }
}
