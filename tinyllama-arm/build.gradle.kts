import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("multiplatform") version "2.3.21"
}

val transformersVersion = "0.31.1"
val coreVersion = "0.31.2"

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
            implementation(project.dependencies.platform("sk.ainet.transformers:skainet-transformers-bom:$transformersVersion"))
            // The transformers BOM transitively pins sk.ainet.core:* to 0.31.0;
            // override to the latest core patch so the native CPU packed-quant path
            // gets the newest fixes.
            implementation(project.dependencies.platform("sk.ainet:skainet-bom:$coreVersion"))

            implementation("sk.ainet.transformers:skainet-transformers-core")
            implementation("sk.ainet.transformers:skainet-transformers-inference-llama")
            implementation("sk.ainet.transformers:skainet-transformers-runtime-kllama")

            implementation("sk.ainet.core:skainet-backend-cpu")
            implementation("sk.ainet.core:skainet-compile-dag")
            implementation("sk.ainet.core:skainet-compile-hlo")
            implementation("sk.ainet.core:skainet-io-core")
            implementation("sk.ainet.core:skainet-io-gguf")
            implementation("sk.ainet.core:skainet-lang-core")

            implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.9.0")
        }

        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain.dependencies {
            // Auto-discovered by SKaiNET on supported JVM hosts; harmless fallback elsewhere.
            implementation("sk.ainet.core:skainet-backend-native-cpu")
        }

        jvmTest.dependencies {
            implementation(kotlin("test-junit5"))
        }
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    maxHeapSize = "8g"
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
}

tasks.register<JavaExec>("runJvm") {
    description = "Run the JVM TinyLlama SKaiNET CLI. Pass args after --args='...'."
    group = "application"
    mainClass.set("com.dtag.skainet.tinyllama.MainKt")
    classpath = files(
        kotlin.jvm().compilations["main"].output.allOutputs,
        configurations["jvmRuntimeClasspath"],
    )
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
}

tasks.register<JavaExec>("exportStableHlo") {
    description = "Export the tiny Llama-shaped SKaiNET DSL graph to StableHLO MLIR."
    group = "verification"
    mainClass.set("com.dtag.skainet.tinyllama.MainKt")
    classpath = files(
        kotlin.jvm().compilations["main"].output.allOutputs,
        configurations["jvmRuntimeClasspath"],
    )
    args("export", "--out", layout.buildDirectory.file("stablehlo/tinyllama_step.mlir").get().asFile.absolutePath)
}

tasks.register<Exec>("compileIree") {
    description = "Compile exported StableHLO MLIR to an IREE vmfb. Requires iree-compile on PATH."
    group = "verification"
    dependsOn("exportStableHlo")
    commandLine(
        "bash",
        "scripts/compile-iree.sh",
        layout.buildDirectory.file("stablehlo/tinyllama_step.mlir").get().asFile.absolutePath,
        layout.buildDirectory.file("iree/tinyllama_step.vmfb").get().asFile.absolutePath,
    )
}
