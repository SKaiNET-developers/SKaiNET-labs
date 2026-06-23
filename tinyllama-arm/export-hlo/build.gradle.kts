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
    implementation(platform(libs.skainet.transformers.bom))
    implementation(platform(libs.skainet.bom))
    api(project(":model"))

    // DSL -> DAG -> StableHLO. These compiler modules are JVM-only.
    implementation(libs.skainet.compile.dag)
    implementation(libs.skainet.compile.hlo)
    implementation(libs.kotlinx.coroutines.core)
}

tasks.register<JavaExec>("exportStableHlo") {
    description = "Export the synthetic TinyLlama-shaped SKaiNET DSL graph to StableHLO MLIR."
    group = "verification"
    mainClass.set("com.dtag.skainet.tinyllama.ExportMainKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
    val out = rootProject.layout.buildDirectory.file("stablehlo/tinyllama_step.mlir").get().asFile.absolutePath
    args("--out", out)
}

tasks.register<JavaExec>("exportStableHloReal") {
    description = "Export the REAL TinyLlama graph (loaded GGUF weights) to StableHLO MLIR."
    group = "verification"
    mainClass.set("com.dtag.skainet.tinyllama.ExportMainKt")
    classpath = sourceSets["main"].runtimeClasspath
    // Large heap: densifying TinyLlama to FP32 for graph capture needs several GB.
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g", "-Xmx12g")
    // Run from the repo root so a relative --model (models/...gguf) resolves.
    workingDir = rootProject.projectDir
    val out = rootProject.layout.buildDirectory.file("stablehlo/tinyllama_full.mlir").get().asFile.absolutePath
    // gradleProperty (not findProperty) so -Pmodel doesn't collide with the :model subproject.
    val model = providers.gradleProperty("model").getOrElse("Q4_K_M")
    val ctx = providers.gradleProperty("ctx").getOrElse("32")
    args("--out", out, "--model", model, "--ctx", ctx)
}
