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
    description = "Export the TinyLlama-shaped SKaiNET DSL graph to StableHLO MLIR."
    group = "verification"
    mainClass.set("com.dtag.skainet.tinyllama.ExportMainKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g")
    val out = rootProject.layout.buildDirectory.file("stablehlo/tinyllama_step.mlir").get().asFile.absolutePath
    args("--out", out)
}
