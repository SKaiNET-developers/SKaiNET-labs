import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

// DIAGNOSTIC (perf/a2): the native-ffm Q4_K kernel (priority 100) is a serial C loop, so it
// outranks the parallel Panama kernel (50) and pins decode to ~1 core. Excluding native-cpu on
// the JVM host lets the parallelChunks Panama kernel use all cores. Toggle via -PexcludeNativeCpu.
if (providers.gradleProperty("excludeNativeCpu").orNull == "true") {
    configurations.all { exclude(group = "sk.ainet.core", module = "skainet-backend-native-cpu") }
}

dependencies {
    implementation(project(":model"))
    implementation(project(":eager"))
    implementation(project(":export-hlo"))
    implementation(project(":iree"))
    implementation(libs.kotlinx.coroutines.core)
}

application {
    mainClass.set("sk.ainet.tinyllama.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g", "-Xmx2g")
}

// Convenience task preserving the original UX: ./gradlew runJvm --args='eager --model Q4_K_M ...'
tasks.register<JavaExec>("runJvm") {
    description = "Run the unified TinyLlama benchmark CLI. Pass args after --args='...'."
    group = "application"
    mainClass.set("sk.ainet.tinyllama.MainKt")
    classpath = sourceSets["main"].runtimeClasspath
    // Heap defaults to 2g (right-sized for the packed NATIVE_OPTIMIZED path, perf/a1b);
    // override with -Pxmx=12g for the dense FP32 parity path which needs ~4.4 GB.
    val xmx = providers.gradleProperty("xmx").getOrElse("2g")
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g", "-Xmx$xmx")
    // Run from the repo root so relative paths (models/..., prompts/...) resolve.
    workingDir = rootProject.projectDir
}
