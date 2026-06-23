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
    mainClass.set("com.dtag.skainet.tinyllama.MainKt")
    applicationDefaultJvmArgs = listOf("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g", "-Xmx2g")
}

// Convenience task preserving the original UX: ./gradlew runJvm --args='eager --model Q4_K_M ...'
tasks.register<JavaExec>("runJvm") {
    description = "Run the unified TinyLlama benchmark CLI. Pass args after --args='...'."
    group = "application"
    mainClass.set("com.dtag.skainet.tinyllama.MainKt")
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs("--enable-preview", "--add-modules", "jdk.incubator.vector", "-XX:MaxDirectMemorySize=12g", "-Xmx2g")
    // Run from the repo root so relative paths (models/..., prompts/...) resolve.
    workingDir = rootProject.projectDir
}
