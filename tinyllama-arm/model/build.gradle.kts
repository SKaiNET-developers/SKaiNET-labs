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
    linuxArm64()
    macosArm64()

    sourceSets {
        commonMain.dependencies {
            // api so the BOM version constraints propagate to modules that consume :model.
            api(project.dependencies.platform(libs.skainet.transformers.bom))
            api(project.dependencies.platform(libs.skainet.bom))

            api(libs.skainet.transformers.core)
            api(libs.skainet.transformers.inference.llama)
            api(libs.skainet.backend.cpu)
            api(libs.skainet.io.core)
            api(libs.skainet.io.gguf)
            api(libs.skainet.lang.core)

            implementation(libs.kotlinx.coroutines.core)
            implementation(libs.kotlinx.io.core)
        }

        commonTest.dependencies {
            implementation(kotlin("test"))
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
