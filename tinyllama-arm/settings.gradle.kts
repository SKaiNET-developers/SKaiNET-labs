pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        // NOTE: do NOT add mavenLocal() here. ~/.m2 can hold POM-only (no Gradle Module Metadata)
        // copies of multiplatform libs like kotlinx-io-core — Gradle then resolves their JVM `.jar`
        // variant for *native* compilations (no klib), breaking the linuxArm64/macosArm64 build with
        // spurious "unresolved reference" errors. Consume unreleased SKaiNET/transformers changes via
        // the composite (-PuseLocalSkainet) below, not mavenLocal.
        google()
        mavenCentral()
    }
}

// SKaiNET is consumed from published Maven Central releases by default — versions pinned in
// gradle/libs.versions.toml (transformers + core BOMs). That is the release/CI path.
//
// Dev-only opt-in: run with -PuseLocalSkainet=true to substitute sk.ainet.transformers:* and
// (transitively) sk.ainet.core:* with the sibling source checkouts, for iterating cross-repo
// changes (e.g. board memory/perf work) against unreleased fixes before they're published.
// transformers' own settings then chains includeBuild("../../SKaiNET") under the same flag.
//
// EXPLICIT dependencySubstitution is required: Gradle's automatic composite substitution matches
// included-build projects by `group:projectName`, but the transformers projects are named
// `:llm-runtime:kllama` etc. while they PUBLISH as `skainet-transformers-runtime-kllama`
// (POM_ARTIFACT_ID). The names never match, so auto-substitution silently falls through to the
// Maven release — including on native targets, which is why local fused-load/board work was
// untestable. We map each published coordinate to its project path by hand. The BOM stays on Maven
// (constraints only; the substituted projects override versions anyway).
if (providers.gradleProperty("useLocalSkainet").orNull == "true") {
    includeBuild("../../SKaiNET-transformers") {
        dependencySubstitution {
            substitute(module("sk.ainet.transformers:skainet-transformers-api"))
                .using(project(":llm-api"))
            substitute(module("sk.ainet.transformers:skainet-transformers-transformer-core"))
                .using(project(":transformer-core"))
            substitute(module("sk.ainet.transformers:skainet-transformers-core"))
                .using(project(":llm-core"))
            substitute(module("sk.ainet.transformers:skainet-transformers-agent"))
                .using(project(":llm-agent"))
            substitute(module("sk.ainet.transformers:skainet-transformers-providers"))
                .using(project(":llm-providers"))
            substitute(module("sk.ainet.transformers:skainet-transformers-inference-llama"))
                .using(project(":llm-inference:llama"))
            substitute(module("sk.ainet.transformers:skainet-transformers-inference-qwen"))
                .using(project(":llm-inference:qwen"))
            substitute(module("sk.ainet.transformers:skainet-transformers-runtime-kllama"))
                .using(project(":llm-runtime:kllama"))
            substitute(module("sk.ainet.transformers:skainet-transformers-performance"))
                .using(project(":llm-performance"))
        }
    }
}

rootProject.name = "skainet-tinyllama-iree"

include(":model", ":eager", ":export-hlo", ":iree", ":bench")

