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
        google()
        mavenCentral()
    }
}

// Composite build to fix + verify SKaiNET-transformers against this repo. Opt-in:
// -PuseLocalSkainet=true substitutes sk.ainet.transformers:* with the sibling
// ../SKaiNET-transformers source checkout. sk.ainet.core:* stays from Maven (the bug is in
// transformers, and building ../SKaiNET's android/pipeline modules needs more toolchain).
if (providers.gradleProperty("useLocalSkainet").orNull == "true") {
    includeBuild("../SKaiNET-transformers") {
        // The TF modules publish via POM_ARTIFACT_ID (skainet-transformers-*), which Gradle's
        // composite substitution does NOT match against the project names — so map explicitly.
        dependencySubstitution {
            substitute(module("sk.ainet.transformers:skainet-transformers-core")).using(project(":llm-core"))
            substitute(module("sk.ainet.transformers:skainet-transformers-transformer-core")).using(project(":transformer-core"))
            substitute(module("sk.ainet.transformers:skainet-transformers-api")).using(project(":llm-api"))
            substitute(module("sk.ainet.transformers:skainet-transformers-inference-llama")).using(project(":llm-inference:llama"))
            substitute(module("sk.ainet.transformers:skainet-transformers-runtime-kllama")).using(project(":llm-runtime:kllama"))
        }
    }
}

rootProject.name = "skainet-tinyllama-iree"

include(":model", ":eager", ":export-hlo", ":iree", ":bench")

