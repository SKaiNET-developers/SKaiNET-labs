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
    includeBuild("../SKaiNET-transformers")
}

rootProject.name = "skainet-tinyllama-iree"

include(":model", ":eager", ":export-hlo", ":iree", ":bench")

