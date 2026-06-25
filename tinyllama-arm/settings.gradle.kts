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

// SKaiNET is consumed from published Maven Central releases only — versions pinned in
// gradle/libs.versions.toml (transformers + core BOMs). The previous opt-in composite build
// (-PuseLocalSkainet) against sibling ../SKaiNET-transformers / ../SKaiNET source checkouts has
// been removed now that the required fixes are released (transformers 0.32.0, core 0.32.2).

rootProject.name = "skainet-tinyllama-iree"

include(":model", ":eager", ":export-hlo", ":iree", ":bench")

