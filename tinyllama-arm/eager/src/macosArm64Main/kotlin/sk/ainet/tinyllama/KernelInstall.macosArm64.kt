package sk.ainet.tinyllama

import sk.ainet.exec.kernel.installNativeKernels

/**
 * SKaiNET 0.40.1 added a macosArm64 target with an embedded NEON archive to
 * `skainet-backend-native-cpu` (core 3acf18ae). This used to be a no-op because
 * Accelerate only overrides dense FP32 matmul — the actual Q4_K decode hot path fell
 * through to `ScalarKernelProvider` (see `PlatformCpuOpsFactory.apple.kt`). Verified
 * 2026-08-13: 0.52 -> 25.6 tok/s (40-tok, golden output unchanged) — see
 * perf/apple-neon-macos in docs/perf-history.csv.
 */
internal actual fun installPlatformKernels() {
    installNativeKernels()
}
