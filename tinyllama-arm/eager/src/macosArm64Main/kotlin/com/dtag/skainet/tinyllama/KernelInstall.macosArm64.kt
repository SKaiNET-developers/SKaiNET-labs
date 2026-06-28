package com.dtag.skainet.tinyllama

/**
 * No-op: on macosArm64 the CPU ops are Accelerate-backed (see
 * `PlatformCpuOpsFactory.apple.kt`), so the registry kernels are unused and
 * `skainet-backend-native-cpu` is not a dependency on this target.
 */
internal actual fun installPlatformKernels() {
    // intentionally empty
}
