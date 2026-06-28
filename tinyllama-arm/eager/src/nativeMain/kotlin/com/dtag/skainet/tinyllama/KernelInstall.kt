package com.dtag.skainet.tinyllama

/**
 * Register the platform's fast matmul kernels into the SKaiNET `KernelRegistry`
 * before any quantized `ops.matmul` runs.
 *
 * Kotlin/Native has no `ServiceLoader`, so the NEON `KernelProvider` from
 * `skainet-backend-native-cpu` must be registered by hand. This is `expect`
 * because that backend only ships `linuxArm64` (the board) — on `macosArm64`
 * the CPU ops go through Accelerate and there is nothing to register.
 *
 * - **linuxArm64**: `installNativeKernels()` → registers `NativeKnKernelProvider`
 *   (priority 100, NEON Q4_K/Q5_K/Q6_K/Q8_0/Q4_0). The scalar provider (priority 0)
 *   is registered by the linux CPU-ops factory and remains the per-format fallback.
 * - **macosArm64**: no-op (Accelerate-backed ops, registry kernels unused).
 */
internal expect fun installPlatformKernels()
