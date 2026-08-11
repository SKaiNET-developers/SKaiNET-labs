package sk.ainet.tinyllama

import sk.ainet.exec.kernel.installNativeKernels

/**
 * Register the K/N NEON kernel provider (`NativeKnKernelProvider`, priority 100)
 * backed by the cinterop static archive `libskainet_kernels.a`. The linux CPU-ops
 * factory separately pins `ScalarKernelProvider` (priority 0) as the per-format
 * fallback, so quant types without a C kernel still resolve.
 */
internal actual fun installPlatformKernels() {
    installNativeKernels()
}
