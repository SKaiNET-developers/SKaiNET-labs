/*
 * Microbenchmark: SKaiNET in-house NEON kernels vs Arm KleidiAI micro-kernels
 * at TinyLlama-1.1B decode GEMV shapes (m=1), single thread.
 *
 * Fairness rules:
 *  - Weight packing is done ONCE outside the timed loop for both sides
 *    (SKaiNET weights arrive pre-packed from GGUF; KleidiAI RHS packing is a
 *    load-time step in a real integration).
 *  - Per-token activation quantization IS timed for both sides: SKaiNET's
 *    q4k/q8_0 kernels quantize activations to int8 internally per call;
 *    KleidiAI does it in kai_run_lhs_quant_pack_qai8dxp_f32, so that call is
 *    inside the timed region.
 *  - Weights rotate across enough copies to exceed the host's cache
 *    hierarchy (~96 MB working set), modeling real decode where every layer
 *    weight streams from DRAM each token.
 *  - Compiled with the board's exact flags: -O3 -ffast-math
 *    -march=armv8.2-a+fp16+dotprod  (Cortex-A55 parity: dotprod, NO i8mm).
 */
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "skainet_kernels.h"

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

#define BL 32 /* KleidiAI quantization block length, matches Q4_0/Q8_0 granularity */

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static inline uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill_random_bytes(uint8_t* p, size_t nbytes) {
    for (size_t i = 0; i < nbytes; ++i) p[i] = (uint8_t)rnd();
}

/* ---- SKaiNET ggml-block weight generators (sane fp16 scales, random codes) */

static void gen_q4k(uint8_t* w, int32_t n, int32_t k) {
    const int32_t nblk = k / 256;
    fill_random_bytes(w, (size_t)nblk * n * 144);
    for (int32_t b = 0; b < nblk; ++b)
        for (int32_t o = 0; o < n; ++o) {
            uint8_t* blk = w + ((size_t)b * n + o) * 144;
            blk[0] = 0x00; blk[1] = 0x3C; /* d = 1.0 (fp16) */
            blk[2] = 0x00; blk[3] = 0x2C; /* dmin small */
        }
}

static void gen_q6k(uint8_t* w, int32_t n, int32_t k) {
    const int32_t nblk = k / 256;
    fill_random_bytes(w, (size_t)nblk * n * 210);
    for (int32_t b = 0; b < nblk; ++b)
        for (int32_t o = 0; o < n; ++o) {
            uint8_t* blk = w + ((size_t)b * n + o) * 210;
            blk[208] = 0x00; blk[209] = 0x3C; /* d = 1.0 (fp16) */
        }
}

static void gen_q8_0(uint8_t* w, int32_t n, int32_t k) {
    const int32_t nblk = k / 32;
    fill_random_bytes(w, (size_t)nblk * n * 34);
    for (int32_t b = 0; b < nblk; ++b)
        for (int32_t o = 0; o < n; ++o) {
            uint8_t* blk = w + ((size_t)b * n + o) * 34;
            blk[0] = 0x00; blk[1] = 0x3C; /* d = 1.0 (fp16) */
        }
}

/* ---- KleidiAI ukernel table ---- */

typedef struct {
    const char* name;
    size_t (*get_mr)(void);
    size_t (*get_nr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_offset)(size_t m_idx, size_t k);
    size_t (*get_rhs_packed_offset)(size_t n_idx, size_t k, size_t bl);
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed, float* dst,
        size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max);
} kai_gemv;

#define KAI_VARIANT(suffix)                                          \
    {                                                                \
        #suffix,                                                     \
        kai_get_mr_matmul_clamp_f32_##suffix,                        \
        kai_get_nr_matmul_clamp_f32_##suffix,                        \
        kai_get_kr_matmul_clamp_f32_##suffix,                        \
        kai_get_sr_matmul_clamp_f32_##suffix,                        \
        kai_get_lhs_packed_offset_matmul_clamp_f32_##suffix,         \
        kai_get_rhs_packed_offset_matmul_clamp_f32_##suffix,         \
        kai_run_matmul_clamp_f32_##suffix,                           \
    }

static const kai_gemv KAI_GEMVS[] = {
    KAI_VARIANT(qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod),
    KAI_VARIANT(qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod),
    KAI_VARIANT(qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod),
};

/* ---- benchmark core ---- */

typedef struct {
    const char* name;
    int32_t n, k;
} shape_t;

/* TinyLlama-1.1B decode (hidden 2048, GQA kv 256, ffn 5632, vocab 32000) */
static const shape_t SHAPES[] = {
    {"attn q/o   2048x2048", 2048, 2048},
    {"attn kv     256x2048", 256, 2048},
    {"ffn gate/up 5632x2048", 5632, 2048},
    {"ffn down   2048x5632", 2048, 5632},
    {"lm_head   32000x2048", 32000, 2048},
};

#define TARGET_WS (96ull * 1024 * 1024) /* cache-busting working set */
#define MAX_COPIES 8

static int n_copies(size_t weight_bytes) {
    size_t c = (TARGET_WS + weight_bytes - 1) / weight_bytes;
    if (c < 1) c = 1;
    if (c > MAX_COPIES) c = MAX_COPIES;
    return (int)c;
}

/* run fn(ctx, copy_idx) repeatedly; returns µs per call */
typedef void (*bench_fn)(void* ctx, int copy);
static double bench(bench_fn fn, void* ctx, int copies) {
    for (int i = 0; i < 3; ++i) fn(ctx, i % copies); /* warmup */
    int iters = 8;
    for (;;) {
        double t0 = now_s();
        for (int i = 0; i < iters; ++i) fn(ctx, i % copies);
        double dt = now_s() - t0;
        if (dt > 0.35) return dt * 1e6 / iters;
        iters = (int)(iters * (0.5 / (dt > 1e-4 ? dt : 1e-4))) + 1;
    }
}

/* SKaiNET kernel benches */
typedef struct {
    void (*kernel)(const float*, int32_t, const uint8_t*, int32_t, int32_t, int32_t, float*, int32_t);
    const float* input;
    uint8_t** weights;
    int32_t n, k;
    float* output;
} sk_ctx;

static void sk_run(void* p, int copy) {
    sk_ctx* c = (sk_ctx*)p;
    c->kernel(c->input, 0, c->weights[copy], 0, c->k, c->n, c->output, 0);
}

/* KleidiAI benches: timed region = LHS quant+pack + matmul */
typedef struct {
    const kai_gemv* uk;
    const float* input;
    uint8_t* lhs_packed;
    uint8_t** rhs_packed;
    int32_t n, k;
    float* output;
} kai_ctx;

static void kai_run(void* p, int copy) {
    kai_ctx* c = (kai_ctx*)p;
    const size_t mr = c->uk->get_mr(), kr = c->uk->get_kr(), sr = c->uk->get_sr();
    kai_run_lhs_quant_pack_qai8dxp_f32(
        1, (size_t)c->k, mr, kr, sr, 0, c->input, (size_t)c->k * sizeof(float), c->lhs_packed);
    const void* lhs = c->lhs_packed + c->uk->get_lhs_packed_offset(0, (size_t)c->k);
    const void* rhs = c->rhs_packed[copy] + c->uk->get_rhs_packed_offset(0, (size_t)c->k, BL);
    c->uk->run_matmul(
        1, (size_t)c->n, (size_t)c->k, BL, lhs, rhs, c->output, (size_t)c->n * sizeof(float), sizeof(float), -FLT_MAX,
        FLT_MAX);
}

static void report(const char* kernel, const shape_t* s, double us, size_t wbytes) {
    double gbs = (double)wbytes / (us * 1e-6) / 1e9;
    double gflops = 2.0 * s->n * s->k / (us * 1e-6) / 1e9;
    printf("%-22s  %-46s %9.1f us  %7.2f GB/s  %7.2f GFLOP/s\n", s->name, kernel, us, gbs, gflops);
}

int main(void) {
    printf("m=1 GEMV, single thread, weights rotated over >=96MB working set\n");
    printf("timed per call: activation int8 quantization + matmul (both sides)\n\n");

    const size_t n_shapes = sizeof(SHAPES) / sizeof(SHAPES[0]);
    for (size_t si = 0; si < n_shapes; ++si) {
        const shape_t* s = &SHAPES[si];
        const int32_t n = s->n, k = s->k;

        float* input = malloc((size_t)k > (size_t)n ? (size_t)k * 4 : (size_t)n * 4);
        for (int32_t i = 0; i < k; ++i) input[i] = (float)(rnd() & 0xFF) / 256.0f - 0.5f;
        float* output = malloc((size_t)n * sizeof(float));

        /* --- SKaiNET kernels --- */
        struct {
            const char* name;
            void (*kernel)(const float*, int32_t, const uint8_t*, int32_t, int32_t, int32_t, float*, int32_t);
            size_t bytes;
            void (*gen)(uint8_t*, int32_t, int32_t);
        } sks[] = {
            {"skainet q4k (dotprod)", skainet_q4k_matmul, (size_t)(k / 256) * n * 144, gen_q4k},
            {"skainet q6k", skainet_q6k_matmul, (size_t)(k / 256) * n * 210, gen_q6k},
            {"skainet q8_0 (dotprod)", skainet_q8_0_matmul, (size_t)(k / 32) * n * 34, gen_q8_0},
        };
        for (size_t ki = 0; ki < sizeof(sks) / sizeof(sks[0]); ++ki) {
            int copies = n_copies(sks[ki].bytes);
            uint8_t* weights[MAX_COPIES];
            for (int c = 0; c < copies; ++c) {
                weights[c] = malloc(sks[ki].bytes);
                sks[ki].gen(weights[c], n, k);
            }
            sk_ctx ctx = {sks[ki].kernel, input, weights, n, k, output};
            double us = bench(sk_run, &ctx, copies);
            report(sks[ki].name, s, us, sks[ki].bytes);
            for (int c = 0; c < copies; ++c) free(weights[c]);
        }

        /* --- KleidiAI GEMV variants --- */
        for (size_t ki = 0; ki < sizeof(KAI_GEMVS) / sizeof(KAI_GEMVS[0]); ++ki) {
            const kai_gemv* uk = &KAI_GEMVS[ki];
            const size_t nr = uk->get_nr(), kr = uk->get_kr(), sr = uk->get_sr(), mr = uk->get_mr();

            /* native RHS: nxk 4-bit codes + bf16 scale per 32-block */
            const size_t rhs_stride = (size_t)k / 2;
            const size_t scale_stride = (size_t)(k / BL) * sizeof(uint16_t);
            uint8_t* rhs_native = malloc((size_t)n * rhs_stride);
            uint16_t* rhs_scales = malloc((size_t)n * scale_stride);
            fill_random_bytes(rhs_native, (size_t)n * rhs_stride);
            for (size_t i = 0; i < (size_t)n * (k / BL); ++i) rhs_scales[i] = 0x3F80; /* bf16 1.0 */

            const size_t rhs_packed_size =
                kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0((size_t)n, (size_t)k, nr, kr, sr, BL, kai_dt_bf16);
            int copies = n_copies(rhs_packed_size);
            uint8_t* rhs_packed[MAX_COPIES];
            struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params = {
                .lhs_zero_point = 1, .rhs_zero_point = 8, .scale_dt = kai_dt_bf16};
            for (int c = 0; c < copies; ++c) {
                rhs_packed[c] = malloc(rhs_packed_size);
                kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                    1, (size_t)n, (size_t)k, nr, kr, sr, BL, rhs_native, rhs_stride, NULL, rhs_scales, scale_stride,
                    rhs_packed[c], 0, &params);
            }

            const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(1, (size_t)k, mr, kr, sr);
            uint8_t* lhs_packed = malloc(lhs_packed_size);

            kai_ctx ctx = {uk, input, lhs_packed, rhs_packed, n, k, output};
            double us = bench(kai_run, &ctx, copies);
            char label[80];
            snprintf(label, sizeof label, "kleidiai %s", uk->name);
            report(label, s, us, rhs_packed_size);

            for (int c = 0; c < copies; ++c) free(rhs_packed[c]);
            free(lhs_packed);
            free(rhs_native);
            free(rhs_scales);
        }
        printf("\n");
        free(input);
        free(output);
    }
    return 0;
}
