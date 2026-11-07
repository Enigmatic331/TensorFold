// DeepSeek-V4.1-Flash's own checkpoint format on GB10 (sm_121): the linears and experts of
// deepseek-ai/DeepSeek-V4.1-Flash-1017 as its reference inference code computes them (inference/model.py `linear`,
// kernel.py `act_quant`, `fp8_gemm`, `fp4_gemm`; MIT, Copyright (c) 2023 DeepSeek). Written for this port from that
// arithmetic, no code copied:
//   - an activation is rounded to bf16, then quantized to FP8 E4M3 with one power-of-two (UE8M0) scale per 32 values
//     (amax floored at 1e-4, scale 2^ceil(log2(amax / 448)));
//   - FP8 weights carry one UE8M0 scale per 32 x 32 block, MXFP4 weights (E2M1 pairs, low nibble first) one per 32
//     values of a row; every E2M1 value is exact in E4M3, so both run on the same e4m3 tensor-core MMA;
//   - each 32-wide K block's MMA sum is scaled by both scales and added to an fp32 accumulator, K blocks in order.
// A row's result never depends on the other rows of a launch (one row's MMA lanes never mix with another's) and every
// layer's K split is fixed by its shape, so drafted, concurrent and resumed rows get the bits of a row alone.
// NVIDIA's NVFP4 release of the same checkpoint (nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) keeps all of that but for the
// routed experts of its 40 target layers: E2M1 weights with an E4M3 scale per 16 values and a global scale, and E2M1
// activations of each expert's own input_scale, on the block-scaled FP4 MMA (the TF_DSN_FMT_NV paths: see mma_fp4 and
// tf_dsn_nv_quant).
// Raw device pointers, the caller's stream, extern "C" names (tf_dsn_*).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

namespace {

constexpr int kBlk = 32;           // the scale block (fp8 and fp4, weights and activations)
constexpr float kFp8Max = 448.0f;  // E4M3's largest finite value

// dtype codes of the engine's EXL3 calls (exl3_linear.DType): f16, bf16, f32
__device__ __forceinline__ float load_x(const void* x, int dtype, size_t i) {
    if (dtype == 1) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[i]);
    if (dtype == 0) return __half2float(reinterpret_cast<const __half*>(x)[i]);
    return reinterpret_cast<const float*>(x)[i];
}

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// kernel.py fast_round_scale: 2^ceil(log2(amax * (1/448))) from the fp32 bits (exponent, plus one if any mantissa bit)
__device__ __forceinline__ float round_scale(float amax) {
    const float v = amax * (1.0f / kFp8Max);
    const uint32_t b = __float_as_uint(v);
    const int e = int((b >> 23) & 0xff) - 127 + ((b & 0x7fffff) ? 1 : 0);
    return __uint_as_float(uint32_t(e + 127) << 23);
}

__device__ __forceinline__ float ue8m0(uint8_t e) { return __uint_as_float(uint32_t(e) << 23); }

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    return uint8_t(__nv_cvt_float_to_fp8(fminf(fmaxf(v, -kFp8Max), kFp8Max), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float from_e4m3(uint8_t q) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
    return __half2float(__half(h));
}

// E2M1 code -> E4M3 byte: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (convert.py FP4_TABLE), all exact in E4M3
__constant__ uint8_t kE2m1ToE4m3[16] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
                                        0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc};

// two packed bytes (four E2M1 values, low nibble first) -> four E4M3 bytes, first value in the low byte
__device__ __forceinline__ uint32_t fp4x4_to_e4m3(uint16_t p) {
    return uint32_t(kE2m1ToE4m3[p & 0xf]) | uint32_t(kE2m1ToE4m3[(p >> 4) & 0xf]) << 8 |
           uint32_t(kE2m1ToE4m3[(p >> 8) & 0xf]) << 16 | uint32_t(kE2m1ToE4m3[(p >> 12) & 0xf]) << 24;
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// NVFP4 x NVFP4 on the block-scaled MMA (sm_121a): 64 inputs, an E4M3 scale per 16 of them on both sides, the sums
// added to d in place. TensorFold's mma4.cuh (ashhart/TensorFold), whose scale registers are read as there: sa a word of
// four scales, lanes t 0 and 1 giving A rows g and g + 8 (t 2, 3 are not read); sb the four of B row g.
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                        uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    const uint16_t z = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z),
          "h"(z));
#else
    __trap();
#endif
}

// A weight row's 64-input step scales from its two per-32 bytes: NVFP4's per-16 E4M3 scales come in equal pairs here
// (the converter kept MXFP4's power-of-two blocks; tf-dsv41-rank-cache refuses a checkpoint where a pair differs).
__device__ __forceinline__ uint32_t nv_scales(uint32_t lo, uint32_t hi) {
    return (lo | (lo << 8)) | ((hi | (hi << 8)) << 16);
}

// |v| to the nearest E2M1 magnitude code (ties to the even code), saturating at 6, with v's sign (nvfp4q.cuh).
__device__ __forceinline__ uint32_t nv_e2m1(float v) {
    const float a = fabsf(v);
    uint32_t c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return (v < 0.0f && c != 0) ? (c | 8u) : c;
}

__device__ __forceinline__ void store_y(void* y, int dtype, size_t i, float v) {
    if (dtype == 1) reinterpret_cast<__nv_bfloat16*>(y)[i] = __float2bfloat16_rn(v);
    else if (dtype == 0) reinterpret_cast<__half*>(y)[i] = __float2half_rn(v);
    else reinterpret_cast<float*>(y)[i] = v;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// act_quant: x [M, K] (row stride ldx, dtype f16/bf16/f32) -> q [M, K] E4M3 (row stride ldq) and s [M, K/32] fp32
// powers of two (row stride lds); kernel.py act_quant(x, 32, "ue8m0") of x.to(bf16). With `fake` (non-null) the
// dequantized values go to fake [M, K] bf16 instead (act_quant(..., inplace=True), wo_a's input). A warp a 32-group.
extern "C" __global__ void tf_dsn_act_quant_kernel(const void* __restrict__ x, int x_dtype, int64_t ldx, int M, int K,
                                                   uint8_t* __restrict__ q, int64_t ldq, float* __restrict__ s,
                                                   int64_t lds, __nv_bfloat16* __restrict__ fake, int64_t ldf) {
    const int groups = K / kBlk;
    const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= int64_t(M) * groups) return;
    const int m = int(warp / groups), g = int(warp % groups);
    const int k = g * kBlk + lane;
    const float v = bf16r(load_x(x, x_dtype, size_t(m) * ldx + k));
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    const uint8_t c = to_e4m3(v / sc);
    if (fake) {
        fake[size_t(m) * ldf + k] = __float2bfloat16_rn(from_e4m3(c) * sc);
    } else {
        q[size_t(m) * ldq + k] = c;
        if (lane == 0) s[size_t(m) * lds + g] = sc;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The MX GEMM: y [M, N] = sum over K blocks kb of (q[m, kb] . w[n, kb]) * s[m, kb] * ws(n, kb).
// A CTA is 4 warps; a warp owns 32 output columns (four m16n8 tiles) and MT m16 row tiles, over its K split's blocks.
// Within a 32-wide block the K order is permuted the same way for both operands (each thread reads 8 contiguous bytes
// of a row: its 4 low values feed the MMA's k = 4t.., its 4 high ones k = 16 + 4t..), a fixed choice, so a dot product
// is the same sum for every launch.
// Weights: fmt 0, FP8 [N, K] bytes (row stride ldw) with UE8M0 scales [N/32, K/32] (row stride ldws); fmt 1, MXFP4
// [N, K/2] packed (row stride ldw bytes) with UE8M0 scales [N, K/32] (row stride ldws).
// Rows: q [M, K] E4M3 (row stride ldq) with s [M, K/32] fp32 (row stride lds); `rows` (optional) maps the launch's row
// i to the activation row rows[i] (an expert's gathered rows).
// Split K: blockIdx.y is the split (of `splits`, each `kblocks` K blocks); with splits > 1 the partial goes to
// z [splits, M, N] fp32 and tf_dsn_reduce sums it in split order; else the epilogue writes y directly.
struct MxArgs {
    const uint8_t* q;
    const float* s;
    const int* rows;
    int64_t ldq, lds;
    const uint8_t* w;
    const uint8_t* ws;
    int64_t ldw, ldws;
    int fmt;
    int M, N, K;
    int splits, kblocks;
    float* z;
    void* y;
    int64_t ldy;
    int y_dtype;
    int accumulate;  // y += result (fp32 y only)
};

template <int MT>
__device__ __forceinline__ void mx_tile(const MxArgs& a, int n0, int m0, int split, float (&acc)[MT][4][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kb0 = split * a.kblocks;
    const int kb1 = min(kb0 + a.kblocks, a.K / kBlk);
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    // this thread's activation rows (g and g + 8 of each m16 tile), clamped (rows past M read row 0 and are not stored)
    int arow[MT][2];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int r = m0 + i * 16 + g + h * 8;
            r = r < a.M ? r : 0;
            arow[i][h] = a.rows ? a.rows[r] : r;
        }
#pragma unroll 2
    for (int kb = kb0; kb < kb1; kb++) {
        const int k = kb * kBlk + t * 8;
        // B: the four n8 tiles' rows n0 + 8j + g, 8 values from k (FP8: 8 bytes; FP4: 4 packed bytes)
        uint32_t b[4][2];
        float wsc[4][2];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + j * 8 + g;
            if (a.fmt == 0) {
                const uint2 v = *reinterpret_cast<const uint2*>(a.w + size_t(n) * a.ldw + k);
                b[j][0] = v.x;
                b[j][1] = v.y;
            } else {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(a.w + size_t(n) * a.ldw + k / 2);
                b[j][0] = fp4x4_to_e4m3(uint16_t(v & 0xffff));
                b[j][1] = fp4x4_to_e4m3(uint16_t(v >> 16));
            }
            // the C fragment's columns n0 + 8j + 2t, + 1: their weight scales for this block
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int nc = n0 + j * 8 + 2 * t + c;
                wsc[j][c] = a.fmt == 0 ? ue8m0(a.ws[size_t(nc / kBlk) * a.ldws + kb]) : ue8m0(a.ws[size_t(nc) * a.ldws + kb]);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; i++) {
            uint32_t af[4];
            const uint2 lo = *reinterpret_cast<const uint2*>(a.q + size_t(arow[i][0]) * a.ldq + k);
            const uint2 hi = *reinterpret_cast<const uint2*>(a.q + size_t(arow[i][1]) * a.ldq + k);
            af[0] = lo.x;  // row g, k 4t..
            af[1] = hi.x;  // row g + 8
            af[2] = lo.y;  // row g, k 16 + 4t..
            af[3] = hi.y;
            const float sa0 = a.s[size_t(arow[i][0]) * a.lds + kb];
            const float sa1 = a.s[size_t(arow[i][1]) * a.lds + kb];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float d[4];
                mma_e4m3(d, af, b[j]);
                // kernel.py: accum += C_local * scale_a * scale_b (powers of two: exact products)
                acc[i][j][0] += d[0] * sa0 * wsc[j][0];
                acc[i][j][1] += d[1] * sa0 * wsc[j][1];
                acc[i][j][2] += d[2] * sa1 * wsc[j][0];
                acc[i][j][3] += d[3] * sa1 * wsc[j][1];
            }
        }
    }
}

template <int MT>
__device__ __forceinline__ void mx_gemm_kernel(MxArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int split = blockIdx.y;
    for (int m0 = blockIdx.z * MT * 16; m0 < a.M; m0 += gridDim.z * MT * 16) {
        float acc[MT][4][4];
        mx_tile<MT>(a, n0, m0, split, acc);
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = m0 + i * 16 + g + (c >> 1) * 8;
                    const int n = n0 + j * 8 + 2 * t + (c & 1);
                    if (m >= a.M) continue;
                    const float v = acc[i][j][c];
                    if (a.splits > 1) {
                        a.z[(size_t(split) * a.M + m) * a.N + n] = v;
                    } else if (a.accumulate) {
                        float* yp = reinterpret_cast<float*>(a.y) + size_t(m) * a.ldy + n;
                        *yp += v;
                    } else {
                        store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
                    }
                }
    }
}

// the split partials z [splits, M, N] summed in split order into y (dtype, row stride ldy), or added to fp32 y
extern "C" __global__ void tf_dsn_reduce_kernel(const float* __restrict__ z, int splits, int M, int N, void* y,
                                                int64_t ldy, int y_dtype, int accumulate) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(M) * N) return;
    float v = z[i];
    for (int s = 1; s < splits; s++) v += z[size_t(s) * M * N + i];
    const int m = int(i / N), n = int(i % N);
    if (accumulate) reinterpret_cast<float*>(y)[size_t(m) * ldy + n] += v;
    else store_y(y, y_dtype, size_t(m) * ldy + n, v);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm1_kernel(MxArgs a) { mx_gemm_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm2_kernel(MxArgs a) { mx_gemm_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm4_kernel(MxArgs a) { mx_gemm_kernel<4>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_mx_gemm8_kernel(MxArgs a) { mx_gemm_kernel<8>(a); }

// ---------------------------------------------------------------------------------------------------------------------
// Grouped experts (MoE). The routed picks pick [R, S] (expert ids; >= E or < 0: not this node's) with weights
// wts [R, S] fp32 are grouped by expert: count, offsets, and each expert's (row, slot) entries.
//   tf_dsn_group_count: cnt[E] (zeroed by the caller) += picks
//   tf_dsn_group_scan:  off[E + 1] exclusive prefix, act[] the experts with a pick in id order, nact[0] their count
//   tf_dsn_group_place: ent[off[e] + i] = r * S + s and erow[off[e] + i] = r (in no particular order within an expert:
//                       a row's arithmetic is its own, so the order changes no bits)
extern "C" __global__ void tf_dsn_group_count_kernel(const int* __restrict__ pick, int RS, int E, int* __restrict__ cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RS) return;
    const int e = pick[i];
    if (e >= 0 && e < E) atomicAdd(cnt + e, 1);
}

extern "C" __global__ void tf_dsn_group_scan_kernel(const int* __restrict__ cnt, int E, int* __restrict__ off,
                                                    int* __restrict__ fill, int* __restrict__ act, int* __restrict__ nact) {
    // one thread: E is a few hundred
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int o = 0, n = 0;
    for (int e = 0; e < E; e++) {
        off[e] = o;
        fill[e] = 0;
        if (cnt[e]) act[n++] = e;
        o += cnt[e];
    }
    off[E] = o;
    nact[0] = n;
}

extern "C" __global__ void tf_dsn_group_place_kernel(const int* __restrict__ pick, int RS, int S, int E,
                                                     const int* __restrict__ off, int* __restrict__ fill,
                                                     int* __restrict__ ent, int* __restrict__ erow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= RS) return;
    const int e = pick[i];
    if (e < 0 || e >= E) return;
    const int at = off[e] + atomicAdd(fill + e, 1);
    ent[at] = i;
    erow[at] = i / S;
}

// Gate / up of the grouped experts with the SwiGLU epilogue (model.py Expert.forward): for expert act[blockIdx.y]'s
// entries and a 32-column tile of the node's pack of its intermediate (its main blocks, then its rest block: tile t is
// the pack's rows and the output columns 32 t ..), g = bf16(x . w1), u = bf16(x . w3), then
// h = bf16(silu(min(g, limit)) * clamp(u, -limit, limit) * wts[entry]) (wts null: the shared expert, no weight) into
// h [entry, out column] (row stride ldh). x is quantized already (q, s: one row per routed row, erow[] of an entry).
// w1 / w3 / their scales: an expert e's tensors at w1[e] etc. (pointer tables), so experts may sit anywhere.
struct GuArgs {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const int* ent;     // [entries]: r * S + s
    const int* erow;    // [entries]: r, the entry's activation row
    const int* off;     // [E + 1]
    const int* act;     // [E]: active experts
    const int* nact;    // [1]
    const uint64_t* w1; // [E] pointers: MXFP4 [I, D/2] (fmt 1) or FP8 [I, D] (fmt 0)
    const uint64_t* w3;
    const uint64_t* s1; // [E] pointers: their UE8M0 scales
    const uint64_t* s3;
    int64_t ldw, ldws;  // the routed experts' strides (fmt)
    int fmt;
    int D;
    int shared;         // the FP8 expert's id (the shared expert), -1: none
    int64_t sldw, sldws; // its strides
    const int* owner;   // optional [E]: the last rest_tiles tiles (the rest block) only for experts with owner[e] != 0
    int n_tiles, rest_tiles;
    const float* wts;   // [R * S] or null
    float limit;
    __nv_bfloat16* h;   // the pack: main columns [P, ldh] (P entries), then with rest tiles their columns [P, rest]
    int64_t ldh;
    int P;              // the entries (r * S), where the rest columns start (P * ldh elements in)
    // split K (decode): splits > 1 writes each split's g and u partials to z [splits, 2, P, n_tiles * 32] fp32 and
    // tf_dsn_gu_finish sums them in split order and applies the epilogue; splits 1: the epilogue here
    int splits, kblocks;
    float* z;
    int early_exit;     // Track C (TF_DS_EXPERT_EXIT): the decode kernel's rows past the active experts leave at once
    // fmt 2, NVFP4: the routed entries' codes and E4M3 scales (tf_dsn_nv_quant, a row an entry), each expert's gate and
    // up alpha (its input_scale * weight_scale_2); s1 / s3 hold each weight row's scale per 32 (nv_scales)
    const uint8_t* nq;
    const uint8_t* ns;
    int64_t ldnq, ldns;
    const float* a1;
    const float* a3;
};

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }

// gate / up's epilogue for one (entry, column): g and u (fp32 sums) into the pack
__device__ __forceinline__ void gu_store(const GuArgs& a, int entry, int col, float g_acc, float u_acc) {
    // fp4_gemm / fp8_gemm write bf16; Expert.forward takes .float() of them
    float gv = bf16r(g_acc), uv = bf16r(u_acc);
    if (a.limit > 0.f) {
        uv = fminf(fmaxf(uv, -a.limit), a.limit);
        gv = fminf(gv, a.limit);
    }
    float hv = silu_f(gv) * uv;
    if (a.wts) hv = hv * a.wts[entry];
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    // the pack as the 2D exchange carries it: every entry's main columns, then every entry's rest
    const size_t at = col < main ? size_t(entry) * a.ldh + col
                                 : size_t(a.P) * a.ldh + size_t(entry) * (a.rest_tiles * 32) + (col - main);
    a.h[at] = __float2bfloat16_rn(hv);
}

template <int MT>
__device__ __forceinline__ void gu_kernel(GuArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x * 4 + warp;
    if (tile >= a.n_tiles) return;
    if (a.owner && tile >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const int c0 = tile * 32;
    const bool fp8 = e == a.shared;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += MT * 16) {
        float ag[MT][4][4], au[MT][4][4];
        MxArgs m{};
        m.q = a.q; m.s = a.s; m.ldq = a.ldq; m.lds = a.lds;
        m.ldw = fp8 ? a.sldw : a.ldw; m.ldws = fp8 ? a.sldws : a.ldws; m.fmt = fp8 ? 0 : a.fmt;
        const int split = blockIdx.z;
        m.M = cnt; m.K = a.D; m.splits = a.splits; m.kblocks = a.splits > 1 ? a.kblocks : a.D / kBlk;
        m.rows = a.erow + base;  // the entries' activation rows
        m.w = reinterpret_cast<const uint8_t*>(a.w1[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s1[e]);
        mx_tile<MT>(m, c0, m0, split, ag);
        m.w = reinterpret_cast<const uint8_t*>(a.w3[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s3[e]);
        mx_tile<MT>(m, c0, m0, split, au);
        const int ncols = a.n_tiles * 32;
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int mi = m0 + i * 16 + g + (c >> 1) * 8;
                    if (mi >= cnt) continue;
                    const int entry = a.ent[base + mi];
                    const int col = c0 + j * 8 + 2 * t + (c & 1);
                    if (a.splits > 1) {
                        float* zs = a.z + size_t(split) * 2 * a.P * ncols;
                        zs[size_t(entry) * ncols + col] = ag[i][j][c];
                        zs[size_t(a.P + entry) * ncols + col] = au[i][j][c];
                    } else {
                        gu_store(a, entry, col, ag[i][j][c], au[i][j][c]);
                    }
                }
    }
}

// gate / up's split partials summed in split order (the same order for every row count), then the epilogue: one
// thread a column of expert act[blockIdx.y], over its entries (grid: x over the columns, y as tf_dsn_gu)
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu_finish_kernel(GuArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int ncols = a.n_tiles * 32;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncols) return;
    if (a.owner && col / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int mi = 0; mi < cnt; mi++) {
        const int entry = a.ent[base + mi];
        float gs = 0.f, us = 0.f;
        for (int sp = 0; sp < a.splits; sp++) {
            const float* zs = a.z + size_t(sp) * 2 * a.P * ncols;
            gs += zs[size_t(entry) * ncols + col];
            us += zs[size_t(a.P + entry) * ncols + col];
        }
        gu_store(a, entry, col, gs, us);
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu1_kernel(GuArgs a) { gu_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu2_kernel(GuArgs a) { gu_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_gu4_kernel(GuArgs a) { gu_kernel<4>(a); }

// Down of the grouped experts: for expert act[blockIdx.y]'s entries, y_e = h_q . w2 over the node's output columns
// (`n` of them, 32-column tiles), into out [entry, n] fp32; with round_bf16 the bf16 value widened (Expert.forward
// returns bf16), which only a rank holding the whole intermediate can do (a TP rank's is a partial sum).
// q / s: the entries' quantized h (tf_dsn_act_quant of h [entries, I], one row an entry).
struct DnArgs {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const int* ent;
    const int* off;
    const int* act;
    const int* nact;
    const uint64_t* w2; // [E]: MXFP4 [n, I/2] or FP8 [n, I] (this node's output columns)
    const uint64_t* s2;
    int64_t ldw, ldws;
    int fmt;
    int shared;         // the FP8 expert's id, -1: none
    int64_t sldw, sldws;
    int I, N;
    int round_bf16;     // 1: each expert's output rounded to bf16 (Expert.forward's dtype; a whole intermediate only)
    float* out;
    int64_t ldo;
    // split K (decode): splits > 1 writes each split's partial to z [splits, entries, N] fp32 and tf_dsn_dn_finish sums
    // them in split order into out
    int splits, kblocks;
    float* z;
    int P;              // the entries (z's row count)
    int early_exit;     // Track C (TF_DS_EXPERT_EXIT): as GuArgs'
    // fmt 2, NVFP4: as GuArgs'
    const uint8_t* nq;
    const uint8_t* ns;
    int64_t ldnq, ldns;
    const float* a2;
};

template <int MT>
__device__ __forceinline__ void dn_kernel(DnArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += MT * 16) {
        float acc[MT][4][4];
        MxArgs m{};
        m.q = a.q; m.s = a.s; m.ldq = a.ldq; m.lds = a.lds;
        m.rows = a.ent + base;
        m.w = reinterpret_cast<const uint8_t*>(a.w2[e]);
        m.ws = reinterpret_cast<const uint8_t*>(a.s2[e]);
        const bool fp8 = e == a.shared;
        m.ldw = fp8 ? a.sldw : a.ldw; m.ldws = fp8 ? a.sldws : a.ldws; m.fmt = fp8 ? 0 : a.fmt;
        const int split = blockIdx.z;
        m.M = cnt; m.K = a.I; m.splits = a.splits; m.kblocks = a.splits > 1 ? a.kblocks : a.I / kBlk;
        mx_tile<MT>(m, n0, m0, split, acc);
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int mi = m0 + i * 16 + g + (c >> 1) * 8;
                    if (mi >= cnt) continue;
                    const int entry = a.ent[base + mi];
                    const float v = acc[i][j][c];
                    const int n = n0 + j * 8 + 2 * t + (c & 1);
                    if (a.splits > 1) a.z[(size_t(split) * a.P + entry) * a.N + n] = v;
                    else a.out[size_t(entry) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
                }
    }
}

// down's split partials summed in split order into out: one thread a column of expert act[blockIdx.y], over its
// entries
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn_finish_kernel(DnArgs a) {
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= a.N) return;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int mi = 0; mi < cnt; mi++) {
        const int entry = a.ent[base + mi];
        float v = 0.f;
        for (int sp = 0; sp < a.splits; sp++) v += a.z[(size_t(sp) * a.P + entry) * a.N + n];
        a.out[size_t(entry) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn1_kernel(DnArgs a) { dn_kernel<1>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn2_kernel(DnArgs a) { dn_kernel<2>(a); }
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dn4_kernel(DnArgs a) { dn_kernel<4>(a); }

// The routed sum (MoE.forward: y[idx] += expert(x[idx]) in expert id order, fp32): y [R, N] fp32 (row stride ldy) =
// (accumulate ? y : 0) + the row's picked entries' out [r * S + s, N] in ascending expert id.
extern "C" __global__ void tf_dsn_combine_kernel(const float* __restrict__ out, int64_t ldo, const int* __restrict__ pick,
                                                 int R, int S, int E, int N, float* __restrict__ y, int64_t ldy,
                                                 int accumulate) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(R) * N) return;
    const int r = int(i / N), n = int(i % N);
    int order[16];
    int k = 0;
    for (int s = 0; s < S && s < 16; s++) {
        const int e = pick[r * S + s];
        if (e < 0 || e >= E) continue;
        int j = k++;
        while (j > 0 && pick[r * S + order[j - 1]] > e) { order[j] = order[j - 1]; j--; }
        order[j] = s;
    }
    float v = accumulate ? y[size_t(r) * ldy + n] : 0.f;
    for (int j = 0; j < k; j++) v += out[size_t(r * S + order[j]) * ldo + n];
    y[size_t(r) * ldy + n] = v;
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode GEMV of the experts (a decode round's few rows): a warp per output column, its lanes over the column's 32-wide
// K blocks (one 16-byte load an MXFP4 block, two an FP8 block, and the block's scale), each block decoded once into
// registers and applied to every one of the expert's rows: block sum (32 exact products, in order) * row scale * weight
// scale into the row's fp32 accumulator; the lanes' accumulators summed by a fixed shuffle tree. A row's arithmetic is
// its own, so its bits are the same at every row count. Grid: x the columns / 8 (8 warps a CTA), y the active experts.

constexpr int kGvRows = 8;  // the rows a pass over a column's weights takes (more rows: more passes)

// one weight block of row n at block kb as 32 floats (times nothing: the scale is apart)
__device__ __forceinline__ void gv_load_block(const uint8_t* w, int64_t ldw, int fmt, int n, int kb, float (&v)[32],
                                              const float* lut) {
    if (fmt == 1) {
        const uint4 b = *reinterpret_cast<const uint4*>(w + size_t(n) * ldw + size_t(kb) * 16);
        const uint32_t ws[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) v[i * 8 + j] = lut[(ws[i] >> (4 * j)) & 0xf];
    } else {
        const uint4* p = reinterpret_cast<const uint4*>(w + size_t(n) * ldw + size_t(kb) * 32);
        const uint4 b0 = p[0], b1 = p[1];
        const uint32_t ws[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) v[i * 4 + j] = from_e4m3(uint8_t(ws[i] >> (8 * j)));
    }
}

// the weight scale of row n, block kb (fp8: one per 32 x 32 block; fp4: one per 32 values of a row)
__device__ __forceinline__ float gv_wscale(const uint8_t* ws, int64_t ldws, int fmt, int n, int kb) {
    return fmt == 1 ? ue8m0(ws[size_t(n) * ldws + kb]) : ue8m0(ws[size_t(n / kBlk) * ldws + kb]);
}

// a block's dot product with an activation row's block (q codes at qrow, 32 bytes), in order
__device__ __forceinline__ float gv_dot(const float (&v)[32], const uint8_t* qrow) {
    const uint4* p = reinterpret_cast<const uint4*>(qrow);
    const uint4 a0 = p[0], a1 = p[1];
    const uint32_t as[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) d = fmaf(v[i * 4 + j], from_e4m3(uint8_t(as[i] >> (8 * j))), d);
    return d;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void gv_lut(float* lut) {
    // E2M1: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (convert.py FP4_TABLE)
    if (threadIdx.x < 16) {
        const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        lut[threadIdx.x] = threadIdx.x < 8 ? t[threadIdx.x] : -t[threadIdx.x - 8];
    }
    __syncthreads();
}

// a dense FP8 / FP4 linear as tf_dsn_mx_gemm (splits unused): y [M, N] of a decode launch's rows
extern "C" __global__ void __launch_bounds__(256) tf_dsn_mx_gemv_kernel(MxArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= a.N) return;
    const int nkb = a.K / kBlk;
    for (int r0 = 0; r0 < a.M; r0 += kGvRows) {
        const int nr = min(kGvRows, a.M - r0);
        float acc[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) acc[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float v[32];
            gv_load_block(a.w, a.ldw, a.fmt, n, kb, v, lut);
            const float sw = gv_wscale(a.ws, a.ldws, a.fmt, n, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.rows ? a.rows[r0 + i] : r0 + i;
                acc[i] += gv_dot(v, a.q + size_t(row) * a.ldq + size_t(kb) * kBlk) * a.s[size_t(row) * a.lds + kb] * sw;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float v = warp_sum(acc[i]);
            if (lane != 0) continue;
            const int m = r0 + i;
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += v;
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
        }
    }
}

// gate / up as tf_dsn_gu (the same arguments; splits unused): h of every entry of expert act[blockIdx.y]
extern "C" __global__ void __launch_bounds__(256) tf_dsn_gu_gemv_kernel(GuArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.x * 8 + warp;
    const int ncols = a.n_tiles * 32;
    if (col >= ncols) return;
    if (a.owner && col / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const uint8_t* w1 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    const uint8_t* w3 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    const uint8_t* s1 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    const uint8_t* s3 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    const int nkb = a.D / kBlk;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r0 = 0; r0 < cnt; r0 += kGvRows) {
        const int nr = min(kGvRows, cnt - r0);
        float ag[kGvRows], au[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) ag[i] = au[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float vg[32], vu[32];
            gv_load_block(w1, ldw, fmt, col, kb, vg, lut);
            gv_load_block(w3, ldw, fmt, col, kb, vu, lut);
            const float sg = gv_wscale(s1, ldws, fmt, col, kb), su = gv_wscale(s3, ldws, fmt, col, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.erow[base + r0 + i];
                const uint8_t* qr = a.q + size_t(row) * a.ldq + size_t(kb) * kBlk;
                const float sa = a.s[size_t(row) * a.lds + kb];
                ag[i] += gv_dot(vg, qr) * sa * sg;
                au[i] += gv_dot(vu, qr) * sa * su;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float g = warp_sum(ag[i]), u = warp_sum(au[i]);
            if (lane == 0) gu_store(a, a.ent[base + r0 + i], col, g, u);
        }
    }
}

// down as tf_dsn_dn (the same arguments; splits unused): out of every entry of expert act[blockIdx.y]
extern "C" __global__ void __launch_bounds__(256) tf_dsn_dn_gemv_kernel(DnArgs a) {
    __shared__ float lut[16];
    gv_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= a.N) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const uint8_t* w2 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    const uint8_t* s2 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    const int nkb = a.I / kBlk;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r0 = 0; r0 < cnt; r0 += kGvRows) {
        const int nr = min(kGvRows, cnt - r0);
        float acc[kGvRows];
#pragma unroll
        for (int i = 0; i < kGvRows; i++) acc[i] = 0.f;
        for (int kb = lane; kb < nkb; kb += 32) {
            float v[32];
            gv_load_block(w2, ldw, fmt, n, kb, v, lut);
            const float sw = gv_wscale(s2, ldws, fmt, n, kb);
#pragma unroll
            for (int i = 0; i < kGvRows; i++) {
                if (i >= nr) break;
                const int row = a.ent[base + r0 + i];
                acc[i] += gv_dot(v, a.q + size_t(row) * a.ldq + size_t(kb) * kBlk) * a.s[size_t(row) * a.lds + kb] * sw;
            }
        }
#pragma unroll
        for (int i = 0; i < kGvRows; i++) {
            if (i >= nr) break;
            const float v = warp_sum(acc[i]);
            if (lane == 0) a.out[size_t(a.ent[base + r0 + i]) * a.ldo + n] = a.round_bf16 ? bf16r(v) : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode MMA with the operands swapped ("swap AB"): the tensor core's 16-wide side is 16 weight rows (output columns),
// its 8-wide side up to 8 activation rows, so a decode round's rows share one m16n8k32 MMA a K block and each weight
// row's block is read whole (32 bytes FP8, 16 bytes MXFP4). Both operands take the tile kernel's K order in a block
// (a thread's 8 contiguous bytes: low 4 to k = 4t.., high 4 to k = 16 + 4t..). Every (column, row) of the result is its
// own MMA output, so a row's bits never depend on the other rows. A warp: 16 columns; K blocks in order.

constexpr int kSabRows = 8;

__device__ __forceinline__ void sab_lut(uint8_t* lut) {
    if (threadIdx.x < 16) lut[threadIdx.x] = kE2m1ToE4m3[threadIdx.x];
    __syncthreads();
}

__device__ __forceinline__ uint32_t sab_fp4(const uint8_t* lut, uint16_t p) {
    return uint32_t(lut[p & 0xf]) | uint32_t(lut[(p >> 4) & 0xf]) << 8 | uint32_t(lut[(p >> 8) & 0xf]) << 16 |
           uint32_t(lut[(p >> 12) & 0xf]) << 24;
}

// weight rows n0 + g and n0 + g + 8 at block kb: the A fragment
__device__ __forceinline__ void sab_a(const uint8_t* w, int64_t ldw, int fmt, int n0, int kb, const uint8_t* lut,
                                      uint32_t (&a)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const size_t row = size_t(n0 + g + 8 * h);
        if (fmt == 1) {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(w + row * ldw + size_t(kb) * 16 + t * 4);
            a[h] = sab_fp4(lut, uint16_t(v & 0xffff));
            a[2 + h] = sab_fp4(lut, uint16_t(v >> 16));
        } else {
            const uint2 v = *reinterpret_cast<const uint2*>(w + row * ldw + size_t(kb) * 32 + t * 8);
            a[h] = v.x;
            a[2 + h] = v.y;
        }
    }
}

// one 16-column, up-to-8-row block step: acc (C fragment: columns g, g + 8 by rows 2t, 2t + 1) += sum * sa * sw
__device__ __forceinline__ void sab_step(const uint32_t (&a)[4], const uint8_t* qrow, float sa0, float sa1, float sw0,
                                         float sw1, float (&acc)[4]) {
    const int t = threadIdx.x & 3;
    const uint2 bv = *reinterpret_cast<const uint2*>(qrow + t * 8);
    const uint32_t b[2] = {bv.x, bv.y};
    float d[4];
    mma_e4m3(d, a, b);
    acc[0] += d[0] * sa0 * sw0;
    acc[1] += d[1] * sa1 * sw0;
    acc[2] += d[2] * sa0 * sw1;
    acc[3] += d[3] * sa1 * sw1;
}

// columns n0 .. n0 + 15 of a weight (w, ws, fmt), K blocks kb0, kb0 + kstep, ... below nkb, for up to 8 rows: row i
// (< nr) is activation row rows_of(i); a thread's B row is g (its lane group), its C rows 2t, 2t + 1
template <typename RowOf>
__device__ __forceinline__ void sab_cols(const uint8_t* w, const uint8_t* ws, int64_t ldw, int64_t ldws, int fmt,
                                         int n0, int kb0, int kstep, int nkb, const uint8_t* q, int64_t ldq,
                                         const float* s, int64_t lds, int nr, RowOf rows_of, const uint8_t* lut,
                                         float (&acc)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    const int rb = rows_of(g < nr ? g : 0);
    const int r0 = rows_of(2 * t < nr ? 2 * t : 0), r1 = rows_of(2 * t + 1 < nr ? 2 * t + 1 : 0);
    const uint8_t* qb = q + size_t(rb) * ldq;
    const int c0 = n0 + g, c1 = n0 + g + 8;
#pragma unroll 4
    for (int kb = kb0; kb < nkb; kb += kstep) {
        uint32_t a[4];
        sab_a(w, ldw, fmt, n0, kb, lut, a);
        const float sw0 = fmt == 1 ? ue8m0(ws[size_t(c0) * ldws + kb]) : ue8m0(ws[size_t(c0 / kBlk) * ldws + kb]);
        const float sw1 = fmt == 1 ? ue8m0(ws[size_t(c1) * ldws + kb]) : ue8m0(ws[size_t(c1 / kBlk) * ldws + kb]);
        sab_step(a, qb + size_t(kb) * kBlk, s[size_t(r0) * lds + kb], s[size_t(r1) * lds + kb], sw0, sw1, acc);
    }
}

constexpr int kSabWarps = 8;  // a CTA's warps: one 16-column group, the K blocks dealt round them

// the CTA's warps' partials summed in warp order (a fixed order): warp 0's lanes end with the sums
__device__ __forceinline__ void sab_reduce(float (&acc)[4], float (*red)[32][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int c = 0; c < 4; c++) red[warp][lane][c] = acc[c];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = 0.f;
            for (int w2 = 0; w2 < kSabWarps; w2++) v += red[w2][lane][c];
            acc[c] = v;
        }
    }
    __syncthreads();
}

// a dense FP8 / FP4 linear (MxArgs; splits unused): y [M, N], a CTA 16 columns, rows 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_mx_sab_kernel(MxArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.N) return;
    for (int m0 = 0; m0 < a.M; m0 += kSabRows) {
        const int nr = min(kSabRows, a.M - m0);
        float acc[4];
        sab_cols(a.w, a.ws, a.ldw, a.ldws, a.fmt, n0, warp, kSabWarps, a.K / kBlk, a.q, a.ldq, a.s, a.lds, nr,
                 [&](int i) { return a.rows ? a.rows[m0 + i] : m0 + i; }, lut, acc);
        sab_reduce(acc, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            const int m = m0 + i, n = n0 + g + 8 * (c >> 1);
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += acc[c];
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, acc[c]);
        }
    }
}

// gate / up (GuArgs; splits unused): a CTA 16 columns of the pack of expert act[blockIdx.y], its entries 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_gu_sab_kernel(GuArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.n_tiles * 32) return;
    if (a.owner && n0 / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += kSabRows) {
        const int nr = min(kSabRows, cnt - m0);
        auto rows_of = [&](int i) { return a.erow[base + m0 + i]; };
        float ag[4], au[4];
        sab_cols(reinterpret_cast<const uint8_t*>(a.w1[e]), reinterpret_cast<const uint8_t*>(a.s1[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.D / kBlk, a.q, a.ldq, a.s, a.lds, nr, rows_of, lut, ag);
        sab_cols(reinterpret_cast<const uint8_t*>(a.w3[e]), reinterpret_cast<const uint8_t*>(a.s3[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.D / kBlk, a.q, a.ldq, a.s, a.lds, nr, rows_of, lut, au);
        sab_reduce(ag, red);
        sab_reduce(au, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            gu_store(a, a.ent[base + m0 + i], n0 + g + 8 * (c >> 1), ag[c], au[c]);
        }
    }
}

// down (DnArgs; splits unused): a CTA 16 output columns of expert act[blockIdx.y], its entries 8 at a time
extern "C" __global__ void __launch_bounds__(256) tf_dsn_dn_sab_kernel(DnArgs a) {
    __shared__ uint8_t lut[16];
    __shared__ float red[kSabWarps][32][4];
    sab_lut(lut);
    if (int(blockIdx.y) >= a.nact[0]) return;
    const int e = a.act[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    if (n0 >= a.N) return;
    const bool fp8 = e == a.shared;
    const int fmt = fp8 ? 0 : a.fmt;
    const int64_t ldw = fp8 ? a.sldw : a.ldw, ldws = fp8 ? a.sldws : a.ldws;
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int m0 = 0; m0 < cnt; m0 += kSabRows) {
        const int nr = min(kSabRows, cnt - m0);
        float acc[4];
        sab_cols(reinterpret_cast<const uint8_t*>(a.w2[e]), reinterpret_cast<const uint8_t*>(a.s2[e]), ldw, ldws, fmt,
                 n0, warp, kSabWarps, a.I / kBlk, a.q, a.ldq, a.s, a.lds, nr,
                 [&](int i) { return a.ent[base + m0 + i]; }, lut, acc);
        sab_reduce(acc, red);
        if (warp != 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int i = 2 * t + (c & 1);
            if (i >= nr) continue;
            const float v = acc[c];
            a.out[size_t(a.ent[base + m0 + i]) * a.ldo + n0 + g + 8 * (c >> 1)] = a.round_bf16 ? bf16r(v) : v;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// bf16 linears with fp32 accumulation (F.linear of bf16 values: the head's fp32 GEMM of bf16 values, the compressor's,
// the indexer's wk and weights_proj, wo_a's einsum per group): y [M, N] = x [M, K] . w [N, K]^T, x bf16 (row stride
// ldx; group g reads x + g * gx and w + g * gw, y + g * gy: wo_a's block-diagonal groups as blockIdx.z), K in
// 16-column MMA steps in order, one warp 32 columns, 16 rows a tile.
struct BfArgs {
    const __nv_bfloat16* x;
    int64_t ldx, gx;
    const __nv_bfloat16* w;
    int64_t ldw, gw;
    int M, N, K;
    void* y;
    int64_t ldy, gy;
    int y_dtype;
    float scale;  // y = acc * scale (1: none); weights_proj's softmax_scale * n_heads^-0.5 after its bf16 rounding
    int round_first; // 1: y = bf16(bf16(acc) * scale) (weights_proj: the linear's bf16 output, then the scale)
    // split K (decode, one group): splits > 1 writes each split's partial to z [splits, M, N] fp32 (blockIdx.z the
    // split) and tf_dsn_bf_reduce sums them in split order, then rounds / scales as below; ksteps: 16-column steps a split
    int splits, ksteps;
    float* z;
};

// the bf16 GEMM's epilogue for one value
__device__ __forceinline__ void bf_store(const BfArgs& a, void* y, int m, int n, float v) {
    if (a.round_first) v = bf16r(v);
    if (a.scale != 1.f) v = v * a.scale;
    store_y(y, a.y_dtype, size_t(m) * a.ldy + n, v);
}

extern "C" __global__ void __launch_bounds__(128) tf_dsn_bf16_gemm_kernel(BfArgs a) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 4 + warp) * 32;
    if (n0 >= a.N) return;
    const int grp = a.splits > 1 ? 0 : blockIdx.z;
    const int split = a.splits > 1 ? blockIdx.z : 0;
    const int k0 = split * a.ksteps * 16;
    const int k1 = a.splits > 1 ? min(a.K, k0 + a.ksteps * 16) : a.K;
    const __nv_bfloat16* x = a.x + size_t(grp) * a.gx;
    const __nv_bfloat16* w = a.w + size_t(grp) * a.gw;
    for (int m0 = blockIdx.y * 16; m0 < a.M; m0 += gridDim.y * 16) {
        float acc[4][4] = {};
        const int r0 = min(m0 + g, a.M - 1), r1 = min(m0 + g + 8, a.M - 1);
#pragma unroll 4
        for (int k = k0; k < k1; k += 16) {
            uint32_t af[4];
            af[0] = *reinterpret_cast<const uint32_t*>(x + size_t(r0) * a.ldx + k + 2 * t);
            af[1] = *reinterpret_cast<const uint32_t*>(x + size_t(r1) * a.ldx + k + 2 * t);
            af[2] = *reinterpret_cast<const uint32_t*>(x + size_t(r0) * a.ldx + k + 8 + 2 * t);
            af[3] = *reinterpret_cast<const uint32_t*>(x + size_t(r1) * a.ldx + k + 8 + 2 * t);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int n = n0 + j * 8 + g;
                uint32_t b[2];
                b[0] = *reinterpret_cast<const uint32_t*>(w + size_t(n) * a.ldw + k + 2 * t);
                b[1] = *reinterpret_cast<const uint32_t*>(w + size_t(n) * a.ldw + k + 8 + 2 * t);
                mma_bf16(acc[j], af, b);
            }
        }
        char* yb = reinterpret_cast<char*>(a.y);
        const int esz = a.y_dtype == 2 ? 4 : 2;
        void* y = yb + size_t(grp) * a.gy * esz;
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + j * 8 + 2 * t + (c & 1);
                if (a.splits > 1) a.z[(size_t(split) * a.M + m) * a.N + n] = acc[j][c];
                else bf_store(a, y, m, n, acc[j][c]);
            }
    }
}

// the bf16 GEMM's split partials summed in split order, then its epilogue
extern "C" __global__ void tf_dsn_bf_reduce_kernel(BfArgs a) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(a.M) * a.N) return;
    float v = 0.f;
    for (int sp = 0; sp < a.splits; sp++) v += a.z[size_t(sp) * a.M * a.N + i];
    bf_store(a, a.y, int(i / a.N), int(i % a.N), v);
}

// bf16 -> fp16 (round to nearest; the checkpoint's bf16 tensors that a kernel of the EXL3 lane reads as fp16, the
// drafter's Markov head)
extern "C" __global__ void tf_dsn_bf16_to_f16_kernel(const __nv_bfloat16* __restrict__ in, __half* __restrict__ out,
                                                     int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(__bfloat162float(in[i]));
}

// ---------------------------------------------------------------------------------------------------------------------
// Prefill tile GEMMs (prompt chunks). The arithmetic of tf_dsn_mx_gemm, tf_dsn_gu and tf_dsn_dn exactly: each 32-wide
// K block's m16n8k32 sum from zero (both operands in mx_tile's K order within the block), times s[m, kb] * ws(n, kb),
// added to the fp32 accumulator, K blocks in order; with a shape's K split (mx.zig splits()) each split's partial is
// summed from zero and the partials added in split order as tf_dsn_reduce adds them. So every output has the tile
// kernels' bits, a row's bits never depend on the other rows, and the engine may pick either kernel for any launch.
// What differs is the schedule: a CTA of 8 warps computes a BM x BN tile (NWT weights at once: gate and up share the
// activation rows), its operands staged through shared memory by cp.async in a STAGES-deep pipeline of KBS K blocks a
// stage (activation rows gathered by a row table: an expert's entries), MXFP4 decoded to E4M3 by byte permutes at the
// fragment load. Grouped experts run over a tile list (expert, first row) built after the grouping, so a skewed expert
// takes as many CTAs as its rows need. (A multi-stage cp.async pipeline in the manner of CUTLASS's sm80 GEMMs; written
// for this kernel, no code copied.)
namespace pf {

__device__ __forceinline__ uint32_t sptr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp4(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(ok ? 4 : 0));
}
// with an L2 eviction policy (createpolicy): streamed weights evict first, reused activation rows last
__device__ __forceinline__ void cp16h(void* dst, const void* src, bool ok, uint64_t pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(sptr(dst)), "l"(src),
                 "r"(ok ? 16 : 0), "l"(pol));
}
__device__ __forceinline__ uint64_t evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ uint64_t evict_last() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
    return p;
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait_groups() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// four E2M1 codes (16 bits, low nibble first) -> four E4M3 bytes: kE2m1ToE4m3's values (the magnitudes 0, 0.5 .. 6 by
// one byte permute of the eight-byte table, the sign bit moved to bit 7)
__device__ __forceinline__ uint32_t e2m1x4(uint32_t p) {
    const uint32_t mag = __byte_perm(0x3c383000u, 0x4c484440u, p & 0x7777u);
    const uint32_t q = __byte_perm(p, 0u, 0x1100u);  // bytes b0 b0 b1 b1: value i's sign at bit 3 or 7 of byte i
    return mag | ((q << 4) & 0x00800080u) | (q & 0x80008000u);
}

constexpr int kThreads = 256;  // 8 warps a CTA

// A tile's shapes and shared memory: BM rows by BN columns of NWT weights, warps WGM (rows) x WGN (columns), FMT 1
// MXFP4 / 0 FP8 weights, KBS K blocks a stage, STAGES stages. Row strides are padded so a warp's fragment loads hit
// distinct banks (activation / FP8 rows: 8-byte loads; MXFP4 rows: 4-byte loads).
template <int BM_, int BN_, int WGM_, int WGN_, int NWT_, int FMT_, int KBS_, int STAGES_, int SWZ_ = 1>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, NWT = NWT_, FMT = FMT_, KBS = KBS_, STAGES = STAGES_;
    static constexpr bool SWZ = SWZ_ != 0;  // rows unpadded, their 16-byte pieces XOR-swizzled (else padded rows)
    static constexpr int WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
    static constexpr int SA = KBS * 32 + (SWZ ? 0 : 32);
    static constexpr int SB = FMT ? KBS * 16 + (SWZ ? 0 : 16) : KBS * 32 + (SWZ ? 0 : 32);
    static constexpr int A_BYTES = BM * SA;
    static constexpr int B_BYTES = NWT * BN * SB;
    static constexpr int AS_BYTES = BM * KBS * 4;
    static constexpr int WS_BYTES = FMT ? NWT * BN * 4 : 0;
    static constexpr int STAGE = (A_BYTES + B_BYTES + AS_BYTES + WS_BYTES + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(WGM * WGN * 32 == kThreads, "8 warps");
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    static_assert(FMT == 0 || KBS == 4, "MXFP4 scales stage as one 4-byte word a column");
    static_assert(KBS == 2 || KBS == 4, "stage");
    // a byte row's 16-byte piece c of row r: where it sits. 8-byte fragment rows (activations, FP8 weights: a half
    // warp reads rows r..r+3, two pieces each) and 4-byte ones (MXFP4: a warp reads rows r..r+7, one piece each) land
    // on distinct banks.
    __device__ static __forceinline__ int pa(int r, int c) {
        if (!SWZ) return c;
        return KBS == 4 ? c ^ ((r & 3) << 1) : c ^ (((r >> 1) & 1) << 1);
    }
    __device__ static __forceinline__ int pb(int r, int c) {
        if (!SWZ) return c;
        if (FMT) return c ^ ((r >> 1) & 3);
        return pa(r, c);
    }
};

// The operands of one tile: activation codes q / scales s (row strides ldq, lds) of the rows srow[] names (shared
// memory, -1: a padding row), NWT weights w[] / ws[] (row strides ldw, ldws) from column n0.
struct Ops {
    const uint8_t* q;
    const float* s;
    int64_t ldq, lds;
    const uint8_t* w0;  // weight 0 (gate, or the one weight) and 1 (up): scalars, so no local copy is indexed
    const uint8_t* w1;
    const uint8_t* ws0;
    const uint8_t* ws1;
    int64_t ldw, ldws;
    int n0;
    int stream;  // 1: the weights stream through once (grouped experts): they evict first, the activation rows last
    __device__ __forceinline__ const uint8_t* w(int i) const { return i ? w1 : w0; }
    __device__ __forceinline__ const uint8_t* ws(int i) const { return i ? ws1 : ws0; }
};

template <class C>
__device__ __forceinline__ void load_stage(char* st, const int* srow, const Ops& o, int kb0) {
    const int tid = threadIdx.x;
    char* A = st;
    char* B = A + C::A_BYTES;
    char* AS = B + C::B_BYTES;
    char* WS = AS + C::AS_BYTES;
    const uint64_t pa = o.stream ? evict_last() : 0, pw = o.stream ? evict_first() : 0;
    constexpr int CPA = C::KBS * 2;  // 16-byte pieces of an activation row's stage
    for (int c = tid; c < C::BM * CPA; c += kThreads) {
        const int r = c / CPA, cc = c % CPA;
        const int row = srow[r];
        const bool ok = row >= 0;
        void* dst = A + r * C::SA + C::pa(r, cc) * 16;
        const uint8_t* src = o.q + size_t(ok ? row : 0) * o.ldq + kb0 * 32 + cc * 16;
        if (o.stream) cp16h(dst, src, ok, pa);
        else cp16(dst, src, ok);
    }
    for (int r = tid; r < C::BM; r += kThreads) {
        const int row = srow[r];
        const bool ok = row >= 0;
        const float* src = o.s + size_t(ok ? row : 0) * o.lds + kb0;
        if (C::KBS == 4) cp16(AS + r * 16, src, ok);
        else cp8(AS + r * 8, src, ok);
    }
    constexpr int CPB = C::FMT ? C::KBS : C::KBS * 2;
    for (int c = tid; c < C::NWT * C::BN * CPB; c += kThreads) {
        const int r = c / CPB, cc = c % CPB;
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        void* dst = B + r * C::SB + C::pb(r, cc) * 16;
        const uint8_t* src = o.w(wi) + size_t(col) * o.ldw + kb0 * (C::FMT ? 16 : 32) + cc * 16;
        if (o.stream) cp16h(dst, src, true, pw);
        else cp16(dst, src, true);
    }
    if (C::FMT)
        for (int r = tid; r < C::NWT * C::BN; r += kThreads) {
            const int wi = r / C::BN, col = o.n0 + r % C::BN;
            cp4(WS + r * 4, o.ws(wi) + size_t(col) * o.ldws + kb0, true);
        }
}

// The tile's K loop over nkb K blocks: acc[wi][i][j][c] (mx_tile's C fragments) for weight wi, this warp's m16 tile i
// and n8 tile j. fold > 0: a K split every `fold` blocks (the partials summed from zero, added in split order).
// mvalid: the tile's rows from mvalid on are padding (their m16 tiles skipped).
// FP8 weights: wss holds the tile's weight scales [NWT][BN / 32][nkb] (fp8_scales filled it; any barrier after).
template <class C>
__device__ __forceinline__ void fp8_scales(uint8_t* wss, const Ops& o, int nkb) {
    constexpr int nb = C::BN / kBlk;
    for (int i = threadIdx.x; i < C::NWT * nb * nkb; i += kThreads) {
        const int wi = i / (nb * nkb), r = i % (nb * nkb);
        const int blk = r / nkb, kb = r % nkb;
        wss[i] = o.ws(wi)[size_t(o.n0 / kBlk + blk) * o.ldws + kb];
    }
}
constexpr int kWss = 2048;  // the slab's bytes (2 weights x 4 blocks x 256 K blocks)

template <class C, bool FOLD, bool FULL, bool FAST>
__device__ __forceinline__ void run_tile1(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                          int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
    float part[C::NWT][C::MT][C::NT][4];
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    part[wi][i][j][c] = 0.f;
                    if (FOLD) acc[wi][i][j][c] = 0.f;
                }
    bool first = true;
    const int nst = nkb / C::KBS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) load_stage<C>(smem + s * C::STAGE, srow, o, s * C::KBS);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) load_stage<C>(smem + (nx % C::STAGES) * C::STAGE, srow, o, nx * C::KBS);
            commit();
        }
        const char* A = smem + (st % C::STAGES) * C::STAGE;
        const char* B = A + C::A_BYTES;
        const float* AS = reinterpret_cast<const float*>(B + C::B_BYTES);
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(B + C::B_BYTES + C::AS_BYTES);
        const int kb0 = st * C::KBS;
        // this thread's activation rows' scales (rows g, g + 8 of each m16 tile) and, MXFP4, its C columns' weight
        // scales (a word of KBS bytes a column)
        float sa[C::MT][2][C::KBS];
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = wm * C::WM + i * 16 + g + h * 8;
#pragma unroll
                for (int kk = 0; kk < C::KBS; kk++) sa[i][h][kk] = AS[r * C::KBS + kk];
            }
        uint32_t wsw[C::NWT][C::NT][2];
        if (C::FMT) {
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int c = 0; c < 2; c++) wsw[wi][j][c] = WS[wi * C::BN + wn * C::WN + j * 8 + 2 * t + c];
        }
#pragma unroll
        for (int kk = 0; kk < C::KBS; kk++) {
            const int kb = kb0 + kk;
            uint32_t af[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                const int r = wm * C::WM + i * 16 + g;
                const int pc = kk * 2 + (t >> 1), po = (t & 1) * 8;
                const uint2 lo = *reinterpret_cast<const uint2*>(A + r * C::SA + C::pa(r, pc) * 16 + po);
                const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * C::SA + C::pa(r + 8, pc) * 16 + po);
                af[i][0] = lo.x;
                af[i][1] = hi.x;
                af[i][2] = lo.y;
                af[i][3] = hi.y;
            }
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++) {
                // every MMA of the K block first (independent, so their latencies overlap), then the scaled sums
                float d[C::MT][C::NT][4];
                float wsc[C::NT][2];
#pragma unroll
                for (int j = 0; j < C::NT; j++) {
                    const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                    uint32_t b[2];
                    if (C::FMT) {
                        const uint32_t v = *reinterpret_cast<const uint32_t*>(B + br * C::SB + C::pb(br, kk) * 16 + t * 4);
                        b[0] = e2m1x4(v & 0xffffu);
                        b[1] = e2m1x4(v >> 16);
                        wsc[j][0] = ue8m0(uint8_t(wsw[wi][j][0] >> (8 * kk)));
                        wsc[j][1] = ue8m0(uint8_t(wsw[wi][j][1] >> (8 * kk)));
                    } else {
                        const uint2 v = *reinterpret_cast<const uint2*>(B + br * C::SB + C::pb(br, kk * 2 + (t >> 1)) * 16 + (t & 1) * 8);
                        b[0] = v.x;
                        b[1] = v.y;
                        // an FP8 scale covers a 32 x 32 block: one for the n8 tile's columns
                        const int blk = (wn * C::WN + j * 8) / kBlk;
                        wsc[j][0] = wsc[j][1] = ue8m0(wss[(wi * (C::BN / kBlk) + blk) * nkb + kb]);
                    }
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        mma_e4m3(d[i][j], af[i], b);
                    }
                }
                if (!C::FMT && FAST) {
                    // FP8 with every weight scale in [2^-104, 1] (run_tile checked the slab): d * (sa * ws) is the
                    // same exact product as (d * sa) * ws (powers of two, every intermediate a normal float), so the
                    // same FMA result, with one multiply a row and K block instead of one an output
#pragma unroll
                    for (int j = 0; j < C::NT; j++)
#pragma unroll
                        for (int i = 0; i < C::MT; i++) {
                            if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                            const float s0 = sa[i][0][kk] * wsc[j][0], s1 = sa[i][1][kk] * wsc[j][0];
                            part[wi][i][j][0] = fmaf(d[i][j][0], s0, part[wi][i][j][0]);
                            part[wi][i][j][1] = fmaf(d[i][j][1], s0, part[wi][i][j][1]);
                            part[wi][i][j][2] = fmaf(d[i][j][2], s1, part[wi][i][j][2]);
                            part[wi][i][j][3] = fmaf(d[i][j][3], s1, part[wi][i][j][3]);
                        }
                } else
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (!FULL && wm * C::WM + i * 16 >= mvalid) continue;
                        part[wi][i][j][0] += d[i][j][0] * sa[i][0][kk] * wsc[j][0];
                        part[wi][i][j][1] += d[i][j][1] * sa[i][0][kk] * wsc[j][1];
                        part[wi][i][j][2] += d[i][j][2] * sa[i][1][kk] * wsc[j][0];
                        part[wi][i][j][3] += d[i][j][3] * sa[i][1][kk] * wsc[j][1];
                    }
            }
            if (FOLD && ((kb + 1) % fold == 0 || kb + 1 == nkb)) {
                // a split's end: its partial into the sum (tf_dsn_reduce: z[0], then + z[1], ...), the next from zero
#pragma unroll
                for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                    for (int i = 0; i < C::MT; i++)
#pragma unroll
                        for (int j = 0; j < C::NT; j++)
#pragma unroll
                            for (int c = 0; c < 4; c++) {
                                acc[wi][i][j][c] = first ? part[wi][i][j][c] : acc[wi][i][j][c] + part[wi][i][j][c];
                                part[wi][i][j][c] = 0.f;
                            }
                first = false;
            }
        }
    }
    wait_groups<0>();
    if (!FOLD) {
#pragma unroll
        for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
            for (int i = 0; i < C::MT; i++)
#pragma unroll
                for (int j = 0; j < C::NT; j++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[wi][i][j][c] = part[wi][i][j][c];
    }
}

// run_tile1 with the row check compiled out when every row of the tile is valid
template <class C, bool FOLD>
__device__ __forceinline__ void run_tile(char* smem, const int* srow, const uint8_t* wss, const Ops& o, int nkb,
                                         int fold, int mvalid, float (&acc)[C::NWT][C::MT][C::NT][4]) {
    // FP8: the fast scaling when every weight scale of the tile is a power of two in [2^-104, 1] (UE8M0 bytes 23..127:
    // with an activation scale in [2^-22, 2^100] no product of the scales leaves the normal range)
    bool fast = false;
    if (!C::FMT) {
        bool ok = true;
        for (int i = threadIdx.x; i < C::NWT * (C::BN / kBlk) * nkb; i += kThreads) ok = ok && wss[i] >= 23 && wss[i] <= 127;
        fast = __syncthreads_and(ok);
    }
    if (fast) {
        if (mvalid >= C::BM) run_tile1<C, FOLD, true, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile1<C, FOLD, false, true>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    } else {
        if (mvalid >= C::BM) run_tile1<C, FOLD, true, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
        else run_tile1<C, FOLD, false, false>(smem, srow, wss, o, nkb, fold, mvalid, acc);
    }
}

// the tile list of the grouped experts: (expert, first row) for every BM rows of every active expert, in act order
__device__ __forceinline__ void tiles_of(const int* off, const int* act, const int* nact, int bm, int2* tiles,
                                         int* ntiles) {
    int n = 0;
    for (int i = 0; i < nact[0]; i++) {
        const int e = act[i];
        const int cnt = off[e + 1] - off[e];
        for (int m0 = 0; m0 < cnt; m0 += bm) tiles[n++] = make_int2(e, m0);
    }
    ntiles[0] = n;
}

#ifndef PF_GEMM
#define PF_GEMM 128, 128, 2, 4, 1, 0, 2, 3
#endif
#ifndef PF_GEMM_FOLD
// WP2 (docs/prefill-plan.md): Hector's measured tuning for the K-split projections (wq_a / wkv / wq_b and every
// other dense FP8 GEMM that splits, since they share this kernel) -- 4 K-blocks a stage, 3 stages instead of 2
// and 4. Bit-identical to the previous config on real weights (tools/dsv41-zig/native/check_native.py,
// attn.wq_a/wq_b/wkv/wo_b and ffn.shared_experts.w1/w2); docs/prefill-plan-progress.md has the per-op timings.
#define PF_GEMM_FOLD 64, 128, 2, 4, 1, 0, 4, 3
#endif
#ifndef PF_GU4
#define PF_GU4 64, 128, 2, 4, 2, 1, 4, 3
#endif
#ifndef PF_GU8
#define PF_GU8 64, 64, 2, 4, 2, 0, 4, 3
#endif
#ifndef PF_DN4
#define PF_DN4 64, 128, 2, 4, 1, 1, 4, 3
#endif
#ifndef PF_DN8
#define PF_DN8 64, 128, 2, 4, 1, 0, 2, 3
#endif
#ifndef PF_MINB
#define PF_MINB 1
#endif
#ifndef PF_HINT
#define PF_HINT 0
#endif
using GemmCfg = Cfg<PF_GEMM>;          // dense FP8: warps 64 x 32
using GemmFoldCfg = Cfg<PF_GEMM_FOLD>;  // dense FP8 with a K split: warps 32 x 32 (partials and sums)
using GuCfg4 = Cfg<PF_GU4>;            // gate / up MXFP4: warps 32 x 32 of both weights
using GuCfg8 = Cfg<PF_GU8>;            // gate / up of the FP8 shared expert: column sub-tiles of the MXFP4 tile
using DnCfg4 = Cfg<PF_DN4>;            // down MXFP4
using DnCfg8 = Cfg<PF_DN8>;            // down of the FP8 shared expert
static_assert(GuCfg4::BM == 64 && GuCfg8::BM == 64 && DnCfg4::BM == 64 && DnCfg8::BM == 64, "the tile list's rows");
static_assert(GuCfg4::BN % GuCfg8::BN == 0 && DnCfg4::BN == DnCfg8::BN, "the FP8 expert's sub-tiles");
constexpr int kGemmSmem = GemmCfg::SMEM > GemmFoldCfg::SMEM ? GemmCfg::SMEM : GemmFoldCfg::SMEM;
constexpr int kGuSmem = GuCfg4::SMEM > GuCfg8::SMEM ? GuCfg4::SMEM : GuCfg8::SMEM;
constexpr int kDnSmem = DnCfg4::SMEM > DnCfg8::SMEM ? DnCfg4::SMEM : DnCfg8::SMEM;
constexpr int kBm = 64;  // the grouped tiles' rows

extern __shared__ __align__(128) char smem[];  // the stages (dynamic)

// gate / up's epilogue of a tile (gu_store for every valid entry): acc[0] the gate's sums, acc[1] the up's
template <class C>
__device__ __forceinline__ void gu_tile_store(const GuArgs& a, const float (&acc)[2][C::MT][C::NT][4], int n0, int base,
                                              int m0, int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int mi = m0 + wm * C::WM + i * 16 + g + (c >> 1) * 8;
                if (mi >= cnt) continue;
                const int col = n0 + wn * C::WN + j * 8 + 2 * t + (c & 1);
                gu_store(a, a.ent[base + mi], col, acc[0][i][j][c], acc[1][i][j][c]);
            }
}

// down's epilogue of a tile: each valid entry's columns into out (fp32; round_bf16: the bf16 value widened)
template <class C>
__device__ __forceinline__ void dn_tile_store(const DnArgs& a, const float (&acc)[1][C::MT][C::NT][4], int n0, int base,
                                              int m0, int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int mi = m0 + wm * C::WM + i * 16 + g + h * 8;
                if (mi >= cnt) continue;
                const int n = n0 + wn * C::WN + j * 8 + 2 * t;
                float v0 = acc[0][i][j][2 * h], v1 = acc[0][i][j][2 * h + 1];
                if (a.round_bf16) {
                    v0 = bf16r(v0);
                    v1 = bf16r(v1);
                }
                *reinterpret_cast<float2*>(a.out + size_t(a.ent[base + mi]) * a.ldo + n) = make_float2(v0, v1);
            }
}

}  // namespace pf

// The tile list after the grouping (one thread): tiles [P / 64 + E] int2, ntiles [1].
extern "C" __global__ void tf_dsn_pf_tiles_kernel(const int* __restrict__ off, const int* __restrict__ act,
                                                  const int* __restrict__ nact, int2* __restrict__ tiles,
                                                  int* __restrict__ ntiles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pf::tiles_of(off, act, nact, pf::kBm, tiles, ntiles);
}

// The dense MX GEMM of a prompt chunk (MxArgs as tf_dsn_mx_gemm, FP8 weights, N a multiple of 128, K of 64): grid
// (N / 128, ceil(M / 128)), 256 threads, kPfGemmSmem dynamic bytes. Splits > 1 fold in the CTA (no partials buffer).
template <class C, bool FOLD>
__device__ __forceinline__ void pf_gemm(const MxArgs& a) {
    __shared__ int srow[C::BM];
    __shared__ uint8_t wss[pf::kWss];
    const int m0 = blockIdx.x * C::BM;  // row tiles fastest: a wave's CTAs share their weight tiles in L2
    for (int r = threadIdx.x; r < C::BM; r += pf::kThreads) {
        const int m = m0 + r;
        srow[r] = m < a.M ? (a.rows ? a.rows[m] : m) : -1;
    }
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = a.w; o.ws0 = o.ws1 = a.ws; o.ldw = a.ldw; o.ldws = a.ldws;
    o.n0 = blockIdx.y * C::BN;
    const int nkb = a.K / kBlk;
    pf::fp8_scales<C>(wss, o, nkb);
    __syncthreads();
    float acc[1][C::MT][C::NT][4];
    pf::run_tile<C, FOLD>(pf::smem, srow, wss, o, nkb, a.kblocks, a.M - m0, acc);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int i = 0; i < C::MT; i++)
#pragma unroll
        for (int j = 0; j < C::NT; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * C::WM + i * 16 + g + h * 8;
                if (m >= a.M) continue;
                const int n = o.n0 + wn * C::WN + j * 8 + 2 * t;
                const float v0 = acc[0][i][j][2 * h], v1 = acc[0][i][j][2 * h + 1];
                if (a.accumulate) {
                    float2* yp = reinterpret_cast<float2*>(reinterpret_cast<float*>(a.y) + size_t(m) * a.ldy + n);
                    float2 cur = *yp;
                    cur.x += v0;
                    cur.y += v1;
                    *yp = cur;
                } else {
                    store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v0);
                    store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n + 1, v1);
                }
            }
}

// The dense MX GEMM of a prompt chunk (MxArgs as tf_dsn_mx_gemm, FP8 weights, N a multiple of 128, K of 64 and at most
// 8192), 256 threads, kPfGemmSmem dynamic bytes: splits 1, grid (ceil(M / 128), N / 128); splits > 1 (folded in the
// CTA, no partials buffer): tf_dsn_pf_gemm_fold, grid (ceil(M / 64), N / 128).
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_kernel(MxArgs a) { pf_gemm<pf::GemmCfg, false>(a); }
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gemm_fold_kernel(MxArgs a) { pf_gemm<pf::GemmFoldCfg, true>(a); }

// Gate / up of the grouped experts for a prompt chunk (GuArgs as tf_dsn_gu, splits 1; the pack's columns a multiple of
// 128, its main block's too): grid (n_tiles * 32 / 128, P / 64 + E), 256 threads, kPfGuSmem dynamic bytes; blockIdx.y
// the tile (tiles[], ntiles from tf_dsn_pf_tiles).
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gu_kernel(GuArgs a, const int2* __restrict__ tiles,
                                                                         const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    const int c0 = blockIdx.x * pf::GuCfg4::BN;
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    if (a.owner && c0 >= main && a.owner[e] == 0) return;
    __shared__ int srow[pf::kBm];
    __shared__ uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < pf::kBm; r += pf::kThreads) srow[r] = m0 + r < cnt ? a.erow[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    o.w1 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    o.ws0 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    o.ws1 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    if (fp8) {
        using C = pf::GuCfg8;
        for (int half = 0; half < pf::GuCfg4::BN / C::BN; half++) {
            o.n0 = c0 + half * C::BN;
            if (half) __syncthreads();  // the stages' buffers and the scales are the next sub-tile's
            pf::fp8_scales<C>(wss, o, a.D / kBlk);
            __syncthreads();
            float acc[2][C::MT][C::NT][4];
            pf::run_tile<C, false>(pf::smem, srow, wss, o, a.D / kBlk, 0, cnt - m0, acc);
            pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
        }
    } else {
        using C = pf::GuCfg4;
        o.n0 = c0;
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        pf::run_tile<C, false>(pf::smem, srow, wss, o, a.D / kBlk, 0, cnt - m0, acc);
        pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// Down of the grouped experts for a prompt chunk (DnArgs as tf_dsn_dn, splits 1; N a multiple of 128): grid
// (N / 128, P / 64 + E), 256 threads, kPfDnSmem dynamic bytes, the tile list of the same grouping.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_dn_kernel(DnArgs a, const int2* __restrict__ tiles,
                                                                         const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    __shared__ int srow[pf::kBm];
    __shared__ uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < pf::kBm; r += pf::kThreads) srow[r] = m0 + r < cnt ? a.ent[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    o.ws0 = o.ws1 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    o.n0 = blockIdx.x * pf::DnCfg4::BN;
    if (fp8) {
        using C = pf::DnCfg8;
        pf::fp8_scales<C>(wss, o, a.I / kBlk);
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        pf::run_tile<C, false>(pf::smem, srow, wss, o, a.I / kBlk, 0, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    } else {
        using C = pf::DnCfg4;
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        pf::run_tile<C, false>(pf::smem, srow, wss, o, a.I / kBlk, 0, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// NVFP4's prefill tile (the routed experts of a prompt chunk): 64 entries by 128 columns, operands staged as the MXFP4
// tile's but an entry's codes half as wide and their scales its own bytes, one block-scaled MMA a 64-input step.
namespace pf {
template <int BM_, int BN_, int WGM_, int WGN_, int NWT_, int STAGES_>
struct NvCfg {
    static constexpr int BM = BM_, BN = BN_, WGM = WGM_, WGN = WGN_, NWT = NWT_, STAGES = STAGES_;
    static constexpr int STEPS = 2;  // 64-input steps a stage
    static constexpr int WM = BM / WGM, WN = BN / WGN, MT = WM / 16, NT = WN / 8;
    static constexpr int SR = STEPS * 32;  // a staged row's bytes, either operand's codes
    static constexpr int A_BYTES = BM * SR;
    static constexpr int B_BYTES = NWT * BN * SR;
    static constexpr int AS_BYTES = BM * STEPS * 4;  // an entry's four E4M3 scales a step
    static constexpr int WS_BYTES = NWT * BN * 4;    // a weight row's two per-32 scales a step
    static constexpr int STAGE = (A_BYTES + B_BYTES + AS_BYTES + WS_BYTES + 127) / 128 * 128;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(WGM * WGN * 32 == kThreads, "8 warps");
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile");
    // a row's 16-byte piece c: a warp reads eight rows' 4-byte pieces of one c, on distinct banks
    __device__ static __forceinline__ int pc(int r, int c) { return c ^ ((r >> 1) & 3); }
};
using NvGuCfg = NvCfg<64, 128, 2, 4, 2, 3>;
using NvDnCfg = NvCfg<64, 128, 2, 4, 1, 3>;
static_assert(NvGuCfg::SMEM <= kGuSmem && NvDnCfg::SMEM <= kDnSmem, "the launches' dynamic shared memory");

// One stage: step s0 and s0 + 1 of the entries' codes and scales (q, ns; rows srow[], -1: padding) and the weights.
template <class C>
__device__ __forceinline__ void nv_load(char* st, const int* srow, const uint8_t* q, int64_t ldq, const uint8_t* ns,
                                        int64_t ldns, const Ops& o, int s0) {
    const int tid = threadIdx.x;
    char* A = st;
    char* B = A + C::A_BYTES;
    char* AS = B + C::B_BYTES;
    char* WS = AS + C::AS_BYTES;
    constexpr int CP = C::STEPS * 2;  // 16-byte pieces of a row's stage
    for (int c = tid; c < C::BM * CP; c += kThreads) {
        const int r = c / CP, cc = c % CP;
        const int row = srow[r];
        const bool ok = row >= 0;
        cp16(A + r * C::SR + C::pc(r, cc) * 16, q + size_t(ok ? row : 0) * ldq + s0 * 32 + cc * 16, ok);
    }
    for (int r = tid; r < C::BM; r += kThreads) {
        const int row = srow[r];
        const bool ok = row >= 0;
        cp8(AS + r * 8, ns + size_t(ok ? row : 0) * ldns + s0 * 4, ok);
    }
    for (int c = tid; c < C::NWT * C::BN * CP; c += kThreads) {
        const int r = c / CP, cc = c % CP;
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        cp16(B + r * C::SR + C::pc(r, cc) * 16, o.w(wi) + size_t(col) * o.ldw + s0 * 32 + cc * 16, true);
    }
    for (int r = tid; r < C::NWT * C::BN; r += kThreads) {
        const int wi = r / C::BN, col = o.n0 + r % C::BN;
        cp4(WS + r * 4, o.ws(wi) + size_t(col) * o.ldws + s0 * 2, true);
    }
}

// The tile's steps: acc[wi][i][j][c] as run_tile's (entry row g + 8 (c >> 1), column 2 t + (c & 1) of the n8 tile), the
// MMA's own accumulation in K order. mvalid: the tile's rows from mvalid on are padding.
template <class C>
__device__ __forceinline__ void nv_run(char* smem, const int* srow, const uint8_t* q, int64_t ldq, const uint8_t* ns,
                                       int64_t ldns, const Ops& o, int steps, int mvalid,
                                       float (&acc)[C::NWT][C::MT][C::NT][4]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / C::WGN, wn = warp % C::WGN;
#pragma unroll
    for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[wi][i][j][c] = 0.f;
    const int nst = steps / C::STEPS;
#pragma unroll
    for (int s = 0; s < C::STAGES - 1; s++) {
        if (s < nst) nv_load<C>(smem + s * C::STAGE, srow, q, ldq, ns, ldns, o, s * C::STEPS);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<C::STAGES - 2>();
        __syncthreads();
        {
            const int nx = st + C::STAGES - 1;
            if (nx < nst) nv_load<C>(smem + (nx % C::STAGES) * C::STAGE, srow, q, ldq, ns, ldns, o, nx * C::STEPS);
            commit();
        }
        const char* A = smem + (st % C::STAGES) * C::STAGE;
        const char* B = A + C::A_BYTES;
        const char* AS = B + C::B_BYTES;
        const uint32_t* WS = reinterpret_cast<const uint32_t*>(AS + C::AS_BYTES);
#pragma unroll
        for (int p = 0; p < C::STEPS; p++) {
            uint32_t af[C::MT][4], sa[C::MT];
#pragma unroll
            for (int i = 0; i < C::MT; i++) {
                const int r = wm * C::WM + i * 16 + g, r8 = r + 8;
                af[i][0] = *reinterpret_cast<const uint32_t*>(A + r * C::SR + C::pc(r, 2 * p) * 16 + 4 * t);
                af[i][1] = *reinterpret_cast<const uint32_t*>(A + r8 * C::SR + C::pc(r8, 2 * p) * 16 + 4 * t);
                af[i][2] = *reinterpret_cast<const uint32_t*>(A + r * C::SR + C::pc(r, 2 * p + 1) * 16 + 4 * t);
                af[i][3] = *reinterpret_cast<const uint32_t*>(A + r8 * C::SR + C::pc(r8, 2 * p + 1) * 16 + 4 * t);
                sa[i] = *reinterpret_cast<const uint32_t*>(AS + ((t & 1) ? r8 : r) * 8 + p * 4);
            }
#pragma unroll
            for (int wi = 0; wi < C::NWT; wi++)
#pragma unroll
                for (int j = 0; j < C::NT; j++) {
                    const int br = wi * C::BN + wn * C::WN + j * 8 + g;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(B + br * C::SR + C::pc(br, 2 * p) * 16 + 4 * t);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(B + br * C::SR + C::pc(br, 2 * p + 1) * 16 + 4 * t);
                    const uint32_t w = WS[br];
                    const uint32_t sb = nv_scales((w >> (16 * p)) & 0xffu, (w >> (16 * p + 8)) & 0xffu);
#pragma unroll
                    for (int i = 0; i < C::MT; i++) {
                        if (wm * C::WM + i * 16 >= mvalid) continue;
                        mma_fp4(acc[wi][i][j], af[i], b0, b1, sa[i], sb);
                    }
                }
        }
    }
    wait_groups<0>();
}
}  // namespace pf

// tf_dsn_pf_gu with NVFP4 routed experts (fmt 2; the shared expert stays FP8): the same grid and tile list.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_gu_nv_kernel(GuArgs a, const int2* __restrict__ tiles,
                                                                            const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    const int c0 = blockIdx.x * pf::NvGuCfg::BN;
    const int main = (a.n_tiles - a.rest_tiles) * 32;
    if (a.owner && c0 >= main && a.owner[e] == 0) return;
    __shared__ int srow[pf::kBm];
    __shared__ uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    const bool fp8 = e == a.shared;
    for (int r = threadIdx.x; r < pf::kBm; r += pf::kThreads)
        srow[r] = m0 + r < cnt ? (fp8 ? a.erow[base + m0 + r] : a.ent[base + m0 + r]) : -1;
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = reinterpret_cast<const uint8_t*>(a.w1[e]);
    o.w1 = reinterpret_cast<const uint8_t*>(a.w3[e]);
    o.ws0 = reinterpret_cast<const uint8_t*>(a.s1[e]);
    o.ws1 = reinterpret_cast<const uint8_t*>(a.s3[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    if (fp8) {
        using C = pf::GuCfg8;
        for (int half = 0; half < pf::NvGuCfg::BN / C::BN; half++) {
            o.n0 = c0 + half * C::BN;
            if (half) __syncthreads();  // the stages' buffers and the scales are the next sub-tile's
            pf::fp8_scales<C>(wss, o, a.D / kBlk);
            __syncthreads();
            float acc[2][C::MT][C::NT][4];
            pf::run_tile<C, false>(pf::smem, srow, wss, o, a.D / kBlk, 0, cnt - m0, acc);
            pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
        }
    } else {
        using C = pf::NvGuCfg;
        o.n0 = c0;
        __syncthreads();
        float acc[2][C::MT][C::NT][4];
        pf::nv_run<C>(pf::smem, srow, a.nq, a.ldnq, a.ns, a.ldns, o, a.D / 64, cnt - m0, acc);
        const float a1 = a.a1[e], a3 = a.a3[e];
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    acc[0][i][j][c] *= a1;
                    acc[1][i][j][c] *= a3;
                }
        pf::gu_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// tf_dsn_pf_dn with NVFP4 routed experts: the same grid and tile list.
extern "C" __global__ void __launch_bounds__(256, PF_MINB) tf_dsn_pf_dn_nv_kernel(DnArgs a, const int2* __restrict__ tiles,
                                                                            const int* __restrict__ ntiles) {
    if (int(blockIdx.y) >= ntiles[0]) return;
    const int2 tl = tiles[blockIdx.y];
    const int e = tl.x, m0 = tl.y;
    __shared__ int srow[pf::kBm];
    __shared__ uint8_t wss[pf::kWss];
    const int base = a.off[e], cnt = a.off[e + 1] - base;
    for (int r = threadIdx.x; r < pf::kBm; r += pf::kThreads) srow[r] = m0 + r < cnt ? a.ent[base + m0 + r] : -1;
    const bool fp8 = e == a.shared;
    pf::Ops o{};
    o.q = a.q; o.s = a.s; o.ldq = a.ldq; o.lds = a.lds;
    o.w0 = o.w1 = reinterpret_cast<const uint8_t*>(a.w2[e]);
    o.ws0 = o.ws1 = reinterpret_cast<const uint8_t*>(a.s2[e]);
    o.ldw = fp8 ? a.sldw : a.ldw;
    o.ldws = fp8 ? a.sldws : a.ldws;
    o.stream = PF_HINT;
    o.n0 = blockIdx.x * pf::NvDnCfg::BN;
    if (fp8) {
        using C = pf::DnCfg8;
        pf::fp8_scales<C>(wss, o, a.I / kBlk);
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        pf::run_tile<C, false>(pf::smem, srow, wss, o, a.I / kBlk, 0, cnt - m0, acc);
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    } else {
        using C = pf::NvDnCfg;
        __syncthreads();
        float acc[1][C::MT][C::NT][4];
        pf::nv_run<C>(pf::smem, srow, a.nq, a.ldnq, a.ns, a.ldns, o, a.I / 64, cnt - m0, acc);
        const float a2 = a.a2[e];
#pragma unroll
        for (int i = 0; i < C::MT; i++)
#pragma unroll
            for (int j = 0; j < C::NT; j++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[0][i][j][c] *= a2;
        pf::dn_tile_store<C>(a, acc, o.n0, base, m0, cnt);
    }
}

// The bf16 GEMM of a prompt chunk (BfArgs as tf_dsn_bf16_gemm, splits 1): the same MMA chain (m16n8k16 accumulating
// in its C fragment, 16-column K steps in order, the standard fragment layout), so the same bits, on 128 x 128 CTA tiles
// staged by cp.async (32 K columns a stage, 4 stages) and read by ldmatrix. N a multiple of 128, K of 32; grid
// (ceil(M / 128), N / 128, groups), 256 threads, kPfBfSmem dynamic bytes.
namespace pf {
constexpr int kBfBM = 128, kBfBN = 128, kBfKS = 32, kBfStages = 4;
constexpr int kBfRow = kBfKS * 2 + 16;  // a staged row's bytes (80: ldmatrix's eight rows on distinct banks)
constexpr int kBfStage = (kBfBM + kBfBN) * kBfRow;
constexpr int kBfSmem = kBfStages * kBfStage;

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sptr(p)));
}
}  // namespace pf

extern "C" __global__ void __launch_bounds__(256, 1) tf_dsn_pf_bf16_kernel(BfArgs a) {
    using namespace pf;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int wm = warp / 4, wn = warp % 4;  // warp tile 64 rows x 32 columns
    const int grp = blockIdx.z;
    const __nv_bfloat16* x = a.x + size_t(grp) * a.gx;
    const __nv_bfloat16* w = a.w + size_t(grp) * a.gw;
    const int m0 = blockIdx.x * kBfBM, n0 = blockIdx.y * kBfBN;  // row tiles fastest (the weight tiles shared in L2)
    const int nst = a.K / kBfKS;
    auto load = [&](int st, int slot) {
        char* A = smem + slot * kBfStage;
        char* B = A + kBfBM * kBfRow;
        const int k0 = st * kBfKS;
        // 4 16-byte pieces a row, 128 + 128 rows: 4 pieces a thread
        for (int c = tid; c < (kBfBM + kBfBN) * 4; c += kThreads) {
            const int r = c >> 2, cc = c & 3;
            if (r < kBfBM) {
                const int m = m0 + r;
                const bool ok = m < a.M;
                cp16(A + r * kBfRow + cc * 16, x + size_t(ok ? m : 0) * a.ldx + k0 + cc * 8, ok);
            } else {
                const int n = n0 + r - kBfBM;
                cp16(B + (r - kBfBM) * kBfRow + cc * 16, w + size_t(n) * a.ldw + k0 + cc * 8, true);
            }
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
    const int mvalid = a.M - m0;
#pragma unroll
    for (int s = 0; s < kBfStages - 1; s++) {
        if (s < nst) load(s, s);
        commit();
    }
    for (int st = 0; st < nst; st++) {
        wait_groups<kBfStages - 2>();
        __syncthreads();
        if (st + kBfStages - 1 < nst) load(st + kBfStages - 1, (st + kBfStages - 1) % kBfStages);
        commit();
        const char* A = smem + (st % kBfStages) * kBfStage;
        const char* B = A + kBfBM * kBfRow;
#pragma unroll
        for (int ks = 0; ks < kBfKS / 16; ks++) {
            // A: m16 tile i's x4 (rows +0..7 / +8..15, k +0..7 / +8..8): lanes 0-15 rows, 16-31 the high k half
            uint32_t af[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int r = wm * 64 + i * 16 + (lane & 15);
                ldsm4(af[i], A + r * kBfRow + ks * 32 + (lane >> 4) * 16);
            }
            // B: two n8 tiles a x4 (n rows +0..7 at k +0, +8; the next n8 tile's the same)
            uint32_t bf[4][2];
#pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                uint32_t r4[4];
                const int n = wn * 32 + jj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(r4, B + n * kBfRow + ks * 32 + ((lane >> 3) & 1) * 16);
                bf[2 * jj][0] = r4[0];
                bf[2 * jj][1] = r4[1];
                bf[2 * jj + 1][0] = r4[2];
                bf[2 * jj + 1][1] = r4[3];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (wm * 64 + i * 16 >= mvalid) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) mma_bf16(acc[i][j], af[i], bf[j]);
            }
        }
    }
    wait_groups<0>();
    char* yb = reinterpret_cast<char*>(a.y);
    const int esz = a.y_dtype == 2 ? 4 : 2;
    void* y = yb + size_t(grp) * a.gy * esz;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * 64 + i * 16 + g + (c >> 1) * 8;
                if (m >= a.M) continue;
                const int n = n0 + wn * 32 + j * 8 + 2 * t + (c & 1);
                bf_store(a, y, m, n, acc[i][j][c]);
            }
}

// the prefill kernels' dynamic shared memory (mx.zig pf_*_smem)
#ifndef PF_SWEEP
static_assert(pf::kGemmSmem == 76800 && pf::kGuSmem == 79872 && pf::kDnSmem == 53760 && pf::kBfSmem == 81920, "prefill shared memory");
#endif

// Decode tensor-core kernels ("dec"): a decode round's few rows on the MMA with the operands swapped (the m16 side 16
// weight rows, the n8 side 8 activation rows), the weights streamed through a per-warp ring of cp.async stages in
// shared memory (16-byte copies, 128 bytes of each of the tile's 16 rows a stage, several stages in flight: the
// GB10's memory wants many bytes outstanding and whole 128-byte runs of a row) and read back with ldmatrix. A warp
// owns a 16-row weight tile over a K chunk; the CTA's `ks` warps on one tile, and `cs` CTAs on it (grid z), split K
// as the layer's shape alone decides, the warps' partials summed in warp order in shared memory and the CTAs' in
// grid-z order by the last CTA to finish (a counter a tile group, reset by that CTA). Every (weight row, activation
// row) result is its own MMA output and the K order is fixed, so a row's bits never depend on the other rows of a
// launch or on how many there are.
//   - FP8 (fmt 0): 4 K blocks a stage, the MMA's natural K order;
//   - MXFP4 (fmt 1): 8 K blocks a stage. Each E2M1 code goes to the E4M3 byte (code & 7) << 2 | sign << 7, which is
//     the value times 2^-6 exactly (the subnormal 0.5 too), the 2^6 folded into the block's power-of-two weight scale;
//     a thread's 4 bytes (values 8t .. 8t + 7 of its row's block) feed k slots 4t + i (the even values) and
//     16 + 4t + i (the odd ones), the activation's bytes permuted the same way;
//   - bf16 (fmt 2): 64 values a stage (four m16n8k16 steps, k in order), the sum in the MMA's fp32 C.
// FP8 / FP4, a K block: acc = fmaf(mma_sum, scale_a * scale_w, acc), both scales powers of two (kernel.py's
// accum += C_local * scale_a * scale_b, one rounding).
namespace dec {

#ifndef TF_DSN_DEC_LDG
#define TF_DSN_DEC_LDG 1
#endif
#if TF_DSN_DEC_LDG
#define DEC_LD(p) __ldg(p)
#else
#define DEC_LD(p) (*(p))
#endif

// the two configurations (a weight row's bytes a stage, 64 or 128; the ring's stages): the dense linears' (few long
// warps: a deep ring each) and the experts' (many short ones: more resident warps)
#ifndef TF_DSN_DEC_RB_D
#define TF_DSN_DEC_RB_D 64
#endif
#ifndef TF_DSN_DEC_NS_D
#define TF_DSN_DEC_NS_D 4
#endif
#ifndef TF_DSN_DEC_RB_E
#define TF_DSN_DEC_RB_E 64
#endif
#ifndef TF_DSN_DEC_NS_E
#define TF_DSN_DEC_NS_E 2
#endif
#ifndef TF_DSN_DEC_RB_G
#define TF_DSN_DEC_RB_G 128
#endif
#ifndef TF_DSN_DEC_NS_G
#define TF_DSN_DEC_NS_G 2
#endif
// (the experts' gate / up apart: rows of 2560 bytes take 128-byte stages better, down's of 576 bytes 64-byte ones)
constexpr int kRbD = TF_DSN_DEC_RB_D, kNsD = TF_DSN_DEC_NS_D, kRbE = TF_DSN_DEC_RB_E, kNsE = TF_DSN_DEC_NS_E;
constexpr int kRbG = TF_DSN_DEC_RB_G, kNsG = TF_DSN_DEC_NS_G;
#ifndef TF_DSN_DEC_SCB
#define TF_DSN_DEC_SCB RB
#endif
template <int RB>
__host__ __device__ constexpr int slotBytes() { return 16 * RB + TF_DSN_DEC_SCB; }  // a 16-row tile's stage and its scales

// programmatic dependent launch: let the stream's next kernel start (its weights' prologue) and wait for this one's
// predecessor (every global write before it visible); both no-ops for a launch without the attribute
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp4(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr)
                 : "memory");
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
// four E2M1 codes (the low / high nibbles of r's bytes) as E4M3 bytes of the values times 2^-6
__device__ __forceinline__ uint32_t fp4lo(uint32_t r) { return ((r << 2) & 0x1C1C1C1Cu) | ((r << 4) & 0x80808080u); }
__device__ __forceinline__ uint32_t fp4hi(uint32_t r) { return ((r >> 2) & 0x1C1C1C1Cu) | (r & 0x80808080u); }

// a tile row's 16-byte chunk c in a stage, swizzled so ldmatrix's 8 rows of one chunk take 8 different bank groups
template <int kRowB>
__device__ __forceinline__ int swz(int row, int c) {
    return kRowB == 64 ? row * kRowB + ((c ^ ((row >> 1) & 3)) << 4) : row * kRowB + ((c ^ (row & 7)) << 4);
}

template <int FMT>
__device__ __forceinline__ float wscale(uint32_t e) {
    // FP4: the 2^6 of the E4M3 codes folded in
    return FMT == 1 ? __uint_as_float((e + 6) << 23) : __uint_as_float(e << 23);
}

// a format's K values a stage, and a row's bytes for K values (FMT 3: NVFP4, the row's bytes as MXFP4's)
template <int FMT>
__host__ __device__ constexpr int64_t rowBytes(int64_t k) { return FMT == 1 || FMT == 3 ? k / 2 : FMT == 0 ? k : 2 * k; }

// One warp: weight rows n0 .. n0 + 15 of NM matrices (w[m], row stride ldw bytes; scales ws[m], row stride ldws
// bytes), stages st0 .. st1 - 1 of a row of rb bytes (the last stage may hold 64 bytes: K a multiple of 64 values for
// FP4, 64 bytes FP8 / bf16), activation rows i < cnt (row rows_of(i); FP8 / FP4: codes xq, row stride ldq bytes, and
// scales s, lds floats; bf16: xq bf16, ldq elements; FMT 3, NVFP4: codes xq, ldq bytes, and E4M3 scales s (as bytes), lds
// bytes) in RG groups of 8. acc[m][g][c]: weight row n0 + lane / 4 + 8 (c >> 1), activation row 8 g + 2 (lane % 4) +
// (c & 1). FMT 3's sums come unscaled by the expert's alpha.
template <int RB, int NS, int FMT, int NM, int RG, typename RowOf>
__device__ __forceinline__ void core(const uint8_t* const (&w)[NM], const uint8_t* const (&ws)[NM], int64_t ldw,
                                     int64_t ldws, int64_t rb, int n0, int st0, int st1, int sst, const void* xq, int64_t ldq,
                                     const float* s, int64_t lds, int cnt, RowOf rows_of, uint8_t* ring,
                                     float (&acc)[NM][RG][4]) {
    constexpr int kRowB = RB, kChunks = RB / 16, kTileB = 16 * RB, kSlotB = slotBytes<RB>(), kStages = NS;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < NM; m++)
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[m][r][c] = 0.f;
    const int ng = (cnt + 7) >> 3;  // the row groups in use (warp-uniform)
    // this thread's activation rows: its B row g and its C rows 2t, 2t + 1 of each group (past cnt: the first row's)
    const uint8_t* brow[RG];
    int64_t c0[RG], c1[RG];
#pragma unroll
    for (int r = 0; r < RG; r++) {
        const int i = r * 8 + g, j0 = r * 8 + 2 * t, j1 = j0 + 1;
        const int64_t esz = FMT == 2 ? 2 : 1;
        brow[r] = reinterpret_cast<const uint8_t*>(xq) + (cnt ? int64_t(rows_of(i < cnt ? i : 0)) * ldq * esz : 0);
        if (FMT == 3) {
            // the block-scaled MMA takes the scales of the thread's B row (row i's), not of its C rows
            c0[r] = cnt ? int64_t(rows_of(i < cnt ? i : 0)) * lds : 0;
            c1[r] = 0;
        } else {
            c0[r] = cnt ? int64_t(rows_of(j0 < cnt ? j0 : 0)) * lds : 0;
            c1[r] = cnt ? int64_t(rows_of(j1 < cnt ? j1 : 0)) * lds : 0;
        }
    }
    const uint32_t ring_s = smem_u32(ring);
    // ldmatrix: matrices (rows 0-7, chunk b), (rows 8-15, chunk b), (rows 0-7, b + 1), (rows 8-15, b + 1)
    const int lrow = (lane & 7) + 8 * ((lane >> 3) & 1), lch = lane >> 4;
    auto issue = [&](int st, int slot) {
        if (st < st1) {
            const int vc = int(rb - int64_t(st) * kRowB < kRowB ? rb - int64_t(st) * kRowB : kRowB) >> 4;  // valid chunks
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const uint32_t base = ring_s + (slot * NM + m) * kSlotB;
#pragma unroll
                for (int i = 0; i < 16 * kChunks / 32; i++) {
                    const int c = lane + 32 * i, row = c / kChunks, ch = c % kChunks;
                    if (ch < vc)
                        cp16(base + swz<kRowB>(row, ch), w[m] + int64_t(n0 + row) * ldw + int64_t(st) * kRowB + ch * 16);
                }
                if (FMT == 1 || FMT == 3) {
                    // 16 rows x kChunks block scales: kChunks / 4 words a row (a short stage's first only)
                    constexpr int wpr = kChunks / 4;
                    const int row = lane / wpr, h = lane % wpr;
                    if (lane < 16 * wpr && h * 4 < vc)
                        cp4(base + kTileB + row * kChunks + h * 4, ws[m] + int64_t(n0 + row) * ldws + st * kChunks + h * 4);
                } else if (FMT == 0 && lane == 0) {
                    // the tile's kChunks / 2 block scales: the word holding them
                    cp4(base + kTileB, ws[m] + int64_t(n0 >> 5) * ldws + ((st * (kChunks / 2)) & ~3));
                }
            }
        }
        cp_commit();
    };
#pragma unroll
    for (int i = 0; i < kStages - 1; i++) issue(st0 + i * sst, i);
    // the weights' first stages are in flight: now the activations (the predecessor's output) may be read
    pdl_wait();
    int slot = 0;
    for (int st = st0; st < st1; st += sst) {
        cp_wait<kStages - 2>();
        __syncwarp();
        issue(st + (kStages - 1) * sst, (slot + kStages - 1) % kStages);
        const int vc_rt = int(rb - int64_t(st) * kRowB < kRowB ? rb - int64_t(st) * kRowB : kRowB) >> 4;
        // a whole stage with compile-time counts, a short last stage (64 bytes) apart
        auto consume = [&](auto vcc) {
        constexpr int vc = decltype(vcc)::value;
        const uint8_t* sbase = ring + slot * NM * kSlotB;
        uint32_t r[NM][kChunks / 2][4];
#pragma unroll
        for (int m = 0; m < NM; m++) {
            const uint32_t base = ring_s + (slot * NM + m) * kSlotB;
#pragma unroll
            for (int p = 0; p < kChunks / 2; p++)
                if (2 * p < vc) ldsm4(r[m][p], base + swz<kRowB>(lrow, 2 * p + lch));
        }
        if (FMT == 3) {
            // a 64-input step a pair of chunks: one MMA, the scales in hardware
            const uint8_t* srow = sbase + kTileB + ((t & 1) ? g + 8 : g) * kChunks;
#pragma unroll
            for (int p = 0; p < kChunks / 2; p++) {
                if (2 * p >= vc) break;
                const int64_t kg = int64_t(st) * (kChunks / 2) + p;
                const uint32_t sa = nv_scales(srow[2 * p], srow[2 * p + 1]);
#pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    if (rg >= ng) break;
                    const uint32_t b0 = DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + kg * 32 + 4 * t));
                    const uint32_t b1 = DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + kg * 32 + 16 + 4 * t));
                    const uint32_t sb = DEC_LD(reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(s) + c0[rg] + kg * 4));
                    mma_fp4(acc[0][rg], r[0][p], b0, b1, sa, sb);
                }
            }
        } else if (FMT == 2) {
            // 16-value steps: chunks 2p, 2p + 1
#pragma unroll
            for (int p = 0; p < kChunks / 2; p++) {
                if (2 * p >= vc) break;
                const int64_t k = int64_t(st) * (kRowB / 2) + p * 16 + 2 * t;  // elements
#pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    if (rg >= ng) break;
                    const uint32_t b[2] = {DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + k * 2)),
                                           DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + (k + 8) * 2))};
#pragma unroll
                    for (int m = 0; m < NM; m++) mma_bf16(acc[m][rg], r[m][p], b);
                }
            }
        } else {
            constexpr int NB = FMT == 1 ? kChunks : kChunks / 2;  // K blocks a whole stage
            const int nb = FMT == 1 ? vc : vc / 2;
            uint32_t sw[NM][2];  // FP4: rows g, g + 8, blocks 0-3 / 4-7 (bytes); FP8: the tile's word
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const uint8_t* sc = sbase + m * kSlotB + kTileB;
                if (FMT == 1) {
                    sw[m][0] = *reinterpret_cast<const uint32_t*>(sc + g * kChunks + 0);
                    sw[m][1] = kChunks > 4 ? *reinterpret_cast<const uint32_t*>(sc + g * kChunks + 4) : 0u;
                } else {
                    sw[m][0] = *reinterpret_cast<const uint32_t*>(sc);
                }
            }
            uint32_t sw8[NM][2];
#pragma unroll
            for (int m = 0; m < NM; m++) {
                const uint8_t* sc = sbase + m * kSlotB + kTileB;
                if (FMT == 1) {
                    sw8[m][0] = *reinterpret_cast<const uint32_t*>(sc + (g + 8) * kChunks + 0);
                    sw8[m][1] = kChunks > 4 ? *reinterpret_cast<const uint32_t*>(sc + (g + 8) * kChunks + 4) : 0u;
                }
            }
#pragma unroll
            for (int j = 0; j < NB; j++) {
                if (j >= nb) break;
                const int kb = st * NB + j;
                uint32_t a[NM][4];
                float swg[NM], swh[NM];
#pragma unroll
                for (int m = 0; m < NM; m++) {
                    if (FMT == 1) {
                        const uint32_t* rr = r[m][j >> 1];
                        const uint32_t x0 = rr[(j & 1) * 2], x1 = rr[(j & 1) * 2 + 1];
                        a[m][0] = fp4lo(x0);
                        a[m][1] = fp4lo(x1);
                        a[m][2] = fp4hi(x0);
                        a[m][3] = fp4hi(x1);
                        swg[m] = wscale<1>((sw[m][j >> 2] >> (8 * (j & 3))) & 0xff);
                        swh[m] = wscale<1>((sw8[m][j >> 2] >> (8 * (j & 3))) & 0xff);
                    } else {
#pragma unroll
                        for (int c = 0; c < 4; c++) a[m][c] = r[m][j][c];
                        swg[m] = swh[m] = wscale<0>((sw[m][0] >> (8 * (((st * NB) & 3) + j))) & 0xff);
                    }
                }
#pragma unroll
                for (int rg = 0; rg < RG; rg++) {
                    if (rg >= ng) break;
                    uint32_t b[2];
                    if (FMT == 1) {
                        const uint2 v = DEC_LD(reinterpret_cast<const uint2*>(brow[rg] + int64_t(kb) * 32 + 8 * t));
                        b[0] = prmt(v.x, v.y, 0x6420);
                        b[1] = prmt(v.x, v.y, 0x7531);
                    } else {
                        b[0] = DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + int64_t(kb) * 32 + 4 * t));
                        b[1] = DEC_LD(reinterpret_cast<const uint32_t*>(brow[rg] + int64_t(kb) * 32 + 16 + 4 * t));
                    }
                    const float sa0 = DEC_LD(s + c0[rg] + kb), sa1 = DEC_LD(s + c1[rg] + kb);
#pragma unroll
                    for (int m = 0; m < NM; m++) {
                        float d[4];
                        mma_e4m3(d, a[m], b);
                        acc[m][rg][0] = fmaf(d[0], sa0 * swg[m], acc[m][rg][0]);
                        acc[m][rg][1] = fmaf(d[1], sa1 * swg[m], acc[m][rg][1]);
                        acc[m][rg][2] = fmaf(d[2], sa0 * swh[m], acc[m][rg][2]);
                        acc[m][rg][3] = fmaf(d[3], sa1 * swh[m], acc[m][rg][3]);
                    }
                }
            }
        }
        };
        if (vc_rt == kChunks) consume(std::integral_constant<int, kChunks>{});
        else consume(std::integral_constant<int, 4>{});
        slot = (slot + 1) % kStages;
    }
    cp_wait<0>();
    __syncwarp();
}

// The partials of a tile's `ks` warps (warp w: tile w / ks, chunk w % ks) summed in chunk order into the chunk-0
// warp's acc (true for that warp). Every warp of the CTA calls it (it holds __syncthreads).
template <int NM, int RG>
__device__ __forceinline__ bool reduce(float (&acc)[NM][RG][4], int ks, uint8_t* smem, int warp_bytes) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (ks == 1) return true;
    float* mine = reinterpret_cast<float*>(smem + warp * warp_bytes);
#pragma unroll
    for (int m = 0; m < NM; m++)
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) mine[((m * RG + r) * 4 + c) * 32 + lane] = acc[m][r][c];
    __syncthreads();
    if (warp % ks != 0) return false;
    for (int w2 = 1; w2 < ks; w2++) {
        const float* o = reinterpret_cast<const float*>(smem + (warp + w2) * warp_bytes);
#pragma unroll
        for (int m = 0; m < NM; m++)
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[m][r][c] += o[((m * RG + r) * 4 + c) * 32 + lane];
    }
    return true;
}

// a warp's bytes of the CTA's dynamic shared memory: its ring, or its partials (the larger)
template <int RB, int NS>
__host__ __device__ constexpr int warpBytes(int nm, int rg) {
    return NS * nm * slotBytes<RB>() > nm * rg * 4 * 32 * 4 ? NS * nm * slotBytes<RB>() : nm * rg * 4 * 32 * 4;
}

// a K split's part c of `parts` of a stage count: stages lo, lo + step, ... below hi. Interleaved (the default): a
// CTA's warps on a tile read neighbouring stages of its rows at once (longer runs of each row in flight); else
// contiguous chunks.
#ifndef TF_DSN_DEC_INTERLEAVE
#define TF_DSN_DEC_INTERLEAVE 1
#endif
__device__ __forceinline__ void chunk(int nst, int parts, int c, int& lo, int& hi, int& step) {
    if (TF_DSN_DEC_INTERLEAVE) {
        lo = c;
        hi = nst;
        step = parts;
        return;
    }
    const int per = (nst + parts - 1) / parts;
    lo = min(nst, c * per);
    hi = min(nst, lo + per);
    step = 1;
}

}  // namespace dec

// The dense decode linears' arguments (FP8 / FP4 / bf16 weights): y [M, N] (row stride ldy, y_dtype; accumulate: fp32
// y +=) = x [M, K] . w [N, K]^T. FP8 / FP4: x quantized (q, s; rows: optional gather of the launch's rows); bf16: x
// bf16 at q (ldq elements), the bf16 GEMM's epilogue (scale, round_first). ks warps a tile in a CTA (blockDim / 32
// warps, blockDim / 32 / ks tiles), cs CTAs a tile group (grid z); cs > 1: the CTAs' partials in z [cs, M, N] and the
// last one's sum, cnt [the tile groups] int zeros (left zero).
struct DecArgs {
    const uint8_t* q;
    const float* s;
    const int* rows;
    int64_t ldq, lds;
    const uint8_t* w;
    const uint8_t* ws;
    int64_t ldw, ldws;  // bytes
    int fmt;
    int M, N, K;
    int ks, cs;
    float* z;
    int* cnt;
    void* y;
    int64_t ldy;
    int y_dtype;
    int accumulate;
    float scale;
    int round_first;
    // block-diagonal groups (wo_a): output columns n read the inputs of group n / gn, at q + group * gq bytes and
    // s + group * gs floats (gn 0: one group)
    int gn;
    int64_t gq, gs;
};

template <int FMT, int RG>
__device__ __forceinline__ void dec_dense(const DecArgs& a, int bx) {
    static_assert(FMT != 3, "NVFP4 is the routed experts' alone");
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    __shared__ int last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbD, dec::kNsD>(1, RG);
    const int tiles_cta = (blockDim.x >> 5) / a.ks;
    const int n0 = (bx * tiles_cta + warp / a.ks) * 16;
    const bool on = n0 < a.N;
    int lo = 0, hi = 0, sst = 1;
    const int64_t rb = dec::rowBytes<FMT>(a.K);
    if (on) dec::chunk(int((rb + dec::kRbD - 1) / dec::kRbD), a.ks * a.cs, blockIdx.z * a.ks + warp % a.ks, lo, hi, sst);
    float acc[1][RG][4];
    const uint8_t* w[1] = {a.w};
    const uint8_t* ws[1] = {a.ws};
    const int grp = a.gn && on ? n0 / a.gn : 0;
    dec::core<dec::kRbD, dec::kNsD, FMT, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.q + grp * a.gq, a.ldq,
                          a.s ? a.s + grp * a.gs : a.s, a.lds, a.M,
                          [&](int i) { return a.rows ? a.rows[i] : i; }, dsm + warp * wb, acc);
    const bool lead = dec::reduce<1, RG>(acc, a.ks, dsm, wb);
    if (a.cs > 1) {
        // the CTA's partial into z, then the last CTA of the tile group sums the cs partials in order
        if (lead && on) {
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
                    if (m < a.M) a.z[(size_t(blockIdx.z) * a.M + m) * a.N + n] = acc[0][r][c];
                }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const int prev = atomicAdd(a.cnt + bx, 1);
            last = prev == a.cs - 1;
            if (last) a.cnt[bx] = 0;
        }
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (!lead || !on) return;
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
                if (m >= a.M) continue;
                float v = __ldcg(a.z + size_t(m) * a.N + n);
                for (int p = 1; p < a.cs; p++) v += __ldcg(a.z + (size_t(p) * a.M + m) * a.N + n);
                acc[0][r][c] = v;
            }
    } else if (!lead || !on) {
        return;
    }
#pragma unroll
    for (int r = 0; r < RG; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int m = r * 8 + 2 * t + (c & 1), n = n0 + g + 8 * (c >> 1);
            if (m >= a.M) continue;
            float v = acc[0][r][c];
            if (FMT == 2) {
                if (a.round_first) v = bf16r(v);
                if (a.scale != 1.f) v = v * a.scale;
            }
            if (a.accumulate) reinterpret_cast<float*>(a.y)[size_t(m) * a.ldy + n] += v;
            else store_y(a.y, a.y_dtype, size_t(m) * a.ldy + n, v);
        }
}

// gate / up (GuArgs; splits: the warps a tile and matrix): the pack's 16-column tiles of expert act[blockIdx.y], each
// tile's gate (w1) and up (w3) on warps of their own (warp w: tile w / (2 splits), matrix w / splits % 2, chunk
// w % splits), the tile's partials summed in chunk order, then the epilogue on the gate's lead warp. An expert's
// entries go in passes of 8 RG.
template <int RG>
__device__ __forceinline__ void dec_gu(const GuArgs& a) {
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbG, dec::kNsG>(1, RG);
    const int ks = a.splits, per_tile = 2 * ks;
    // the experts in reverse id order: the shared expert (the last id, the L2 prefetch's) first
    const int na = a.nact[0];
    // Track C (docs/UNIFIED-DECODE-IMPROVEMENTS.md, TF_DS_EXPERT_EXIT=1): the grid's rows past the active experts
    // (it is sized min(experts, rows x slots) on the host, which cannot see the count) leave at once, the whole CTA
    // together (no barrier is entered, nothing is written, no other CTA waits on them)
    if (a.early_exit && int(blockIdx.y) >= na) return;
    const bool live = int(blockIdx.y) < na;
    const int e = live ? a.act[na - 1 - blockIdx.y] : 0;
    const int tiles_cta = (blockDim.x >> 5) / per_tile;
    const int n0 = (blockIdx.x * tiles_cta + warp / per_tile) * 16;
    const int mat = warp / ks % 2, ch = warp % ks;
    const int ncols = a.n_tiles * 32;
    bool on = live && n0 < ncols;
    if (on && a.owner && n0 / 32 >= a.n_tiles - a.rest_tiles && a.owner[e] == 0) on = false;
    const bool fp8 = e == a.shared;
    const int base = live ? a.off[e] : 0, all = live ? a.off[e + 1] - base : 0;
    const uint8_t* w[1] = {reinterpret_cast<const uint8_t*>(mat ? a.w3[e] : a.w1[e])};
    const uint8_t* ws[1] = {reinterpret_cast<const uint8_t*>(mat ? a.s3[e] : a.s1[e])};
    const bool nv = a.fmt == 2 && !fp8;
    const float ag = nv && live ? a.a1[e] : 1.f, au = nv && live ? a.a3[e] : 1.f;
    for (int p0 = 0; p0 < max(all, 1); p0 += 8 * RG) {
        const int cnt = on ? min(8 * RG, all - p0) : 0;
        auto rows_of = [&](int i) { return a.erow[base + p0 + i]; };
        float acc[1][RG][4];
        int lo = 0, hi = 0, sst = 1;
        if (nv) {
            const int64_t rb = dec::rowBytes<3>(a.D);
            if (on) dec::chunk(int((rb + dec::kRbG - 1) / dec::kRbG), ks, ch, lo, hi, sst);
            auto ent_of = [&](int i) { return a.ent[base + p0 + i]; };
            dec::core<dec::kRbG, dec::kNsG, 3, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.nq, a.ldnq,
                                reinterpret_cast<const float*>(a.ns), a.ldns, cnt, ent_of, dsm + warp * wb, acc);
        } else if (fp8) {
            const int64_t rb = dec::rowBytes<0>(a.D);
            if (on) dec::chunk(int((rb + dec::kRbG - 1) / dec::kRbG), ks, ch, lo, hi, sst);
            dec::core<dec::kRbG, dec::kNsG, 0, 1, RG>(w, ws, a.sldw, a.sldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        } else {
            const int64_t rb = dec::rowBytes<1>(a.D);
            if (on) dec::chunk(int((rb + dec::kRbG - 1) / dec::kRbG), ks, ch, lo, hi, sst);
            dec::core<dec::kRbG, dec::kNsG, 1, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        }
        // every warp's partials into its region, then the tile's lead warp (matrix 0, chunk 0) sums each matrix's
        float* mine = reinterpret_cast<float*>(dsm + warp * wb);
#pragma unroll
        for (int r = 0; r < RG; r++)
#pragma unroll
            for (int c = 0; c < 4; c++) mine[(r * 4 + c) * 32 + lane] = acc[0][r][c];
        __syncthreads();
        if (warp % per_tile == 0 && on) {
            float gs[RG][4], us[RG][4];
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) gs[r][c] = us[r][c] = 0.f;
            for (int w2 = 0; w2 < ks; w2++) {
                const float* og = reinterpret_cast<const float*>(dsm + (warp + w2) * wb);
                const float* ou = reinterpret_cast<const float*>(dsm + (warp + ks + w2) * wb);
#pragma unroll
                for (int r = 0; r < RG; r++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        gs[r][c] = w2 ? gs[r][c] + og[(r * 4 + c) * 32 + lane] : og[(r * 4 + c) * 32 + lane];
                        us[r][c] = w2 ? us[r][c] + ou[(r * 4 + c) * 32 + lane] : ou[(r * 4 + c) * 32 + lane];
                    }
            }
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int i = r * 8 + 2 * t + (c & 1);
                    if (i >= cnt) continue;
                    gu_store(a, a.ent[base + p0 + i], n0 + g + 8 * (c >> 1), gs[r][c] * ag, us[r][c] * au);
                }
        }
        __syncthreads();
    }
}

// down (DnArgs; splits: the warps a tile): 16 output columns a tile of expert act[blockIdx.y]
template <int RG>
__device__ __forceinline__ void dec_dn(const DnArgs& a) {
    // (no launch_dependents: the next kernel may be any of the round's, programmatic or not, and some of those
    // complete without waiting for their predecessor, which would let a later kernel overtake this one; only
    // tf_dsn_dec_quant, always followed by one of these kernels, which waits, lets its successor start early)
    extern __shared__ __align__(16) uint8_t dsm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wb = dec::warpBytes<dec::kRbE, dec::kNsE>(1, RG);
    if (a.early_exit && int(blockIdx.y) >= a.nact[0]) return;  // Track C: as dec_gu
    const bool live = int(blockIdx.y) < a.nact[0];
    const int e = live ? a.act[blockIdx.y] : 0;
    const int tiles_cta = (blockDim.x >> 5) / a.splits;
    const int n0 = (blockIdx.x * tiles_cta + warp / a.splits) * 16;
    const bool on = live && n0 < a.N;
    const bool fp8 = e == a.shared;
    const int base = live ? a.off[e] : 0, all = live ? a.off[e + 1] - base : 0;
    const uint8_t* w[1] = {reinterpret_cast<const uint8_t*>(a.w2[e])};
    const uint8_t* ws[1] = {reinterpret_cast<const uint8_t*>(a.s2[e])};
    const bool nv = a.fmt == 2 && !fp8;
    const float al = nv && live ? a.a2[e] : 1.f;
    for (int p0 = 0; p0 < max(all, 1); p0 += 8 * RG) {
        const int cnt = on ? min(8 * RG, all - p0) : 0;
        auto rows_of = [&](int i) { return a.ent[base + p0 + i]; };
        float acc[1][RG][4];
        int lo = 0, hi = 0, sst = 1;
        if (nv) {
            const int64_t rb = dec::rowBytes<3>(a.I);
            if (on) dec::chunk(int((rb + dec::kRbE - 1) / dec::kRbE), a.splits, warp % a.splits, lo, hi, sst);
            dec::core<dec::kRbE, dec::kNsE, 3, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.nq, a.ldnq,
                                reinterpret_cast<const float*>(a.ns), a.ldns, cnt, rows_of, dsm + warp * wb, acc);
        } else if (fp8) {
            const int64_t rb = dec::rowBytes<0>(a.I);
            if (on) dec::chunk(int((rb + dec::kRbE - 1) / dec::kRbE), a.splits, warp % a.splits, lo, hi, sst);
            dec::core<dec::kRbE, dec::kNsE, 0, 1, RG>(w, ws, a.sldw, a.sldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        } else {
            const int64_t rb = dec::rowBytes<1>(a.I);
            if (on) dec::chunk(int((rb + dec::kRbE - 1) / dec::kRbE), a.splits, warp % a.splits, lo, hi, sst);
            dec::core<dec::kRbE, dec::kNsE, 1, 1, RG>(w, ws, a.ldw, a.ldws, rb, on ? n0 : 0, lo, hi, sst, a.q, a.ldq, a.s, a.lds, cnt, rows_of,
                                dsm + warp * wb, acc);
        }
        const bool lead = dec::reduce<1, RG>(acc, a.splits, dsm, wb);
        if (lead && on) {
#pragma unroll
            for (int r = 0; r < RG; r++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int i = r * 8 + 2 * t + (c & 1);
                    if (i >= cnt) continue;
                    const float v = acc[0][r][c] * al;
                    a.out[size_t(a.ent[base + p0 + i]) * a.ldo + n0 + g + 8 * (c >> 1)] = a.round_bf16 ? bf16r(v) : v;
                }
        }
        if (a.splits > 1) __syncthreads();  // the regions reused by the next pass
    }
}

// Up to four dense linears of one format and row count in one launch (the linears of one input: a launch and its
// tail instead of each's): linear p takes the CTAs cta0[p] .. cta0[p + 1] - 1, each as its own launch would.
struct DecMulti {
    DecArgs p[4];
    int n;
    int cta0[5];
};

template <int FMT, int RG>
__device__ __forceinline__ void dec_multi(const DecMulti& m) {
    int p = 0;
    while (p + 1 < m.n && int(blockIdx.x) >= m.cta0[p + 1]) p++;
    dec_dense<FMT, RG>(m.p[p], blockIdx.x - m.cta0[p]);
}

#define TF_DSN_DEC(RG)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_fp8_r##RG##_kernel(DecMulti a) { dec_multi<0, RG>(a); } \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_fp4_r##RG##_kernel(DecMulti a) { dec_multi<1, RG>(a); } \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_bf_r##RG##_kernel(DecMulti a) { dec_multi<2, RG>(a); }  \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_gu_r##RG##_kernel(GuArgs a) { dec_gu<RG>(a); }         \
    extern "C" __global__ void __launch_bounds__(256) tf_dsn_dec_dn_r##RG##_kernel(DnArgs a) { dec_dn<RG>(a); }
TF_DSN_DEC(1)
TF_DSN_DEC(2)
TF_DSN_DEC(4)
TF_DSN_DEC(8)
#undef TF_DSN_DEC

// act_quant (tf_dsn_act_quant's codes and scales, the same arithmetic) for decode launches: an ordinary launch that
// lets the next kernel (a decode tensor-core kernel, launched as a programmatic dependent launch) start its weights'
// prologue while this runs. (Itself not a dependent launch: a predecessor of another kind may let it start early yet
// complete before its own predecessor, so only an ordinary launch here keeps every earlier kernel done.)
extern "C" __global__ void tf_dsn_dec_quant_kernel(const void* __restrict__ x, int x_dtype, int64_t ldx, int M, int K,
                                                   uint8_t* __restrict__ q, int64_t ldq, float* __restrict__ s,
                                                   int64_t lds) {
    dec::pdl_launch();
    dec::pdl_wait();
    const int groups = K / kBlk;
    const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= int64_t(M) * groups) return;
    const int m = int(warp / groups), g = int(warp % groups);
    const int k = g * kBlk + lane;
    const float v = bf16r(load_x(x, x_dtype, size_t(m) * ldx + k));
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float sc = round_scale(fmaxf(amax, 1e-4f));
    q[size_t(m) * ldq + k] = to_e4m3(v / sc);
    if (lane == 0) s[size_t(m) * lds + g] = sc;
}

// NVFP4's activations: each routed entry's input as its expert's own codes and scales (nvfp4q.cuh, ashhart/TensorFold, the
// format the checkpoint's input_scale names): per 16 inputs of the row, an E4M3 scale of g * amax / 6 and E2M1 codes of
// x * g / scale, g = 1 / input_scale (a table over the experts). Entry i = r * S + s reads pick[i] and x's row r
// (by_entry: its own row, down's intermediate); entries past the routed experts are skipped. One thread a block of 16.
extern "C" __global__ void __launch_bounds__(128) tf_dsn_nv_quant_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                                                         const int* __restrict__ pick, int S, int by_entry,
                                                                         int P, int K, const float* __restrict__ gt,
                                                                         int routed, uint8_t* __restrict__ nq, int64_t ldnq,
                                                                         uint8_t* __restrict__ ns, int64_t ldns) {
    dec::pdl_launch();
    dec::pdl_wait();
    const int groups = K / 16;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(P) * groups) return;
    const int entry = int(i / groups), blk = int(i % groups);
    const int e = pick[entry];
    if (e < 0 || e >= routed) return;
    const float gv = gt[e];
    const int row = by_entry ? entry : entry / S;
    // bf16 is fp32's top half
    const uint32_t* src = reinterpret_cast<const uint32_t*>(x + size_t(row) * ldx + blk * 16);
    uint32_t bits[8];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        bits[j] = src[j];
        const float lo = __uint_as_float(bits[j] << 16), hi = __uint_as_float(bits[j] & 0xffff0000u);
        amax = fmaxf(amax, fmaxf(fabsf(lo), fabsf(hi)));
    }
    const __nv_fp8_storage_t sf8 = __nv_cvt_float_to_fp8(gv * (amax * (1.0f / 6.0f)), __NV_SATFINITE, __NV_E4M3);
    const float sf = __half2float(__half(__nv_cvt_fp8_to_halfraw(sf8, __NV_E4M3)));
    const float mul = sf != 0.f ? gv / sf : 0.f;
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float lo = __uint_as_float(bits[j] << 16), hi = __uint_as_float(bits[j] & 0xffff0000u);
        w[j >> 2] |= (nv_e2m1(lo * mul) | (nv_e2m1(hi * mul) << 4)) << (8 * (j & 3));
    }
    *reinterpret_cast<uint2*>(nq + size_t(entry) * ldnq + blk * 8) = make_uint2(w[0], w[1]);
    ns[size_t(entry) * ldns + blk] = uint8_t(sf8);
}

// The active experts' down weights touched into L2 while the 2D exchange of the intermediate runs (it leaves DRAM
// idle), in down's CTA order (act[] ascending), up to `budget` bytes: 8 KiB bulk L2 prefetches, one block of 128
// threads, a wave of 128 every wave_ns (DRAM's queue about a wave deep). w2 [E] pointers, the routed experts' down
// `bytes`, the shared expert's (id `shared`) `sbytes`. Writes nothing.
extern "C" __global__ void __launch_bounds__(128) tf_dsn_dec_l2pf_kernel(const int* __restrict__ act,
                                                                         const int* __restrict__ nact,
                                                                         const uint64_t* __restrict__ w2, long long bytes,
                                                                         long long sbytes, int shared, long long budget,
                                                                         long long wave_ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const long long chunk = 8 << 10;
    const int n = nact[0];
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    long long done = 0, wave = 0;
    int i = 0;
    long long off = 0;  // within expert act[i]
    while (i < n && done < budget) {
        if (wave_ns > 0) {
            __syncthreads();
            if (threadIdx.x == 0) {
                for (;;) {
                    unsigned long long t;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                    if (t - t0 >= (unsigned long long)(wave * wave_ns)) break;
                    __nanosleep(256);
                }
            }
            __syncthreads();
            ++wave;
        }
        // this wave's blockDim chunks: thread j takes the j-th chunk after (i, off)
        int ii = i;
        long long oo = off + (long long)threadIdx.x * chunk;
        long long len = 0;
        while (ii < n) {
            len = (act[ii] == shared ? sbytes : bytes) / 16 * 16;
            if (oo < len) break;
            oo -= len;
            ++ii;
        }
        if (ii < n && done + (long long)threadIdx.x * chunk < budget) {
            const long long left = len - oo;
            const unsigned size = (unsigned)(left < chunk ? left : chunk);
            const char* p = reinterpret_cast<const char*>(w2[act[ii]]) + oo;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(size) : "memory");
        }
        // advance (i, off) by a wave
        off += (long long)blockDim.x * chunk;
        done += (long long)blockDim.x * chunk;
        while (i < n) {
            const long long l = (act[i] == shared ? sbytes : bytes) / 16 * 16;
            if (off < l) break;
            off -= l;
            ++i;
        }
    }
#endif
}

// A decode launch's grouping in one CTA (tf_dsn_group_count, _scan and _place of the cnt memset together): the picks
// pick [RS] (ids past E or negative: another node's) counted in shared memory, off [E + 1] their exclusive prefix,
// act [] the experts with a pick in id order and nact [0] their count, then each entry's place: ent [off[e] + i] =
// the entry's index, erow its row (i: the entries of e before it, so in entry order). blockDim 1024, RS <= 4096.
extern "C" __global__ void __launch_bounds__(1024) tf_dsn_dec_group_kernel(const int* __restrict__ pick, int RS, int S,
                                                                           int E, int* __restrict__ off,
                                                                           int* __restrict__ act, int* __restrict__ nact,
                                                                           int* __restrict__ ent, int* __restrict__ erow) {
    extern __shared__ int sh[];  // cnt [E], then off [E + 1]
    int* cnt = sh;
    int* so = sh + E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < RS; i += blockDim.x) {
        const int e = pick[i];
        if (e >= 0 && e < E) atomicAdd(cnt + e, 1);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        // the prefix and the active list, a warp over E in steps of 32 (ballot order: id order)
        const int lane = threadIdx.x;
        int base = 0, nb = 0;
        for (int e0 = 0; e0 < E; e0 += 32) {
            const int e = e0 + lane;
            const int c = e < E ? cnt[e] : 0;
            int x = c;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            if (e < E) so[e] = base + x - c;
            const unsigned m = __ballot_sync(0xffffffffu, c != 0);
            if (c != 0) act[nb + __popc(m & ((1u << lane) - 1))] = e;
            nb += __popc(m);
            base += __shfl_sync(0xffffffffu, x, 31);
        }
        if (lane == 0) {
            so[E] = base;
            nact[0] = nb;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e <= E; e += blockDim.x) off[e] = so[e];
    for (int i = threadIdx.x; i < RS; i += blockDim.x) {
        const int e = pick[i];
        if (e < 0 || e >= E) continue;
        int before = 0;
        for (int j = 0; j < i; j++) before += pick[j] == e;
        const int at = so[e] + before;
        ent[at] = i;
        erow[at] = i / S;
    }
}

// the layouts mx.zig passes (its extern structs assert the same offsets)
static_assert(sizeof(MxArgs) == 128 && offsetof(MxArgs, z) == 96, "MxArgs layout");
static_assert(offsetof(GuArgs, sldw) == 136 && offsetof(GuArgs, owner) == 152 && offsetof(GuArgs, z) == 216 && offsetof(GuArgs, early_exit) == 224 && offsetof(GuArgs, nq) == 232 && sizeof(GuArgs) == 280, "GuArgs layout");
static_assert(offsetof(DnArgs, sldw) == 104 && offsetof(DnArgs, out) == 136 && offsetof(DnArgs, z) == 160 && offsetof(DnArgs, early_exit) == 172 && offsetof(DnArgs, nq) == 176 && sizeof(DnArgs) == 216, "DnArgs layout");
static_assert(sizeof(DecMulti) == 4 * 168 + 24 && offsetof(DecMulti, cta0) == 676, "DecMulti layout");
static_assert(sizeof(DecArgs) == 168 && offsetof(DecArgs, gq) == 152 && offsetof(DecArgs, z) == 96 && offsetof(DecArgs, y) == 112, "DecArgs layout");
static_assert(offsetof(BfArgs, y) == 64 && offsetof(BfArgs, z) == 112 && sizeof(BfArgs) == 120, "BfArgs layout");
