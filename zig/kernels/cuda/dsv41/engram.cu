// Engram's packed records (each row's FP8 E4M3 values, then its E8M0 scales, one per 32) to bf16 rows on the GPU, value
// for value as engram.zig's decodeRow computes them on the host: the E4M3 value exactly in fp32, times
// 2^(scale - 127), rounded to the nearest even bf16, NaN as 0x7fc0. Built with --ftz=false: subnormals stay.
#include <stdint.h>

// 2^e as fp32: subnormal below 2^-126, infinity from 2^128 (engram.zig pow2).
__device__ __forceinline__ float tf_dse_pow2(int e) {
    if (e >= 128) return __int_as_float(0x7f800000);
    if (e >= -126) return __int_as_float((e + 127) << 23);
    if (e >= -149) return __int_as_float(1 << (e + 149));
    return 0.0f;
}

// An E4M3 (fn) byte as fp32, exactly (engram.zig e4m3): 0x7f and 0xff are NaN.
__device__ __forceinline__ float tf_dse_e4m3(uint32_t b) {
    const uint32_t exp = (b >> 3) & 0xfu;
    const uint32_t man = b & 7u;
    if (exp == 15u && man == 7u) return __int_as_float(0x7fc00000);
    const float mag = exp == 0u ? __fmul_rn((float)man, 0x1p-9f)
                                : __fmul_rn(__fadd_rn(1.0f, __fmul_rn((float)man, 0.125f)), tf_dse_pow2((int)exp - 7));
    return (b & 0x80u) ? -mag : mag;
}

// fp32 to bf16 bits, round to nearest even (engram.zig bf16).
__device__ __forceinline__ uint16_t tf_dse_bf16(float x) {
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u) return 0x7fc0;
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// One thread a value: record (r, lo + c) of rec ([rows][slots] records of dim + dim / 32 bytes) into bf16 row (r, c)
// of out ([rows][k][dim]).
extern "C" __global__ void tf_dse_decode_kernel(const uint8_t* __restrict__ rec, uint16_t* __restrict__ out, int rows,
                                                int slots, int lo, int k, int dim) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long n = (long long)rows * k * dim;
    if (i >= n) return;
    const int v = (int)(i % dim);
    const long long rc = i / dim;
    const int c = (int)(rc % k);
    const long long r = rc / k;
    const long long width = dim + dim / 32;
    const uint8_t* src = rec + (r * slots + lo + c) * width;
    const float x = __fmul_rn(tf_dse_e4m3(src[v]), tf_dse_pow2((int)src[dim + v / 32] - 127));
    out[i] = tf_dse_bf16(x);
}
