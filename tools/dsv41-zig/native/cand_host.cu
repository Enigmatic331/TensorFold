// Host launcher of zig/kernels/cuda/dsv41_ops.cu's tf_ds_top_cand_rows_kernel for check_cand.py (ctypes, torch
// pointers, stream 0). Not part of the engine.
#include "../../../zig/kernels/cuda/dsv41_ops.cu"
extern "C" int dsn_top_cand(const float* logits, unsigned long long ld, int n, int rows, int count, float* vals,
                            long long* ids, int* flags) {
    tf_ds_top_cand_rows_kernel<<<rows, 1024>>>(logits, ld, n, count, vals, ids, flags);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return int(e);
    return int(cudaDeviceSynchronize());
}
