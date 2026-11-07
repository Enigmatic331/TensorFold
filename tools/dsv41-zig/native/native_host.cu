// Host launchers of zig/kernels/cuda/dsv41/native.cu for the torch checks in check_native.py (ctypes; raw device
// pointers from torch tensors, the current stream 0). Not part of the engine: the engine launches the same kernels
// through its own driver API.
#include "../../../zig/kernels/cuda/dsv41/native.cu"
#include <stdio.h>

// the decode tensor-core kernels (tf_dsn_dec_*): the row-group template for `rows` rows (an expert's at most), the
// dynamic shared memory of a CTA of `warps` warps
static int dec_rg(int rows) { return rows <= 8 ? 1 : rows <= 16 ? 2 : rows <= 32 ? 4 : 8; }
static void* dec_fn(int kind, int rg) {  // 0 fp8, 1 fp4, 2 bf16, 3 gate / up, 4 down
    void* t[5][4] = {{(void*)tf_dsn_dec_fp8_r1_kernel, (void*)tf_dsn_dec_fp8_r2_kernel, (void*)tf_dsn_dec_fp8_r4_kernel, (void*)tf_dsn_dec_fp8_r8_kernel},
                     {(void*)tf_dsn_dec_fp4_r1_kernel, (void*)tf_dsn_dec_fp4_r2_kernel, (void*)tf_dsn_dec_fp4_r4_kernel, (void*)tf_dsn_dec_fp4_r8_kernel},
                     {(void*)tf_dsn_dec_bf_r1_kernel, (void*)tf_dsn_dec_bf_r2_kernel, (void*)tf_dsn_dec_bf_r4_kernel, (void*)tf_dsn_dec_bf_r8_kernel},
                     {(void*)tf_dsn_dec_gu_r1_kernel, (void*)tf_dsn_dec_gu_r2_kernel, (void*)tf_dsn_dec_gu_r4_kernel, (void*)tf_dsn_dec_gu_r8_kernel},
                     {(void*)tf_dsn_dec_dn_r1_kernel, (void*)tf_dsn_dec_dn_r2_kernel, (void*)tf_dsn_dec_dn_r4_kernel, (void*)tf_dsn_dec_dn_r8_kernel}};
    return t[kind][rg == 1 ? 0 : rg == 2 ? 1 : rg == 4 ? 2 : 3];
}
static cudaError_t dec_launch(int kind, int rg, dim3 grid, int warps, void* args) {
    void* f = dec_fn(kind, rg);
    const int smem = warps * (kind < 3 ? dec::warpBytes<dec::kRbD, dec::kNsD>(1, rg) : kind == 3 ? dec::warpBytes<dec::kRbG, dec::kNsG>(1, rg) : dec::warpBytes<dec::kRbE, dec::kNsE>(1, rg));
    cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return e;
    void* pp[1] = {args};
    return cudaLaunchKernel(f, grid, dim3(32 * warps), pp, smem, 0);
}

extern "C" {

int dsn_dec_quant(const void* x, int x_dtype, long ldx, int M, int K, uint8_t* q, long ldq, float* s, long lds) {
    const long warps = long(M) * (K / 32);
    tf_dsn_dec_quant_kernel<<<(warps * 32 + 255) / 256, 256>>>(x, x_dtype, ldx, M, K, q, ldq, s, lds);
    return int(cudaDeviceSynchronize());
}

// a dense decode linear on the tensor-core kernels: fmt 0 FP8, 1 FP4, 2 bf16 (q: bf16 x then, ldq elements); ks warps
// a tile in a CTA of 8, cs CTAs a tile group (z [cs, M, N], cnt zeroed ints)
int dsn_dec_dense(int fmt, const void* q, const float* s, const int* rows, long ldq, long lds, const uint8_t* w,
                  const uint8_t* ws, long ldw, long ldws, int M, int N, int K, int ks, int cs, float* z, int* cnt,
                  void* y, long ldy, int y_dtype, int accumulate, float scale, int round_first, int multi) {
    DecArgs a{};
    a.q = reinterpret_cast<const uint8_t*>(q); a.s = s; a.rows = rows; a.ldq = ldq; a.lds = lds; a.w = w; a.ws = ws;
    a.ldw = ldw; a.ldws = ldws; a.fmt = fmt; a.M = M; a.N = N; a.K = K; a.ks = ks; a.cs = cs; a.z = z; a.cnt = cnt;
    a.y = y; a.ldy = ldy; a.y_dtype = y_dtype; a.accumulate = accumulate; a.scale = scale; a.round_first = round_first;
    if (M > 64) return -1;
    const int tpc = 8 / ks;
    DecMulti mm{};
    const int ctas = (N / 16 + tpc - 1) / tpc;
    mm.p[0] = a;
    mm.n = 1;
    mm.cta0[1] = ctas;
    if (multi > 1) {
        // the same linear `multi` times in one launch (the parts' outputs one after another in y, their counters apart)
        for (int p = 1; p < multi; p++) {
            mm.p[p] = a;
            mm.p[p].y = reinterpret_cast<char*>(y) + size_t(p) * M * ldy * (y_dtype == 2 ? 4 : 2);
            mm.p[p].cnt = cnt + p * ctas;
            mm.p[p].z = z + size_t(p) * cs * M * N;
            mm.cta0[p + 1] = (p + 1) * ctas;
        }
        mm.n = multi;
    }
    const cudaError_t le = dec_launch(fmt, dec_rg(M), dim3(mm.cta0[mm.n], 1, cs), 8, &mm);
    if (le != cudaSuccess) return int(le);
    return int(cudaDeviceSynchronize());
}

// one dense decode launch with every DecArgs field the engine sets (the block-diagonal groups gn / gq / gs of wo_a's FP8
// twin, accumulate), at the pointers given: check_native.py's G0.1 slab checks call it per 64-row slab at the slab's
// row offsets, as mx.decSlab does
int dsn_dec_dense_g(int fmt, const void* q, const float* s, const int* rows, long ldq, long lds, const uint8_t* w,
                    const uint8_t* ws, long ldw, long ldws, int M, int N, int K, int ks, float* z, int* cnt, void* y,
                    long ldy, int y_dtype, int accumulate, int gn, long gq, long gs) {
    DecArgs a{};
    a.q = reinterpret_cast<const uint8_t*>(q); a.s = s; a.rows = rows; a.ldq = ldq; a.lds = lds; a.w = w; a.ws = ws;
    a.ldw = ldw; a.ldws = ldws; a.fmt = fmt; a.M = M; a.N = N; a.K = K; a.ks = ks; a.cs = 1; a.z = z; a.cnt = cnt;
    a.y = y; a.ldy = ldy; a.y_dtype = y_dtype; a.accumulate = accumulate; a.scale = 1.f; a.round_first = 0;
    a.gn = gn; a.gq = gq; a.gs = gs;
    if (M > 64) return -1;
    const int tpc = 8 / ks;
    DecMulti mm{};
    mm.p[0] = a;
    mm.n = 1;
    mm.cta0[1] = (N / 16 + tpc - 1) / tpc;
    const cudaError_t le = dec_launch(fmt, dec_rg(M), dim3(mm.cta0[1], 1, 1), 8, &mm);
    if (le != cudaSuccess) return int(le);
    return int(cudaDeviceSynchronize());
}

int dsn_act_quant(const void* x, int x_dtype, long ldx, int M, int K, uint8_t* q, long ldq, float* s, long lds,
                  __nv_bfloat16* fake, long ldf) {
    const long warps = long(M) * (K / 32);
    tf_dsn_act_quant_kernel<<<(warps * 32 + 255) / 256, 256>>>(x, x_dtype, ldx, M, K, q, ldq, s, lds, fake, ldf);
    return int(cudaDeviceSynchronize());
}

int dsn_mx_gemm(const uint8_t* q, const float* s, const int* rows, long ldq, long lds, const uint8_t* w,
                const uint8_t* ws, long ldw, long ldws, int fmt, int M, int N, int K, int splits, int mt, float* z,
                void* y, long ldy, int y_dtype, int accumulate) {
    MxArgs a{};
    a.q = q; a.s = s; a.rows = rows; a.ldq = ldq; a.lds = lds; a.w = w; a.ws = ws; a.ldw = ldw; a.ldws = ldws;
    a.fmt = fmt; a.M = M; a.N = N; a.K = K; a.splits = splits; a.kblocks = (K / 32 + splits - 1) / splits;
    a.z = z; a.y = y; a.ldy = ldy; a.y_dtype = y_dtype; a.accumulate = accumulate;
    if (mt == -2) {  // the decode swap-AB MMA
        tf_dsn_mx_sab_kernel<<<(N + 15) / 16, 256>>>(a);
        return int(cudaDeviceSynchronize());
    }
    if (mt < 0) {  // the decode GEMV (splits unused)
        tf_dsn_mx_gemv_kernel<<<(N + 7) / 8, 256>>>(a);
        return int(cudaDeviceSynchronize());
    }
    dim3 grid((N / 32 + 3) / 4, splits, (M + mt * 16 - 1) / (mt * 16));
    switch (mt) {
        case 1: tf_dsn_mx_gemm1_kernel<<<grid, 128>>>(a); break;
        case 2: tf_dsn_mx_gemm2_kernel<<<grid, 128>>>(a); break;
        case 4: tf_dsn_mx_gemm4_kernel<<<grid, 128>>>(a); break;
        default: tf_dsn_mx_gemm8_kernel<<<grid, 128>>>(a); break;
    }
    if (splits > 1) tf_dsn_reduce_kernel<<<(long(M) * N + 255) / 256, 256>>>(z, splits, M, N, y, ldy, y_dtype, accumulate);
    return int(cudaDeviceSynchronize());
}

// Track C: the decode expert kernels' early exit (GuArgs / DnArgs early_exit) for dsn_experts' launches
static int g_expert_exit = 0;
extern "C" void dsn_set_expert_exit(int v) { g_expert_exit = v; }

// the grouped experts of one layer: x [R, D] bf16 -> y [R, N] fp32 (routed sum), every expert's tensors in tables
int dsn_experts(const __nv_bfloat16* x, int R, int D, const int* pick, const float* wts, int S, int E,
                const uint64_t* w1, const uint64_t* w3, const uint64_t* s1, const uint64_t* s3, long ldw_gu,
                long ldws_gu, const uint64_t* w2, const uint64_t* s2, long ldw_d, long ldws_d, int fmt, int I,
                const int* cols, int n_tiles, int N, float limit,
                // scratch
                uint8_t* q, float* s, int* cnt, int* off, int* fill, int* act, int* nact, int* ent, int* erow,
                __nv_bfloat16* h, uint8_t* hq, float* hs, float* out, float* y, int mt,
                // split K (decode launches): gate / up's and down's splits (1: none) and their partials z
                int gu_splits, int dn_splits, float* z) {
    const int RS = R * S;
    dsn_act_quant(x, 1, D, R, D, q, D, s, D / 32, nullptr, 0);
    cudaMemset(cnt, 0, sizeof(int) * E);
    cudaMemset(h, 0, sizeof(__nv_bfloat16) * size_t(RS) * I);
    if (gu_splits <= -10) {  // the decode path's one-CTA grouping
        tf_dsn_dec_group_kernel<<<1, 1024, (2 * E + 1) * 4>>>(pick, RS, S, E, off, act, nact, ent, erow);
    } else {
        tf_dsn_group_count_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, E, cnt);
        tf_dsn_group_scan_kernel<<<1, 1>>>(cnt, E, off, fill, act, nact);
        tf_dsn_group_place_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, S, E, off, fill, ent, erow);
    }
    GuArgs g{};
    g.q = q; g.s = s; g.ldq = D; g.lds = D / 32; g.ent = ent; g.erow = erow; g.off = off; g.act = act; g.nact = nact;
    g.w1 = w1; g.w3 = w3; g.s1 = s1; g.s3 = s3; g.ldw = ldw_gu; g.ldws = ldws_gu; g.fmt = fmt; g.D = D;
    g.shared = -1; g.owner = nullptr; g.n_tiles = n_tiles; g.rest_tiles = 0; g.wts = wts; g.limit = limit; g.h = h; g.ldh = I;
    (void)cols;
    g.P = RS; g.splits = gu_splits > 1 ? gu_splits : 1; g.kblocks = (D / 32 + g.splits - 1) / g.splits; g.z = z;
    g.early_exit = g_expert_exit;
    dim3 gg((n_tiles + 3) / 4, E < RS ? E : RS, g.splits);
    if (gu_splits <= -10) {  // the decode tensor-core kernel, -10 - gu_splits warps a tile and matrix
        g.splits = -10 - gu_splits;
        const int tpc = 8 / (2 * g.splits);
        dec_launch(3, dec_rg(R), dim3((n_tiles * 2 + tpc - 1) / tpc, E < RS ? E : RS), 8, &g);
    } else if (gu_splits == -2) tf_dsn_gu_sab_kernel<<<dim3((n_tiles * 32 + 15) / 16, E < RS ? E : RS), 256>>>(g);  // swap AB
    else if (gu_splits < 0) tf_dsn_gu_gemv_kernel<<<dim3((n_tiles * 32 + 7) / 8, E < RS ? E : RS), 256>>>(g);  // decode GEMV
    else if (mt == 1) tf_dsn_gu1_kernel<<<gg, 128>>>(g); else if (mt == 2) tf_dsn_gu2_kernel<<<gg, 128>>>(g); else tf_dsn_gu4_kernel<<<gg, 128>>>(g);
    if (gu_splits > 1) tf_dsn_gu_finish_kernel<<<dim3((n_tiles * 32 + 127) / 128, E < RS ? E : RS), 128>>>(g);
    dsn_act_quant(h, 1, I, RS, I, hq, I, hs, I / 32, nullptr, 0);
    DnArgs d{};
    d.q = hq; d.s = hs; d.ldq = I; d.lds = I / 32; d.ent = ent; d.off = off; d.act = act; d.nact = nact;
    d.w2 = w2; d.s2 = s2; d.ldw = ldw_d; d.ldws = ldws_d; d.fmt = fmt; d.shared = -1; d.I = I; d.N = N; d.round_bf16 = 1;
    d.out = out; d.ldo = N;
    d.splits = dn_splits > 1 ? dn_splits : 1; d.kblocks = (I / 32 + d.splits - 1) / d.splits; d.z = z; d.P = RS;
    d.early_exit = g_expert_exit;
    dim3 gd((N / 32 + 3) / 4, E < RS ? E : RS, d.splits);
    if (dn_splits <= -10) {
        d.splits = -10 - dn_splits;
        const int tpc = 8 / d.splits;
        dec_launch(4, dec_rg(R), dim3((N / 16 + tpc - 1) / tpc, E < RS ? E : RS), 8, &d);
    } else if (dn_splits == -2) tf_dsn_dn_sab_kernel<<<dim3((N + 15) / 16, E < RS ? E : RS), 256>>>(d);  // swap AB
    else if (dn_splits < 0) tf_dsn_dn_gemv_kernel<<<dim3((N + 7) / 8, E < RS ? E : RS), 256>>>(d);  // decode GEMV
    else if (mt == 1) tf_dsn_dn1_kernel<<<gd, 128>>>(d); else if (mt == 2) tf_dsn_dn2_kernel<<<gd, 128>>>(d); else tf_dsn_dn4_kernel<<<gd, 128>>>(d);
    if (dn_splits > 1) tf_dsn_dn_finish_kernel<<<dim3((N + 127) / 128, E < RS ? E : RS), 128>>>(d);
    tf_dsn_combine_kernel<<<(long(R) * N + 255) / 256, 256>>>(out, N, pick, R, S, E, N, y, N, 0);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "dsn_experts: %s (last launch %s)\n", cudaGetErrorString(err), cudaGetErrorString(cudaGetLastError()));
    return int(err);
}

int dsn_bf16_gemm(const __nv_bfloat16* x, long ldx, long gx, const __nv_bfloat16* w, long ldw, long gw, int M, int N,
                  int K, int groups, void* y, long ldy, long gy, int y_dtype, float scale, int round_first, int splits,
                  float* z) {
    BfArgs a{};
    a.x = x; a.ldx = ldx; a.gx = gx; a.w = w; a.ldw = ldw; a.gw = gw; a.M = M; a.N = N; a.K = K; a.y = y; a.ldy = ldy;
    a.gy = gy; a.y_dtype = y_dtype; a.scale = scale; a.round_first = round_first;
    a.splits = splits; a.ksteps = (K / 16 + splits - 1) / splits; a.z = z;
    dim3 grid((N / 32 + 3) / 4, (M + 15) / 16 < 64 ? (M + 15) / 16 : 64, splits > 1 ? splits : groups);
    tf_dsn_bf16_gemm_kernel<<<grid, 128>>>(a);
    if (splits > 1) tf_dsn_bf_reduce_kernel<<<(long(M) * N + 255) / 256, 256>>>(a);
    return int(cudaDeviceSynchronize());
}
}

// ---------------------------------------------------------------------------------------------------------------------
// The prefill tile GEMMs (native.cu tf_dsn_pf_*), launched as mx.zig launches them.
extern "C" {

int dsn_pf_gemm_launch(MxArgs a, cudaStream_t st) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(tf_dsn_pf_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kGemmSmem);
        cudaFuncSetAttribute(tf_dsn_pf_gemm_fold_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kGemmSmem);
        init = true;
    }
    using F = pf::GemmFoldCfg;
    using G = pf::GemmCfg;
    if (a.splits > 1) tf_dsn_pf_gemm_fold_kernel<<<dim3((a.M + F::BM - 1) / F::BM, a.N / F::BN), 256, pf::kGemmSmem, st>>>(a);
    else tf_dsn_pf_gemm_kernel<<<dim3((a.M + G::BM - 1) / G::BM, a.N / G::BN), 256, pf::kGemmSmem, st>>>(a);
    return int(cudaGetLastError());
}

static int2* pf_tiles = nullptr;
static int* pf_ntiles = nullptr;

static void pf_tiles_init() {
    if (pf_tiles) return;
    cudaMalloc(&pf_tiles, sizeof(int2) * (1 << 20));
    cudaMalloc(&pf_ntiles, sizeof(int));
    cudaFuncSetAttribute(tf_dsn_pf_gu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kGuSmem);
    cudaFuncSetAttribute(tf_dsn_pf_dn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kDnSmem);
}

// gate / up (the grouping in a's tables done): the tile list, then the tiles
int dsn_pf_gu_launch(GuArgs a, int E, cudaStream_t st) {
    pf_tiles_init();
    tf_dsn_pf_tiles_kernel<<<1, 32, 0, st>>>(a.off, a.act, a.nact, pf_tiles, pf_ntiles);
    dim3 grid(a.n_tiles * 32 / pf::GuCfg4::BN, a.P / 64 + E);
    tf_dsn_pf_gu_kernel<<<grid, 256, pf::kGuSmem, st>>>(a, pf_tiles, pf_ntiles);
    return int(cudaGetLastError());
}

int dsn_pf_dn_launch(DnArgs a, int E, cudaStream_t st) {
    pf_tiles_init();
    tf_dsn_pf_tiles_kernel<<<1, 32, 0, st>>>(a.off, a.act, a.nact, pf_tiles, pf_ntiles);
    dim3 grid(a.N / pf::DnCfg4::BN, a.P / 64 + E);
    tf_dsn_pf_dn_kernel<<<grid, 256, pf::kDnSmem, st>>>(a, pf_tiles, pf_ntiles);
    return int(cudaGetLastError());
}

}

extern "C" int dsn_pf_bf16_launch(BfArgs a, int groups, cudaStream_t st) {
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(tf_dsn_pf_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kBfSmem);
        init = true;
    }
    tf_dsn_pf_bf16_kernel<<<dim3((a.M + 127) / 128, a.N / 128, groups), 256, pf::kBfSmem, st>>>(a);
    return int(cudaGetLastError());
}

// The grouped experts of one layer with a shared FP8 expert (id `shared`, its strides apart), through the tile kernels
// (pf 0, mt row tiles) or the prefill tile GEMMs (pf 1): x [R, D] bf16 -> y [R, N] fp32, as the engine runs a prompt
// chunk's (no K split; owner null).
extern "C" int dsn_experts_pf(const __nv_bfloat16* x, int R, int D, const int* pick, const float* wts, int S, int E,
                              const uint64_t* w1, const uint64_t* w3, const uint64_t* s1, const uint64_t* s3, long ldw_gu,
                              long ldws_gu, const uint64_t* w2, const uint64_t* s2, long ldw_d, long ldws_d, int I,
                              int N, float limit, int shared, long sldw_gu, long sldws_gu, long sldw_d, long sldws_d,
                              uint8_t* q, float* s, int* cnt, int* off, int* fill, int* act, int* nact, int* ent, int* erow,
                              __nv_bfloat16* h, uint8_t* hq, float* hs, float* out, float* y, int mt, int pf) {
    const int RS = R * S;
    dsn_act_quant(x, 1, D, R, D, q, D, s, D / 32, nullptr, 0);
    cudaMemset(cnt, 0, sizeof(int) * E);
    cudaMemset(h, 0, sizeof(__nv_bfloat16) * size_t(RS) * I);
    tf_dsn_group_count_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, E, cnt);
    tf_dsn_group_scan_kernel<<<1, 1>>>(cnt, E, off, fill, act, nact);
    tf_dsn_group_place_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, S, E, off, fill, ent, erow);
    GuArgs g{};
    g.q = q; g.s = s; g.ldq = D; g.lds = D / 32; g.ent = ent; g.erow = erow; g.off = off; g.act = act; g.nact = nact;
    g.w1 = w1; g.w3 = w3; g.s1 = s1; g.s3 = s3; g.ldw = ldw_gu; g.ldws = ldws_gu; g.fmt = 1; g.D = D;
    g.shared = shared; g.sldw = sldw_gu; g.sldws = sldws_gu; g.owner = nullptr; g.n_tiles = I / 32; g.rest_tiles = 0;
    g.wts = wts; g.limit = limit; g.h = h; g.ldh = I; g.P = RS; g.splits = 1; g.kblocks = D / 32; g.z = nullptr;
    const int ny = E < RS ? E : RS;
    if (pf) dsn_pf_gu_launch(g, E, 0);
    else if (mt == 1) tf_dsn_gu1_kernel<<<dim3((I / 32 + 3) / 4, ny), 128>>>(g);
    else if (mt == 2) tf_dsn_gu2_kernel<<<dim3((I / 32 + 3) / 4, ny), 128>>>(g);
    else tf_dsn_gu4_kernel<<<dim3((I / 32 + 3) / 4, ny), 128>>>(g);
    dsn_act_quant(h, 1, I, RS, I, hq, I, hs, I / 32, nullptr, 0);
    DnArgs d{};
    d.q = hq; d.s = hs; d.ldq = I; d.lds = I / 32; d.ent = ent; d.off = off; d.act = act; d.nact = nact;
    d.w2 = w2; d.s2 = s2; d.ldw = ldw_d; d.ldws = ldws_d; d.fmt = 1; d.shared = shared; d.sldw = sldw_d; d.sldws = sldws_d;
    d.I = I; d.N = N; d.round_bf16 = 1; d.out = out; d.ldo = N; d.splits = 1; d.kblocks = I / 32; d.z = nullptr; d.P = RS;
    if (pf) dsn_pf_dn_launch(d, E, 0);
    else if (mt == 1) tf_dsn_dn1_kernel<<<dim3((N / 32 + 3) / 4, ny), 128>>>(d);
    else if (mt == 2) tf_dsn_dn2_kernel<<<dim3((N / 32 + 3) / 4, ny), 128>>>(d);
    else tf_dsn_dn4_kernel<<<dim3((N / 32 + 3) / 4, ny), 128>>>(d);
    tf_dsn_combine_kernel<<<(long(R) * N + 255) / 256, 256>>>(out, N, pick, R, S, E, N, y, N, 0);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "dsn_experts_pf: %s\n", cudaGetErrorString(err));
    return int(err);
}

// mx.zig gemm()'s prefill path: the prefill tile GEMM (synchronized)
extern "C" int dsn_pf_gemm(const uint8_t* q, const float* s, long ldq, long lds, const uint8_t* w, const uint8_t* ws,
                           long ldw, long ldws, int M, int N, int K, int splits, void* y, long ldy, int y_dtype,
                           int accumulate) {
    MxArgs a{};
    a.q = q; a.s = s; a.ldq = ldq; a.lds = lds; a.w = w; a.ws = ws; a.ldw = ldw; a.ldws = ldws; a.fmt = 0;
    a.M = M; a.N = N; a.K = K; a.splits = splits; a.kblocks = (K / 32 + splits - 1) / splits; a.y = y; a.ldy = ldy;
    a.y_dtype = y_dtype; a.accumulate = accumulate;
    dsn_pf_gemm_launch(a, 0);
    return int(cudaDeviceSynchronize());
}

// the bf16 GEMM's prefill path (splits 1)
extern "C" int dsn_pf_bf16(const __nv_bfloat16* x, long ldx, long gx, const __nv_bfloat16* w, long ldw, long gw, int M,
                           int N, int K, int groups, void* y, long ldy, long gy, int y_dtype, float scale, int round_first) {
    BfArgs a{};
    a.x = x; a.ldx = ldx; a.gx = gx; a.w = w; a.ldw = ldw; a.gw = gw; a.M = M; a.N = N; a.K = K; a.y = y; a.ldy = ldy;
    a.gy = gy; a.y_dtype = y_dtype; a.scale = scale; a.round_first = round_first; a.splits = 1; a.ksteps = 0; a.z = nullptr;
    dsn_pf_bf16_launch(a, groups, 0);
    return int(cudaDeviceSynchronize());
}

// NVFP4's grouped experts for check_nvfp4.py (no shared expert): x [R, D] bf16 -> y [R, N] fp32 (routed sum) and the
// intermediate h [R * S, I] bf16, on the decode tensor-core kernels (path 0) or the prefill tiles (path 1).
extern "C" int dsn_nv_quant(const __nv_bfloat16* x, long ldx, const int* pick, int S, int by_entry, int P, int K, const float* g,
                            int routed, uint8_t* nq, uint8_t* ns) {
    tf_dsn_nv_quant_kernel<<<(long(P) * (K / 16) + 127) / 128, 128>>>(x, ldx, pick, S, by_entry, P, K, g, routed, nq, K / 2, ns, K / 16);
    return int(cudaDeviceSynchronize());
}

extern "C" int dsn_nv_experts(const __nv_bfloat16* x, int R, int D, const int* pick, const float* wts, int S, int E,
                              const uint64_t* w1, const uint64_t* w3, const uint64_t* s1, const uint64_t* s3,
                              const uint64_t* w2, const uint64_t* s2, const float* g13, const float* g2, const float* a1,
                              const float* a3, const float* a2, int I, int N, float limit, int* cnt, int* off, int* fill,
                              int* act, int* nact, int* ent, int* erow, uint8_t* nq, uint8_t* ns, __nv_bfloat16* h,
                              uint8_t* hnq, uint8_t* hns, float* out, float* y, int path) {
    const int RS = R * S, ny = E < RS ? E : RS;
    cudaMemset(cnt, 0, sizeof(int) * E);
    cudaMemset(h, 0, sizeof(__nv_bfloat16) * size_t(RS) * I);
    if (path == 0) {
        tf_dsn_dec_group_kernel<<<1, 1024, (2 * E + 1) * 4>>>(pick, RS, S, E, off, act, nact, ent, erow);
    } else {
        tf_dsn_group_count_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, E, cnt);
        tf_dsn_group_scan_kernel<<<1, 1>>>(cnt, E, off, fill, act, nact);
        tf_dsn_group_place_kernel<<<(RS + 255) / 256, 256>>>(pick, RS, S, E, off, fill, ent, erow);
    }
    dsn_nv_quant(x, D, pick, S, 0, RS, D, g13, E, nq, ns);
    GuArgs g{};
    g.ent = ent; g.erow = erow; g.off = off; g.act = act; g.nact = nact; g.w1 = w1; g.w3 = w3; g.s1 = s1; g.s3 = s3;
    g.ldw = D / 2; g.ldws = D / 32; g.fmt = 2; g.D = D; g.shared = -1; g.n_tiles = I / 32; g.wts = wts; g.limit = limit;
    g.h = h; g.ldh = I; g.P = RS; g.nq = nq; g.ns = ns; g.ldnq = D / 2; g.ldns = D / 16; g.a1 = a1; g.a3 = a3;
    pf_tiles_init();
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(tf_dsn_pf_gu_nv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kGuSmem);
        cudaFuncSetAttribute(tf_dsn_pf_dn_nv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::kDnSmem);
        init = true;
    }
    if (path == 0) {
        g.splits = 4;
        const int tpc = 8 / (2 * g.splits);
        dec_launch(3, dec_rg(R), dim3((g.n_tiles * 2 + tpc - 1) / tpc, ny), 8, &g);
    } else {
        tf_dsn_pf_tiles_kernel<<<1, 32>>>(off, act, nact, pf_tiles, pf_ntiles);
        tf_dsn_pf_gu_nv_kernel<<<dim3(g.n_tiles * 32 / 128, RS / 64 + E), 256, pf::kGuSmem>>>(g, pf_tiles, pf_ntiles);
    }
    dsn_nv_quant(h, I, pick, S, 1, RS, I, g2, E, hnq, hns);
    DnArgs d{};
    d.ent = ent; d.off = off; d.act = act; d.nact = nact; d.w2 = w2; d.s2 = s2; d.ldw = I / 2; d.ldws = I / 32; d.fmt = 2;
    d.shared = -1; d.I = I; d.N = N; d.round_bf16 = 1; d.out = out; d.ldo = N; d.P = RS; d.nq = hnq; d.ns = hns;
    d.ldnq = I / 2; d.ldns = I / 16; d.a2 = a2;
    if (path == 0) {
        d.splits = 2;
        const int tpc = 8 / d.splits;
        dec_launch(4, dec_rg(R), dim3((N / 16 + tpc - 1) / tpc, ny), 8, &d);
    } else {
        tf_dsn_pf_dn_nv_kernel<<<dim3(N / 128, RS / 64 + E), 256, pf::kDnSmem>>>(d, pf_tiles, pf_ntiles);
    }
    tf_dsn_combine_kernel<<<(long(R) * N + 255) / 256, 256>>>(out, N, pick, R, S, E, N, y, N, 0);
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "dsn_nv_experts: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorString(cudaGetLastError()));
    return int(err);
}
