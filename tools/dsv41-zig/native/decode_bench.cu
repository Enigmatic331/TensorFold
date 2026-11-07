// Decode microbenchmarks of zig/kernels/cuda/dsv41/native.cu on one GB10 (not part of the engine): the decode
// launches of a 2D-TP4 node's layer shapes, each replayed from a CUDA graph over `layers` weight copies (cold L2, as a
// round reads each layer's weights once), timed per launch and as GB/s of the weight bytes read.
//   nvcc -arch=sm_121a -O3 -std=c++17 -o decode_bench decode_bench.cu && ./decode_bench [filter]
#include "../../../zig/kernels/cuda/dsv41/native.cu"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <functional>
#include <string>
#include <array>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static uint32_t rng = 12345;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

// device buffer of n bytes: fp4 / fp8 / scale / activation fills
enum Fill { F_FP4, F_FP8, F_SCALE, F_ONE, F_ZERO };
__global__ void fill_kernel(uint8_t* p, size_t n, int f, uint32_t seed) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t r = uint32_t(i) * 2654435761u ^ seed;
        r ^= r >> 15; r *= 2246822519u; r ^= r >> 13; r *= 3266489917u; r ^= r >> 16;
        if (f == F_ONE) {
            if (i % 4 == 0 && i + 3 < n) *reinterpret_cast<float*>(p + i) = 1.0f / float(1 << (r % 4));
            continue;
        }
        p[i] = f == F_FP4 ? uint8_t(r) : f == F_FP8 ? (uint8_t(r) & 0xBF) : f == F_SCALE ? uint8_t(124 + r % 7) : 0;
    }
}
static void* dbuf(size_t n, Fill f) {
    void* p;
    CK(cudaMalloc(&p, n));
    fill_kernel<<<512, 256>>>((uint8_t*)p, n, f, rnd());
    CK(cudaGetLastError());
    return p;
}

static cudaStream_t st;

// time `body(l)` for l in 0..L-1 as one graph, replayed; returns us per launch-set (per layer)
static double timeit(int L, const std::function<void(int)>& body, int reps = 20) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    for (int l = 0; l < L; l++) body(l);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, st);
    for (int r = 0; r < reps; r++) CK(cudaGraphLaunch(ge, st));
    cudaEventRecord(b, st);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return ms * 1000.0 / reps / L;
}

static const char* filt = nullptr;
static bool want(const char* name) { return !filt || strstr(name, filt); }
static void report(const char* name, double us, double bytes) {
    printf("%-44s %9.2f us  %7.1f GB/s\n", name, us, bytes / us / 1e3);
    fflush(stdout);
}

// ---------------------------------------------------------------- raw read bandwidth
__global__ void bw_kernel(const uint4* p, size_t n, uint32_t* sink) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) sink[0] = acc;
}


// ---------------------------------------------------------------- the dec kernels
static int rgOf(int R) { return R <= 8 ? 1 : R <= 16 ? 2 : R <= 32 ? 4 : 8; }
#define DEC_PICK(kind, rg) (rg == 1 ? (void*)tf_dsn_dec_##kind##_r1_kernel : rg == 2 ? (void*)tf_dsn_dec_##kind##_r2_kernel : rg == 4 ? (void*)tf_dsn_dec_##kind##_r4_kernel : (void*)tf_dsn_dec_##kind##_r8_kernel)
static void* decFn(const char* kind, int rg) {
    if (!strcmp(kind, "fp8")) return DEC_PICK(fp8, rg);
    if (!strcmp(kind, "fp4")) return DEC_PICK(fp4, rg);
    if (!strcmp(kind, "gu")) return DEC_PICK(gu, rg);
    if (!strcmp(kind, "dn")) return DEC_PICK(dn, rg);
    return DEC_PICK(bf, rg);
}
static int warps_cta = getenv("WARPS") ? atoi(getenv("WARPS")) : 8;
static void decLaunch(const char* kind, int rg, int nm, dim3 grid, void* args) {
    void* f = decFn(kind, rg);
    const bool ex = !strcmp(kind, "gu") || !strcmp(kind, "dn");
    const int smem = warps_cta * (!strcmp(kind, "gu") ? dec::warpBytes<dec::kRbG, dec::kNsG>(nm, rg) : ex ? dec::warpBytes<dec::kRbE, dec::kNsE>(nm, rg) : dec::warpBytes<dec::kRbD, dec::kNsD>(nm, rg));
    CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    DecMulti mm{};
    if (!ex) { mm.p[0] = *reinterpret_cast<DecArgs*>(args); mm.n = 1; args = &mm; }
    void* pp[1] = {args};
    CK(cudaLaunchKernel(f, grid, dim3(32 * warps_cta), pp, smem, st));
}

// ---------------------------------------------------------------- dense
struct Dense {
    const char* name;
    int fmt;  // 0 fp8, 1 fp4, 2 bf16
    int K, N;
};

int main(int argc, char** argv) {
    if (argc > 1) filt = argv[1];
    CK(cudaStreamCreate(&st));
    const int L = 24;
    std::vector<int> rows_list = {1, 3, 6};
    if (getenv("ROWS")) { rows_list.clear(); for (char* p = getenv("ROWS"); *p;) { rows_list.push_back(strtol(p, &p, 10)); if (*p) p++; } }
    uint32_t* sink;
    CK(cudaMalloc(&sink, 4));

    {
        // the control: a 512 MiB streaming read (every run, to compare runs taken at different times)
        size_t n = size_t(1) << 29;
        void* p = dbuf(n, F_ZERO);
        double us = timeit(1, [&](int) { bw_kernel<<<48 * 8, 256, 0, st>>>((const uint4*)p, n / 16, sink); }, 10);
        report("control bw 512MiB", us, double(n));
        cudaFree(p);
    }
    if (want("launch")) {
        // a graph's kernel-to-kernel cost: 200 empty launches, and 200 small ones (48 CTAs)
        double us = timeit(200, [&](int) { bw_kernel<<<1, 32, 0, st>>>((const uint4*)sink, 0, sink); }, 20);
        report("launch empty (per kernel)", us, 1);
        us = timeit(200, [&](int) { bw_kernel<<<48, 256, 0, st>>>((const uint4*)sink, 0, sink); }, 20);
        report("launch 48x256 empty (per kernel)", us, 1);
    }
    if (want("bw")) {
        size_t n = size_t(1) << 30;
        void* p = dbuf(n, F_ZERO);
        for (int blocks : {48 * 4, 48 * 8, 48 * 16}) {
            double us = timeit(1, [&](int) { bw_kernel<<<blocks, 256, 0, st>>>((const uint4*)p, n / 16, sink); }, 5);
            char nm[64];
            snprintf(nm, 64, "bw 1GiB blocks=%d", blocks);
            report(nm, us, double(n));
        }
        cudaFree(p);
    }

    // activations: rows up to 64, K up to 8192
    const int maxR = 64, maxK = 8192;
    uint8_t* q = (uint8_t*)dbuf(size_t(maxR) * maxK, F_FP8);
    float* s = (float*)dbuf(size_t(maxR) * maxK / 32 * 4, F_ONE);
    float* y = (float*)dbuf(size_t(maxR) * 40000 * 4, F_ZERO);
    float* z = (float*)dbuf(size_t(64) << 20, F_ZERO);
    __nv_bfloat16* xb = (__nv_bfloat16*)dbuf(size_t(maxR) * maxK * 2, F_ZERO);
    int* cnt0 = (int*)dbuf(4 * 65536, F_ZERO);
    std::vector<std::array<int, 2>> kcs = {{8, 1}, {4, 2}, {8, 2}, {8, 4}};
    if (getenv("KCS")) { kcs.clear(); for (char* p = getenv("KCS"); *p;) { int a2 = strtol(p, &p, 10); p++; int b2 = strtol(p, &p, 10); kcs.push_back({a2, b2}); if (*p) p++; } }

    const Dense dense[] = {
        {"wq_a fp8 5120->1280", 0, 5120, 1280},     {"wkv fp8 5120->512", 0, 5120, 512},
        {"wq_b fp8 1280->8192", 0, 1280, 8192},     {"idx_wq_b fp8 1280->4096", 0, 1280, 4096},
        {"wo_b fp8 4096->2560", 0, 4096, 2560},     {"wo_a(fp8) 4096->1024", 0, 4096, 1024},
        {"shared gu fp8 5120->640", 0, 5120, 640},  {"shared dn fp8 1152->2560", 0, 1152, 2560},
        {"expert gu fp4 5120->640", 1, 5120, 640},  {"expert dn fp4 1152->2560", 1, 1152, 2560},
        {"wo_a bf16 4096->1024", 2, 4096, 1024},    {"head bf16 5120->32320", 2, 5120, 32320},
        {"router bf16 5120->384", 2, 5120, 384},    {"comp_wkv bf16 5120->512", 2, 5120, 512},
    };
    for (const Dense& d : dense) {
        bool any = false;
        for (const char* v : {"gemv", "sab", "bf16", "dec"}) { char nm[96]; snprintf(nm, 96, "%s %s", v, d.name); any |= want(nm); }
        if (!any) continue;
        const size_t wbytes = d.fmt == 1 ? size_t(d.N) * d.K / 2 : d.fmt == 0 ? size_t(d.N) * d.K : size_t(d.N) * d.K * 2;
        const size_t sbytes = d.fmt == 1 ? size_t(d.N) * d.K / 32 : d.fmt == 0 ? size_t(d.N / 32) * d.K / 32 : 0;
        const int Ld = std::min<size_t>(L, std::max<size_t>(2, (size_t(400) << 20) / wbytes));
        std::vector<void*> ws(Ld), ss(Ld);
        for (int l = 0; l < Ld; l++) {
            ws[l] = dbuf(wbytes, d.fmt == 1 ? F_FP4 : d.fmt == 0 ? F_FP8 : F_ZERO);
            ss[l] = sbytes ? dbuf(sbytes, F_SCALE) : nullptr;
        }
        for (int R : rows_list) {
            if (d.fmt == 2) {
                char nm[96];
                snprintf(nm, 96, "bf16 %s R=%d", d.name, R);
                if (want(nm)) {
                // the engine's decode bf16 GEMM: split K as bfSplits
                int steps = d.K / 16, ctas = (d.N / 32 + 3) / 4, sp = 1;
                while (ctas * sp < 96 && sp < 16 && steps / (sp * 2) >= 8) sp *= 2;
                double us = timeit(Ld, [&](int l) {
                    BfArgs a{};
                    a.x = xb; a.ldx = d.K; a.w = (const __nv_bfloat16*)ws[l]; a.ldw = d.K; a.M = R; a.N = d.N; a.K = d.K;
                    a.y = y; a.ldy = d.N; a.y_dtype = 2; a.scale = 1.f; a.splits = sp; a.ksteps = (steps + sp - 1) / sp; a.z = z;
                    dim3 grid((d.N / 32 + 3) / 4, (R + 15) / 16, sp);
                    tf_dsn_bf16_gemm_kernel<<<grid, 128, 0, st>>>(a);
                    if (sp > 1) tf_dsn_bf_reduce_kernel<<<(R * d.N + 255) / 256, 256, 0, st>>>(a);
                });
                report(nm, us, double(wbytes));
                }
                for (auto kc : kcs) {
                    const int ks = kc[0], cs = kc[1];
                    snprintf(nm, 96, "dec%dx%d %s R=%d", ks, cs, d.name, R);
                    if (!want(nm)) continue;
                    const int ntiles = d.N / 16, tpc = warps_cta / ks;
                    double us2 = timeit(Ld, [&](int l) {
                        DecArgs a{};
                        a.q = (const uint8_t*)xb; a.ldq = d.K; a.w = (const uint8_t*)ws[l]; a.ldw = d.K * 2; a.M = R; a.N = d.N; a.K = d.K;
                        a.y = y; a.ldy = d.N; a.y_dtype = 2; a.scale = 1.f; a.ks = ks; a.cs = cs; a.z = z; a.cnt = cnt0;
                        decLaunch("bf", rgOf(R), 1, dim3((ntiles + tpc - 1) / tpc, 1, cs), &a);
                    });
                    report(nm, us2, double(wbytes));
                }
                continue;
            }
            if (d.fmt != 2) for (auto kc : kcs) {
                const int ks = kc[0], cs = kc[1];
                char nm[96];
                snprintf(nm, 96, "dec%dx%d %s R=%d", ks, cs, d.name, R);
                if (!want(nm)) continue;
                const int ntiles = d.N / 16, tpc = warps_cta / ks;
                double us = timeit(Ld, [&](int l) {
                    DecArgs a{};
                    a.q = q; a.s = s; a.ldq = d.K; a.lds = d.K / 32; a.w = (const uint8_t*)ws[l]; a.ws = (const uint8_t*)ss[l];
                    a.ldw = d.fmt == 1 ? d.K / 2 : d.K; a.ldws = d.K / 32; a.fmt = d.fmt; a.M = R; a.N = d.N; a.K = d.K;
                    a.y = y; a.ldy = d.N; a.y_dtype = 2; a.ks = ks; a.cs = cs; a.z = z; a.cnt = cnt0; a.scale = 1.f;
                    decLaunch(d.fmt == 1 ? "fp4" : "fp8", rgOf(R), 1, dim3((ntiles + tpc - 1) / tpc, 1, cs), &a);
                });
                report(nm, us, double(wbytes + sbytes));
            }
            for (int v = 0; v < 2; v++) {
                char nm[96];
                snprintf(nm, 96, "%s %s R=%d", v == 0 ? "gemv" : "sab", d.name, R);
                if (!want(nm)) continue;
                double us = timeit(Ld, [&](int l) {
                    MxArgs a{};
                    a.q = q; a.s = s; a.ldq = d.K; a.lds = d.K / 32; a.w = (const uint8_t*)ws[l]; a.ws = (const uint8_t*)ss[l];
                    a.ldw = d.fmt == 1 ? d.K / 2 : d.K; a.ldws = d.K / 32; a.fmt = d.fmt; a.M = R; a.N = d.N; a.K = d.K;
                    a.y = y; a.ldy = d.N; a.y_dtype = 2;
                    if (v == 0) tf_dsn_mx_gemv_kernel<<<(d.N + 7) / 8, 256, 0, st>>>(a);
                    else tf_dsn_mx_sab_kernel<<<(d.N + 15) / 16, 256, 0, st>>>(a);
                });
                report(nm, us, double(wbytes + sbytes));
            }
        }
        for (int l = 0; l < Ld; l++) { cudaFree(ws[l]); if (ss[l]) cudaFree(ss[l]); }
    }

    // ---------------------------------------------------------------- experts (a 2D node's layer)
    if (want("experts")) {
        const int E = 385, D = 5120, IH = 1152, DN = 2560, MAIN = 512, REST = 128, S = 7;
        const int LE = 4;
        struct Layer { std::vector<uint64_t> w1, w3, s1, s3, w2, s2; uint64_t *dw1, *dw3, *ds1, *ds3, *dw2, *ds2; int* owner; };
        std::vector<Layer> lay(LE);
        std::vector<int> owner_h(E);
        for (int e = 0; e < E; e++) owner_h[e] = (e % 2) == 0;
        for (auto& ly : lay) {
            for (int e = 0; e < E; e++) {
                const bool sh = e == E - 1;
                const int gr = MAIN + REST;  // every expert holds the rest rows (only owners' are read)
                const size_t gub = sh ? size_t(gr) * D : size_t(gr) * D / 2;
                const size_t gus = sh ? size_t(gr / 32) * (D / 32) : size_t(gr) * (D / 32);
                const size_t dnb = sh ? size_t(DN) * IH : size_t(DN) * IH / 2;
                const size_t dns = sh ? size_t(DN / 32) * (IH / 32) : size_t(DN) * (IH / 32);
                ly.w1.push_back((uint64_t)dbuf(gub, sh ? F_FP8 : F_FP4));
                ly.w3.push_back((uint64_t)dbuf(gub, sh ? F_FP8 : F_FP4));
                ly.s1.push_back((uint64_t)dbuf(gus, F_SCALE));
                ly.s3.push_back((uint64_t)dbuf(gus, F_SCALE));
                ly.w2.push_back((uint64_t)dbuf(dnb, sh ? F_FP8 : F_FP4));
                ly.s2.push_back((uint64_t)dbuf(dns, F_SCALE));
            }
            auto up = [&](std::vector<uint64_t>& v) { uint64_t* p; CK(cudaMalloc(&p, 8 * E)); CK(cudaMemcpy(p, v.data(), 8 * E, cudaMemcpyHostToDevice)); return p; };
            ly.dw1 = up(ly.w1); ly.dw3 = up(ly.w3); ly.ds1 = up(ly.s1); ly.ds3 = up(ly.s3); ly.dw2 = up(ly.w2); ly.ds2 = up(ly.s2);
            CK(cudaMalloc(&ly.owner, 4 * E));
            CK(cudaMemcpy(ly.owner, owner_h.data(), 4 * E, cudaMemcpyHostToDevice));
        }
        int *pick, *cnt, *off, *fill, *act, *nact, *ent, *erow;
        float* wts;
        CK(cudaMalloc(&pick, 4 * maxR * S));
        CK(cudaMalloc(&cnt, 4 * E)); CK(cudaMalloc(&off, 4 * (E + 1))); CK(cudaMalloc(&fill, 4 * E));
        CK(cudaMalloc(&act, 4 * E)); CK(cudaMalloc(&nact, 4)); CK(cudaMalloc(&ent, 4 * maxR * S)); CK(cudaMalloc(&erow, 4 * maxR * S));
        wts = (float*)dbuf(4 * maxR * S, F_ONE);
        __nv_bfloat16* h = (__nv_bfloat16*)dbuf(size_t(maxR) * S * IH * 2, F_ZERO);
        uint8_t* hq = (uint8_t*)dbuf(size_t(maxR) * S * IH, F_FP8);
        float* hs = (float*)dbuf(size_t(maxR) * S * IH / 32 * 4, F_ONE);
        float* out = (float*)dbuf(size_t(maxR) * S * DN * 4, F_ZERO);
        // NVFP4: each entry's codes and scales, gate / up's and down's, and the experts' alphas
        uint8_t* nq = (uint8_t*)dbuf(size_t(maxR) * S * D / 2, F_FP4);
        uint8_t* ns = (uint8_t*)dbuf(size_t(maxR) * S * D / 16, F_SCALE);
        uint8_t* hnq = (uint8_t*)dbuf(size_t(maxR) * S * IH / 2, F_FP4);
        uint8_t* hns = (uint8_t*)dbuf(size_t(maxR) * S * IH / 16, F_SCALE);
        float* alpha = (float*)dbuf(4 * E, F_ONE);
        for (int R : rows_list) {
            // picks: 6 distinct routed experts a row (random) and the shared expert last
            std::vector<int> ph(R * S);
            int distinct = 0;
            std::vector<int> seen(E, 0);
            for (int r = 0; r < R; r++) {
                for (int k = 0; k < 6; k++) {
                    int e;
                    bool dup;
                    do { e = rnd() % (E - 1); dup = false; for (int j = 0; j < k; j++) dup |= ph[r * S + j] == e; } while (dup);
                    ph[r * S + k] = e;
                    if (!seen[e]++) distinct++;
                }
                ph[r * S + 6] = E - 1;
            }
            CK(cudaMemcpy(pick, ph.data(), 4 * R * S, cudaMemcpyHostToDevice));
            const int RS = R * S;
            // the grouping once (the timing below is the launches that read weights)
            CK(cudaMemset(cnt, 0, 4 * E));
            tf_dsn_group_count_kernel<<<1, 256>>>(pick, RS, E, cnt);
            tf_dsn_group_scan_kernel<<<1, 32>>>(cnt, E, off, fill, act, nact);
            tf_dsn_group_place_kernel<<<1, 256>>>(pick, RS, S, E, off, fill, ent, erow);
            CK(cudaDeviceSynchronize());
            const int nown = [&] { int n = 0; for (int e = 0; e < E; e++) if (seen[e] && owner_h[e]) n++; return n; }();
            const double gu_bytes = (double(distinct) * (2.0 * MAIN * D / 2) + nown * (2.0 * REST * D / 2)) * 17 / 16 + 2.0 * (MAIN + REST * owner_h[E - 1]) * D;
            const double dn_bytes = double(distinct) * DN * IH / 2 * 17 / 16 + double(DN) * IH;
            const int ny = std::min(E, RS);
            auto gargs = [&](int l) {
                GuArgs g{};
                g.q = q; g.s = s; g.ldq = D; g.lds = D / 32; g.ent = ent; g.erow = erow; g.off = off; g.act = act; g.nact = nact;
                g.w1 = lay[l].dw1; g.w3 = lay[l].dw3; g.s1 = lay[l].ds1; g.s3 = lay[l].ds3; g.ldw = D / 2; g.ldws = D / 32;
                g.fmt = 1; g.D = D; g.shared = E - 1; g.sldw = D; g.sldws = D / 32; g.owner = lay[l].owner;
                g.n_tiles = (MAIN + REST) / 32; g.rest_tiles = REST / 32; g.wts = wts; g.limit = 10.f; g.h = h; g.ldh = MAIN; g.P = RS;
                g.splits = 1;
                return g;
            };
            auto dargs = [&](int l) {
                DnArgs d{};
                d.q = hq; d.s = hs; d.ldq = IH; d.lds = IH / 32; d.ent = ent; d.off = off; d.act = act; d.nact = nact;
                d.w2 = lay[l].dw2; d.s2 = lay[l].ds2; d.ldw = IH / 2; d.ldws = IH / 32; d.fmt = 1; d.shared = E - 1;
                d.sldw = IH; d.sldws = IH / 32; d.I = IH; d.N = DN; d.out = out; d.ldo = DN; d.splits = 1; d.P = RS;
                return d;
            };
            char nm[96];
            snprintf(nm, 96, "experts gu gemv R=%d (%d experts)", R, distinct + 1);
            if (want(nm)) report(nm, timeit(LE, [&](int l) { tf_dsn_gu_gemv_kernel<<<dim3((MAIN + REST + 7) / 8, ny), 256, 0, st>>>(gargs(l)); }), gu_bytes);
            snprintf(nm, 96, "experts gu sab R=%d (%d experts)", R, distinct + 1);
            if (want(nm)) report(nm, timeit(LE, [&](int l) { tf_dsn_gu_sab_kernel<<<dim3((MAIN + REST + 15) / 16, ny), 256, 0, st>>>(gargs(l)); }), gu_bytes);
            for (int ks : {1, 2, 4}) {
                snprintf(nm, 96, "experts gu dec%d R=%d (%d experts)", ks, R, distinct + 1);
                const int tpc = warps_cta / (2 * ks), nt = (MAIN + REST) / 16;
                if (want(nm)) report(nm, timeit(LE, [&](int l) { GuArgs g = gargs(l); g.splits = ks; decLaunch("gu", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &g); }), gu_bytes);
            }
            for (int ks : {1, 2, 4}) {
                snprintf(nm, 96, "experts gu nv%d R=%d (%d experts)", ks, R, distinct + 1);
                const int tpc = warps_cta / (2 * ks), nt = (MAIN + REST) / 16;
                if (want(nm)) report(nm, timeit(LE, [&](int l) { GuArgs g = gargs(l); g.splits = ks; g.fmt = 2; g.nq = nq; g.ns = ns; g.ldnq = D / 2; g.ldns = D / 16; g.a1 = alpha; g.a3 = alpha; decLaunch("gu", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &g); }), gu_bytes);
            }
            for (int ks : {1, 2, 4}) {
                snprintf(nm, 96, "experts dn nv%d R=%d (%d experts)", ks, R, distinct + 1);
                const int tpc = warps_cta / ks, nt = DN / 16;
                if (want(nm)) report(nm, timeit(LE, [&](int l) { DnArgs dd = dargs(l); dd.splits = ks; dd.fmt = 2; dd.nq = hnq; dd.ns = hns; dd.ldnq = IH / 2; dd.ldns = IH / 16; dd.a2 = alpha; decLaunch("dn", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &dd); }), dn_bytes);
            }
            for (int ks : {1, 2}) {
                snprintf(nm, 96, "experts dn dec%d R=%d (%d experts)", ks, R, distinct + 1);
                const int tpc = warps_cta / ks, nt = DN / 16;
                if (want(nm)) report(nm, timeit(LE, [&](int l) { DnArgs dd = dargs(l); dd.splits = ks; decLaunch("dn", rgOf(R), 1, dim3((nt + tpc - 1) / tpc, ny), &dd); }), dn_bytes);
            }
            snprintf(nm, 96, "experts dn gemv R=%d (%d experts)", R, distinct + 1);
            if (want(nm)) report(nm, timeit(LE, [&](int l) { tf_dsn_dn_gemv_kernel<<<dim3((DN + 7) / 8, ny), 256, 0, st>>>(dargs(l)); }), dn_bytes);
            snprintf(nm, 96, "experts dn sab R=%d (%d experts)", R, distinct + 1);
            if (want(nm)) report(nm, timeit(LE, [&](int l) { tf_dsn_dn_sab_kernel<<<dim3((DN + 15) / 16, ny), 256, 0, st>>>(dargs(l)); }), dn_bytes);
        }
    }
    return 0;
}
