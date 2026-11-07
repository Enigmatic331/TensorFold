//! DeepSeek's own checkpoint (config.Format.native) on the GPU: the kernels of zig/kernels/cuda/dsv41/native.cu
//! (cuda.kernels.dsv41_native) for its linears and experts, as DeepSeek's reference computes them (inference/model.py
//! `linear`, `Expert`, `MoE`; kernel.py `act_quant`, `fp8_gemm`, `fp4_gemm`): each activation rounded to bf16 and
//! quantized to E4M3 with a power-of-two scale per 32 values, FP8 weights with a UE8M0 scale per 32 x 32 block, MXFP4
//! experts with one per 32 values of a row, every 32-wide K block's sum scaled into fp32. A row's bits never depend on
//! the other rows of a launch, and a layer's K split depends on its shape alone, so drafted, concurrent and resumed rows
//! get a row's own bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const exl3_linear = @import("exl3_linear.zig");

pub const DType = exl3_linear.DType;

/// native.cu's MxArgs, field for field (C layout).
pub const MxArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    rows: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    y_dtype: c_int = 0,
    accumulate: c_int = 0,
};

/// native.cu's GuArgs.
pub const GuArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    ent: u64 = 0,
    erow: u64 = 0,
    off: u64 = 0,
    act: u64 = 0,
    nact: u64 = 0,
    w1: u64 = 0,
    w3: u64 = 0,
    s1: u64 = 0,
    s3: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 1,
    D: c_int = 0,
    shared: c_int = -1,
    sldw: i64 = 0,
    sldws: i64 = 0,
    owner: u64 = 0,
    n_tiles: c_int = 0,
    rest_tiles: c_int = 0,
    wts: u64 = 0,
    limit: f32 = 0,
    h: u64 = 0,
    ldh: i64 = 0,
    P: c_int = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    early_exit: c_int = 0, // Track C (TF_DS_EXPERT_EXIT)
    // fmt 2, NVFP4
    nq: u64 = 0,
    ns: u64 = 0,
    ldnq: i64 = 0,
    ldns: i64 = 0,
    a1: u64 = 0,
    a3: u64 = 0,
};

/// native.cu's DnArgs.
pub const DnArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    ent: u64 = 0,
    off: u64 = 0,
    act: u64 = 0,
    nact: u64 = 0,
    w2: u64 = 0,
    s2: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 1,
    shared: c_int = -1,
    sldw: i64 = 0,
    sldws: i64 = 0,
    I: c_int = 0,
    N: c_int = 0,
    round_bf16: c_int = 0,
    out: u64 = 0,
    ldo: i64 = 0,
    splits: c_int = 1,
    kblocks: c_int = 0,
    z: u64 = 0,
    P: c_int = 0,
    early_exit: c_int = 0, // Track C (TF_DS_EXPERT_EXIT)
    // fmt 2, NVFP4
    nq: u64 = 0,
    ns: u64 = 0,
    ldnq: i64 = 0,
    ldns: i64 = 0,
    a2: u64 = 0,
};

/// native.cu's BfArgs.
pub const BfArgs = extern struct {
    x: u64 = 0,
    ldx: i64 = 0,
    gx: i64 = 0,
    w: u64 = 0,
    ldw: i64 = 0,
    gw: i64 = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    gy: i64 = 0,
    y_dtype: c_int = 0,
    scale: f32 = 1,
    round_first: c_int = 0,
    splits: c_int = 1,
    ksteps: c_int = 0,
    z: u64 = 0,
};

/// native.cu's DecArgs (the decode tensor-core kernels' dense linears).
pub const DecArgs = extern struct {
    q: u64 = 0,
    s: u64 = 0,
    rows: u64 = 0,
    ldq: i64 = 0,
    lds: i64 = 0,
    w: u64 = 0,
    ws: u64 = 0,
    ldw: i64 = 0,
    ldws: i64 = 0,
    fmt: c_int = 0,
    M: c_int = 0,
    N: c_int = 0,
    K: c_int = 0,
    ks: c_int = 1,
    cs: c_int = 1,
    z: u64 = 0,
    cnt: u64 = 0,
    y: u64 = 0,
    ldy: i64 = 0,
    y_dtype: c_int = 0,
    accumulate: c_int = 0,
    scale: f32 = 1,
    round_first: c_int = 0,
    gn: c_int = 0,
    gq: i64 = 0,
    gs: i64 = 0,
};

/// native.cu's DecMulti: up to four dense linears of one format and row count in one launch.
pub const DecMulti = extern struct {
    p: [4]DecArgs = @splat(.{}),
    n: c_int = 0,
    cta0: [5]c_int = @splat(0),
};

comptime {
    std.debug.assert(@sizeOf(DecMulti) == 4 * 168 + 24 and @offsetOf(DecMulti, "cta0") == 676);
    std.debug.assert(@sizeOf(DecArgs) == 168 and @offsetOf(DecArgs, "z") == 96 and @offsetOf(DecArgs, "y") == 112 and @offsetOf(DecArgs, "gq") == 152);
}

/// The decode tensor-core kernels' shared memory (native.cu namespace dec: kRbD / kNsD for the dense linears, kRbE /
/// kNsE for the experts, slotBytes, warpBytes): a CTA of dec_warps warps, each warp its ring of stages or its partials.
pub const DecCfg = struct { rowb: usize, stages: usize };
pub const dec_dense_cfg: DecCfg = .{ .rowb = 64, .stages = 4 };
pub const dec_experts_cfg: DecCfg = .{ .rowb = 64, .stages = 2 }; // down
pub const dec_gu_cfg: DecCfg = .{ .rowb = 128, .stages = 2 }; // gate / up
pub const dec_warps = 8;
fn decWarpBytes(cfg: DecCfg, rg: usize) usize {
    return @max(cfg.stages * (16 * cfg.rowb + 128), rg * 4 * 32 * 4);
}
fn decShared(cfg: DecCfg, rg: usize) u32 {
    return @intCast(dec_warps * decWarpBytes(cfg, rg));
}
/// The row groups (of 8) a launch's template holds for m rows: 1, 2, 4, 8 (index 0..3); past 64 rows the experts' kernels
/// run passes of 64 (the same arithmetic a row).
fn decRg(m: usize) usize {
    return if (m <= 8) 0 else if (m <= 16) 1 else if (m <= 32) 2 else 3;
}
const dec_counters = 16384;

comptime {
    // the layouts native.cu was compiled with (pointers and int64 8 bytes, ints and floats 4, C padding)
    std.debug.assert(@sizeOf(MxArgs) == 128 and @offsetOf(MxArgs, "z") == 96);
    std.debug.assert(@offsetOf(GuArgs, "sldw") == 136 and @offsetOf(GuArgs, "owner") == 152 and @offsetOf(GuArgs, "z") == 216 and @offsetOf(GuArgs, "early_exit") == 224 and @offsetOf(GuArgs, "nq") == 232 and @sizeOf(GuArgs) == 280);
    std.debug.assert(@offsetOf(DnArgs, "sldw") == 104 and @offsetOf(DnArgs, "out") == 136 and @offsetOf(DnArgs, "z") == 160 and @offsetOf(DnArgs, "early_exit") == 172 and @offsetOf(DnArgs, "nq") == 176 and @sizeOf(DnArgs) == 216);
    std.debug.assert(@offsetOf(BfArgs, "y") == 64 and @offsetOf(BfArgs, "z") == 112 and @sizeOf(BfArgs) == 120);
}

/// The kernels of cuda.kernels.dsv41_native.
pub const Kernels = struct {
    module: cuda.Module,
    act_quant: cuda.Function,
    gemm: [4]cuda.Function, // row tiles of 1, 2, 4, 8 m16 tiles
    reduce: cuda.Function,
    count: cuda.Function,
    scan: cuda.Function,
    place: cuda.Function,
    gu: [3]cuda.Function, // 1, 2, 4
    dn: [3]cuda.Function,
    gu_finish: cuda.Function,
    dn_finish: cuda.Function,
    gu_gemv: cuda.Function,
    dn_gemv: cuda.Function,
    gemv: cuda.Function,
    mx_sab: cuda.Function, // the decode MMA with the operands swapped (16 weight rows by up to 8 activation rows)
    gu_sab: cuda.Function,
    dn_sab: cuda.Function,
    sab: bool, // decode launches on the swap-AB MMA (TF_DS_DECODE_SAB=1; else the GEMV kernels, faster at a round's
    // 1-3 rows on GB10: forward 45.7 against 49.9 ms a round, measured 2026-10-17)
    // the decode tensor-core kernels (tf_dsn_dec_*): [kind][row groups 1, 2, 4, 8], kinds fp8, fp4, bf16 (DecArgs), gate /
    // up (GuArgs), down (DnArgs); the cross-CTA split's counters (int zeros, each launch leaves them zero)
    dec: [5][4]cuda.Function,
    dec_group: cuda.Function, // a decode launch's grouping in one CTA
    dec_quant: cuda.Function, // act_quant for decode launches (a programmatic dependent launch)
    dec_l2pf: cuda.Function, // the active experts' down weights into L2 (beside the 2D exchange)
    dec_on: bool, // TF_DS_DECODE_DEC=0: the GEMV kernels instead (the reference path, measured slower)
    // TF_DS_DEC_PDL=0: the decode tensor-core kernels (dense, gate / up, down) as ordinary launches, the same kernels (the
    // decode plan's G0.2 control (a)); unset or any other value: programmatic dependent launches as served
    dec_pdl: bool,
    // TF_DS_EXPERT_EXIT=1 (Track C): the decode expert kernels' grid rows past the active experts return at once
    expert_exit: bool,
    counters: cuda.DeviceBuffer,
    combine: cuda.Function,
    bf16: cuda.Function,
    bf_reduce: cuda.Function,
    to_f16: cuda.Function,
    // prompt chunks' tile GEMMs (native.cu tf_dsn_pf_*: the tile kernels' bits, operands staged through shared memory)
    pf_gemm: cuda.Function,
    pf_gemm_fold: cuda.Function,
    pf_gu: cuda.Function,
    pf_dn: cuda.Function,
    pf_gu_nv: cuda.Function, // the same with NVFP4 routed experts
    pf_dn_nv: cuda.Function,
    nv_quant: cuda.Function, // NVFP4's activations, an entry's own (a programmatic dependent launch's predecessor)
    pf_tiles: cuda.Function,
    pf_bf16: cuda.Function,
    pf_off: bool, // TF_DS_PF_TILES=0: prompt chunks on the tile kernels (the same bits)

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var dec: [5][4]cuda.Function = undefined;
        const kinds = [_][]const u8{ "fp8", "fp4", "bf", "gu", "dn" };
        for (kinds, 0..) |kn, i| for (0..4) |r| {
            var nb: [64]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&nb, "tf_dsn_dec_{s}_r{d}_kernel", .{ kn, @as(usize, 1) << @intCast(r) }, 0);
            dec[i][r] = try m.function(name);
            try dec[i][r].allowDynamicShared(decShared(if (i < 3) dec_dense_cfg else if (i == 3) dec_gu_cfg else dec_experts_cfg, @as(usize, 1) << @intCast(r)));
        };
        var counters = try cuda.DeviceBuffer.alloc(d, dec_counters * 4);
        errdefer counters.free();
        try d.check(d.api.cuMemsetD32_v2(counters.ptr, 0, dec_counters), "cuMemsetD32");
        return .{
            .dec = dec,
            .dec_group = try m.function("tf_dsn_dec_group_kernel"),
            .dec_quant = try m.function("tf_dsn_dec_quant_kernel"),
            .dec_l2pf = try m.function("tf_dsn_dec_l2pf_kernel"),
            .dec_on = if (std.c.getenv("TF_DS_DECODE_DEC")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true,
            .dec_pdl = if (std.c.getenv("TF_DS_DEC_PDL")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true,
            .expert_exit = if (std.c.getenv("TF_DS_EXPERT_EXIT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
            .counters = counters,
            .module = m,
            .act_quant = try m.function("tf_dsn_act_quant_kernel"),
            .gemm = .{ try m.function("tf_dsn_mx_gemm1_kernel"), try m.function("tf_dsn_mx_gemm2_kernel"), try m.function("tf_dsn_mx_gemm4_kernel"), try m.function("tf_dsn_mx_gemm8_kernel") },
            .reduce = try m.function("tf_dsn_reduce_kernel"),
            .count = try m.function("tf_dsn_group_count_kernel"),
            .scan = try m.function("tf_dsn_group_scan_kernel"),
            .place = try m.function("tf_dsn_group_place_kernel"),
            .gu = .{ try m.function("tf_dsn_gu1_kernel"), try m.function("tf_dsn_gu2_kernel"), try m.function("tf_dsn_gu4_kernel") },
            .dn = .{ try m.function("tf_dsn_dn1_kernel"), try m.function("tf_dsn_dn2_kernel"), try m.function("tf_dsn_dn4_kernel") },
            .gu_finish = try m.function("tf_dsn_gu_finish_kernel"),
            .dn_finish = try m.function("tf_dsn_dn_finish_kernel"),
            .gu_gemv = try m.function("tf_dsn_gu_gemv_kernel"),
            .dn_gemv = try m.function("tf_dsn_dn_gemv_kernel"),
            .gemv = try m.function("tf_dsn_mx_gemv_kernel"),
            .mx_sab = try m.function("tf_dsn_mx_sab_kernel"),
            .gu_sab = try m.function("tf_dsn_gu_sab_kernel"),
            .dn_sab = try m.function("tf_dsn_dn_sab_kernel"),
            .sab = if (std.c.getenv("TF_DS_DECODE_SAB")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
            .combine = try m.function("tf_dsn_combine_kernel"),
            .bf16 = try m.function("tf_dsn_bf16_gemm_kernel"),
            .bf_reduce = try m.function("tf_dsn_bf_reduce_kernel"),
            .to_f16 = try m.function("tf_dsn_bf16_to_f16_kernel"),
            .pf_gemm = try pfFunction(m, "tf_dsn_pf_gemm_kernel", pf_gemm_smem),
            .pf_gemm_fold = try pfFunction(m, "tf_dsn_pf_gemm_fold_kernel", pf_gemm_smem),
            .pf_gu = try pfFunction(m, "tf_dsn_pf_gu_kernel", pf_gu_smem),
            .pf_dn = try pfFunction(m, "tf_dsn_pf_dn_kernel", pf_dn_smem),
            .pf_gu_nv = try pfFunction(m, "tf_dsn_pf_gu_nv_kernel", pf_gu_smem),
            .pf_dn_nv = try pfFunction(m, "tf_dsn_pf_dn_nv_kernel", pf_dn_smem),
            .nv_quant = try m.function("tf_dsn_nv_quant_kernel"),
            .pf_tiles = try m.function("tf_dsn_pf_tiles_kernel"),
            .pf_bf16 = try pfFunction(m, "tf_dsn_pf_bf16_kernel", pf_bf16_smem),
            .pf_off = if (std.c.getenv("TF_DS_PF_TILES")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false,
        };
    }

    fn pfFunction(m: cuda.Module, name: [:0]const u8, smem: u32) !cuda.Function {
        const f = try m.function(name);
        try f.allowDynamicShared(smem);
        return f;
    }

    pub fn unload(k: *Kernels) void {
        k.counters.free();
        k.module.unload();
        k.* = undefined;
    }
};

fn launch(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

/// The prefill tile kernels' dynamic shared memory (native.cu asserts the same: pf::kGemmSmem, kGuSmem, kDnSmem).
pub const pf_gemm_smem: u32 = 76800;
pub const pf_gu_smem: u32 = 79872;
pub const pf_dn_smem: u32 = 53760;
pub const pf_bf16_smem: u32 = 81920;
/// The grouped prefill tiles' rows (native.cu pf::kBm).
pub const pf_bm = 64;
/// Rows from which a dense prompt GEMM takes the prefill tile kernel (fewer: the tile kernel; the same bits either way).
pub const pf_min_rows = 64;

fn launchShared(f: cuda.Function, s: cuda.Stream, grid: [3]usize, smem: u32, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = 256 }, .shared = smem }, s, args);
}

fn aligned(p: u64, a: u64) bool {
    return p % a == 0;
}

/// Whether a dense FP8 linear on these operands fits the prefill GEMM (tf_dsn_pf_gemm: N a multiple of 128, K of 64 up
/// to 8192, cp.async-aligned rows).
fn pfDense(k: *const Kernels, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, accumulate: bool) bool {
    if (k.pf_off or m < pf_min_rows or l.fmt != .fp8) return false;
    if (l.n % 128 != 0 or l.k % 64 != 0 or l.k > 8192) return false;
    if (!aligned(q, 16) or ldq % 16 != 0 or !aligned(sc, 8) or lds % 2 != 0 or !aligned(l.w, 16) or l.k % 16 != 0) return false;
    if (accumulate and (!aligned(y, 8) or ldy % 2 != 0)) return false;
    return true;
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// `n` bf16 values into fp16 (round to nearest).
pub fn bf16ToF16(k: *const Kernels, s: cuda.Stream, in: u64, out: u64, n: usize) !void {
    var a: cuda.Args = .{};
    a.add(in);
    a.add(out);
    a.add(@as(i64, @intCast(n)));
    try launch(k.to_f16, s, .{ cdiv(n, 256), 1, 1 }, 256, &a);
}

/// The rows a GEMM launch takes at most (larger row counts run as launches of this many, each its own rows' bits):
/// it bounds the split partials' buffer.
pub const slab_rows = 1024;

/// A layer's K splits, from its shape alone (a row's sum is the same at every row count): enough CTAs over N (four
/// 32-column warps a CTA) to fill the GPU twice, at least 8 K blocks a split, at most 16 splits.
pub fn splits(k: usize, n: usize) usize {
    const kb = k / 32;
    const ctas = cdiv(n / 32, 4);
    var s: usize = 1;
    while (ctas * s < 96 and s < 16 and kb / (s * 2) >= 8) s *= 2;
    return s;
}

/// A decode-size bf16 linear's K splits (16-column steps), from its shape alone, as splits() for the MX GEMM.
pub fn bfSplits(k: usize, n: usize) usize {
    const steps = k / 16;
    const ctas = cdiv(n / 32, 4);
    var s: usize = 1;
    while (ctas * s < 96 and s < 16 and steps / (s * 2) >= 8) s *= 2;
    return s;
}

/// The decode tensor-core kernels' K split of a dense linear, from its shape alone (fmt 0 FP8, 1 FP4, 2 bf16; n
/// output columns, k inputs): ks warps a 16-column tile in a CTA of dec_warps (each warp keeping at least two stages of
/// a row), cs CTAs a tile (grid z, the last CTA summing their partials in order). Measured on GB10 (decode_bench.cu):
/// a tile's K over all 8 warps of its CTA was the fastest for every shape of the model, and splitting it over CTAs
/// as well was not faster.
pub const DecSplit = struct { ks: usize, cs: usize };
pub fn decSplit(fmt: u2, n: usize, k: usize) DecSplit {
    _ = n;
    const rb: usize = switch (fmt) {
        0 => k,
        1 => k / 2,
        else => 2 * k,
    };
    const nst = cdiv(rb, dec_dense_cfg.rowb);
    var ks: usize = dec_warps;
    while (ks > 1 and nst < 2 * ks) ks /= 2;
    return .{ .ks = ks, .cs = 1 };
}

/// Whether the decode tensor-core kernels take a dense linear of this shape (else the GEMV / the bf16 GEMM: a choice by
/// shape alone).
pub fn decTakes(k: *const Kernels, fmt: u2, n: usize, kdim: usize) bool {
    if (!k.dec_on or n % 16 != 0 or n == 0) return false;
    return switch (fmt) {
        0, 1 => kdim % 128 == 0,
        else => kdim % 32 == 0,
    };
}

/// Floats of split partials a decode tensor-core launch of `rows` rows needs.
pub fn decZFloats(fmt: u2, n: usize, kdim: usize, rows: usize) usize {
    const sp = decSplit(fmt, n, kdim);
    return if (sp.cs > 1) sp.cs * rows * n else 0;
}

/// Dense decode linears on the tensor-core kernels in one launch (each `as`: every field but the split's; one kind, 0
/// FP8, 1 FP4, 2 bf16, and one row count), each as its own launch would compute it.
/// pdl: a programmatic dependent launch (its weights' prologue beside the predecessor), only right after quantDec:
/// the predecessor must be an ordinary launch (all earlier work done when it started) that lets it start early.
/// Past decode_rows rows (the drafter's pass of 13 or more streams: five rows a stream) the rows run in slabs of
/// decode_rows, each its own launch with the slab's row pointers and the split, quantization and epilogue it has alone
/// (the templates store at most decode_rows rows; a row's bits do not depend on the launch's row count). Only the first
/// slab may be a programmatic dependent launch: the later ones follow a decode kernel, so they are ordinary launches.
fn decDenseMulti(k: *const Kernels, s: cuda.Stream, kind: u2, as: []const DecArgs, z: ?Z, pdl: bool) !void {
    if (as.len == 0 or as.len > 4) return error.UnsupportedShape;
    const m: usize = @intCast(as[0].M);
    if (m <= decode_rows) return decDenseLaunch(k, s, kind, as, z, pdl);
    for (as) |a| if (a.M != as[0].M) return error.UnsupportedShape;
    var r0: usize = 0;
    while (r0 < m) : (r0 += decode_rows) {
        var sl: [4]DecArgs = undefined;
        for (as, 0..) |a, i| sl[i] = decSlab(a, r0, @min(decode_rows, m - r0));
        try decDenseLaunch(k, s, kind, sl[0..as.len], z, pdl and r0 == 0);
    }
}

/// Rows r0 .. r0 + mm of a decode launch's arguments: the activation rows (a row list, else the codes and scales: FP8 /
/// FP4 codes ldq bytes a row, bf16 rows ldq elements; scales lds floats) and the output rows (ldy elements of y_dtype).
fn decSlab(a: DecArgs, r0: usize, mm: usize) DecArgs {
    var b = a;
    b.M = @intCast(mm);
    if (a.rows != 0) {
        b.rows = a.rows + r0 * 4;
    } else {
        const qsz: usize = if (a.fmt == 2) 2 else 1;
        b.q = a.q + r0 * @as(usize, @intCast(a.ldq)) * qsz;
        if (a.s != 0) b.s = a.s + r0 * @as(usize, @intCast(a.lds)) * 4;
    }
    const ysz: usize = if (a.accumulate != 0 or a.y_dtype == @intFromEnum(DType.f32)) 4 else 2;
    b.y = a.y + r0 * @as(usize, @intCast(a.ldy)) * ysz;
    return b;
}

fn decDenseLaunch(k: *const Kernels, s: cuda.Stream, kind: u2, as: []const DecArgs, z: ?Z, pdl: bool) !void {
    var mm: DecMulti = .{ .n = @intCast(as.len) };
    const m: usize = @intCast(as[0].M);
    if (m > decode_rows) return error.NotADecodeWindow;
    var ctas: usize = 0;
    var zoff: usize = 0;
    var cs_all: usize = 0;
    for (as, 0..) |a0, i| {
        var a = a0;
        if (a.M != as[0].M) return error.UnsupportedShape;
        const n: usize = @intCast(a.N);
        const sp = decSplit(kind, n, @intCast(a.K));
        if (i == 0) cs_all = sp.cs else if (sp.cs != cs_all) return error.UnsupportedShape;
        a.ks = @intCast(sp.ks);
        a.cs = @intCast(sp.cs);
        const groups = cdiv(n / 16, dec_warps / sp.ks);
        if (sp.cs > 1) {
            const zz = z orelse return error.ScratchTooSmall;
            if (zoff + sp.cs * m * n > zz.floats or ctas + groups > dec_counters) return error.ScratchTooSmall;
            a.z = zz.ptr + zoff * 4;
            a.cnt = k.counters.ptr + ctas * 4;
            zoff += sp.cs * m * n;
        }
        mm.p[i] = a;
        mm.cta0[i] = @intCast(ctas);
        ctas += groups;
    }
    mm.cta0[as.len] = @intCast(ctas);
    const rg = decRg(m);
    var args: cuda.Args = .{};
    args.add(mm);
    try cuda.launch.launch(k.dec[kind][rg], .{ .grid = .{ .x = @intCast(ctas), .y = 1, .z = @intCast(cs_all) }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_dense_cfg, @as(usize, 1) << @intCast(rg)), .pdl = pdl and k.dec_pdl }, s, &args);
}

fn decDense(k: *const Kernels, s: cuda.Stream, kind: u2, a: DecArgs, z: ?Z) !void {
    return decDenseMulti(k, s, kind, &.{a}, z, false);
}

/// The rows a decode-size launch takes at most (a round's window, the drafter's rows): what the split partials of
/// gate / up, down and the bf16 linears are sized for (decodeZFloats).
pub const decode_rows = 64;

/// Split partials (fp32) for decode-size launches: the place and its size in floats. Decode launches split K by their
/// shape alone (so a row's bits are the same at every row count); prompt chunks never split them.
pub const Z = struct { ptr: u64, floats: usize };

/// Floats of decode split partials a layer's experts and bf16 linears need, for `rows` rows of `slots` slots
/// (n_experts: the router's outputs; ih: the indexer's heads, weights_proj's outputs).
pub fn decodeZFloats(lay: weights.Layer, hidden: usize, n_experts: usize, ih: usize, rows: usize, slots: usize) usize {
    var need: usize = 0;
    const ex = lay.experts;
    if (ex.native) |nx| {
        const p = rows * slots;
        const ncols = nx.tiles * 32;
        need = @max(need, splits(ex.dims, ncols) * 2 * p * ncols);
        need = @max(need, splits(ex.down_k, ex.down_n) * p * ex.down_n);
    }
    if (lay.gate_w != 0) need = @max(need, bfSplits(hidden, n_experts) * rows * n_experts);
    if (lay.idx_proj != 0 and ih % 32 == 0) need = @max(need, bfSplits(hidden, ih) * rows * ih);
    for (lay.wo_a[0..lay.groups]) |wo| if (wo.fmt == .bf16) {
        need = @max(need, bfSplits(wo.k, wo.n) * rows * wo.n);
    };
    // the decode linears that may be bf16 (mx.linear decode)
    for ([_]?weights.Linear{ lay.wq_a, lay.wkv, lay.wq_b, lay.wo_b, lay.comp_wkv, lay.comp_wgate, lay.idx_wq_b, lay.idx_wk, lay.engram_wkv }) |ol| {
        const l = ol orelse continue;
        if (l.fmt == .bf16) need = @max(need, bfSplits(l.k, l.n) * rows * l.n);
    }
    // the decode tensor-core kernels' cross-CTA partials (every decode linear, the router, the indexer's weights_proj)
    for ([_]?weights.Linear{ lay.wq_a, lay.wkv, lay.wq_b, lay.wo_b, lay.comp_wkv, lay.comp_wgate, lay.idx_wq_b, lay.idx_wk, lay.engram_wkv, lay.wo_a8 }) |ol| {
        const l = ol orelse continue;
        const f: u2 = switch (l.fmt) {
            .fp8 => 0,
            .fp4 => 1,
            .bf16 => 2,
            .exl3 => continue,
        };
        need = @max(need, decZFloats(f, l.n, l.k, rows));
    }
    if (lay.gate_w != 0) need = @max(need, decZFloats(2, n_experts, hidden, rows));
    if (lay.idx_proj != 0) need = @max(need, decZFloats(2, ih, hidden, rows));
    return need;
}

/// The m16 row tiles a warp holds (1, 2, 4, 8) for `m` rows: the launch's own choice, no bits change with it.
fn tiles(m: usize) usize {
    return if (m <= 16) 0 else if (m <= 32) 1 else if (m <= 64) 2 else 3;
}

/// A linear's scratch: the activations' codes and scales, the split partials, the quant-dequant rows (wo_a's input).
pub const Scratch = struct {
    q: u64, // u8 [rows, max K]
    s: u64, // f32 [rows, max K / 32]
    z: u64, // f32 [zFloats]
    fake: u64, // bf16 [rows, max K]
    rows: usize,
    max_k: usize,

    /// Bytes of q, s, z and fake for `rows` rows of K up to `max_k` and the largest split partials `zf` floats.
    pub fn sizes(rows: usize, max_k: usize, zf: usize) [4]usize {
        return .{ rows * max_k, rows * (max_k / 32) * 4, @max(zf, 1) * 4, rows * max_k * 2 };
    }
};

/// Floats of split partials a linear needs (a slab of rows at most).
pub fn zFloats(l: weights.Linear, rows: usize) usize {
    if (l.fmt != .fp8 and l.fmt != .fp4) return 0;
    const s = splits(l.k, l.n);
    return if (s > 1) s * @min(rows, slab_rows) * l.n else 0;
}

/// act_quant: x [m, kdim] (row stride ldx, dtype) -> q (row stride ldq) and its scales (row stride lds).
pub fn quant(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, sc: u64, lds: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @intFromEnum(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(q);
    a.add(@as(i64, @intCast(ldq)));
    a.add(sc);
    a.add(@as(i64, @intCast(lds)));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    try launch(k.act_quant, s, .{ cdiv(m * (kdim / 32) * 32, 256), 1, 1 }, 256, &a);
}

/// quant for a decode launch (the same codes and scales): with dec_on tf_dsn_dec_quant, an ordinary launch (every
/// earlier kernel done when it starts) that lets the decode tensor-core kernel after it, a programmatic dependent
/// launch, start streaming its first weights meanwhile (that kernel waits for it before reading the codes).
pub fn quantDec(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, q: u64, ldq: usize, sc: u64, lds: usize) !void {
    if (!k.dec_on) return quant(k, s, x, x_dtype, ldx, m, kdim, q, ldq, sc, lds);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @intFromEnum(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(q);
    a.add(@as(i64, @intCast(ldq)));
    a.add(sc);
    a.add(@as(i64, @intCast(lds)));
    try cuda.launch.launch(k.dec_quant, .{ .grid = .{ .x = @intCast(cdiv(m * (kdim / 32) * 32, 256)), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
}

/// act_quant(..., inplace=True): x's quant-dequant rows into out [m, kdim] bf16 (row stride ldo).
pub fn fake(k: *const Kernels, s: cuda.Stream, x: u64, x_dtype: DType, ldx: usize, m: usize, kdim: usize, out: u64, ldo: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(c_int, @intFromEnum(x_dtype)));
    a.add(@as(i64, @intCast(ldx)));
    a.add(@as(c_int, @intCast(m)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    a.add(out);
    a.add(@as(i64, @intCast(ldo)));
    try launch(k.act_quant, s, .{ cdiv(m * (kdim / 32) * 32, 256), 1, 1 }, 256, &a);
}

/// The MX GEMM of a quantized activation (q, sc: row strides ldq, lds) with a native FP8 / FP4 linear: y [m, n]
/// (row stride ldy, dtype; accumulate: fp32 y +=), split as splits(K, N), slabs of slab_rows rows.
pub fn gemm(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: u64) !void {
    if (l.fmt != .fp8 and l.fmt != .fp4) return error.NotAnMxLinear;
    const sp = splits(l.k, l.n);
    const esz: usize = switch (y_dtype) {
        .f32 => 4,
        else => 2,
    };
    if (pfDense(k, l, q, ldq, sc, lds, m, y, ldy, accumulate)) {
        // the prefill tile GEMM: every row at once, a split folded in the CTA (the tile kernel's and reduce's bits)
        const a: MxArgs = .{ .q = q, .s = sc, .ldq = @intCast(ldq), .lds = @intCast(lds), .w = l.w, .ws = l.s, .ldw = l.k, .ldws = l.k / 32, .fmt = 0, .M = @intCast(m), .N = @intCast(l.n), .K = @intCast(l.k), .splits = @intCast(sp), .kblocks = @intCast(cdiv(l.k / 32, sp)), .y = y, .ldy = @intCast(ldy), .y_dtype = @intFromEnum(y_dtype), .accumulate = @intFromBool(accumulate) };
        var args: cuda.Args = .{};
        args.add(a);
        if (sp > 1) return launchShared(k.pf_gemm_fold, s, .{ cdiv(m, 64), l.n / 128, 1 }, pf_gemm_smem, &args);
        return launchShared(k.pf_gemm, s, .{ cdiv(m, 128), l.n / 128, 1 }, pf_gemm_smem, &args);
    }
    var r0: usize = 0;
    while (r0 < m) : (r0 += slab_rows) {
        const mm = @min(slab_rows, m - r0);
        const t = tiles(mm);
        const a: MxArgs = .{
            .q = q + r0 * ldq,
            .s = sc + r0 * lds * 4,
            .ldq = @intCast(ldq),
            .lds = @intCast(lds),
            .w = l.w,
            .ws = l.s,
            .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
            .ldws = if (l.fmt == .fp4) l.k / 32 else l.k / 32,
            .fmt = if (l.fmt == .fp4) 1 else 0,
            .M = @intCast(mm),
            .N = @intCast(l.n),
            .K = @intCast(l.k),
            .splits = @intCast(sp),
            .kblocks = @intCast(cdiv(l.k / 32, sp)),
            .z = z,
            .y = y + r0 * ldy * esz,
            .ldy = @intCast(ldy),
            .y_dtype = @intFromEnum(y_dtype),
            .accumulate = @intFromBool(accumulate),
        };
        var args: cuda.Args = .{};
        args.add(a);
        const rows_cta = (@as(usize, 1) << @intCast(t)) * 16;
        try launch(k.gemm[t], s, .{ cdiv(l.n / 32, 4), sp, cdiv(mm, rows_cta) }, 128, &args);
        if (sp > 1) {
            var ra: cuda.Args = .{};
            ra.add(z);
            ra.add(@as(c_int, @intCast(sp)));
            ra.add(@as(c_int, @intCast(mm)));
            ra.add(@as(c_int, @intCast(l.n)));
            ra.add(a.y);
            ra.add(@as(i64, @intCast(ldy)));
            ra.add(@as(c_int, @intFromEnum(y_dtype)));
            ra.add(@as(c_int, @intFromBool(accumulate)));
            try launch(k.reduce, s, .{ cdiv(mm * l.n, 256), 1, 1 }, 256, &ra);
        }
    }
}

/// The decode GEMV of a quantized activation with a native FP8 / FP4 linear (gemm's arguments, no split): a warp a
/// column, each row its own arithmetic, so a row's bits are the same at any row count.
pub fn gemv(k: *const Kernels, s: cuda.Stream, l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, z: ?Z) !void {
    if (l.fmt != .fp8 and l.fmt != .fp4) return error.NotAnMxLinear;
    const f: u2 = if (l.fmt == .fp4) 1 else 0;
    if (decTakes(k, f, l.n, l.k)) return decDense(k, s, f, .{
        .q = q,
        .s = sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
        .ldws = l.k / 32,
        .fmt = f,
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @intFromEnum(y_dtype),
        .accumulate = @intFromBool(accumulate),
    }, z);
    const a: MxArgs = .{
        .q = q,
        .s = sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = if (l.fmt == .fp4) l.k / 2 else l.k,
        .ldws = l.k / 32,
        .fmt = if (l.fmt == .fp4) 1 else 0,
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @intFromEnum(y_dtype),
        .accumulate = @intFromBool(accumulate),
    };
    var args: cuda.Args = .{};
    args.add(a);
    if (k.sab) return launch(k.mx_sab, s, .{ cdiv(l.n, 16), 1, 1 }, 256, &args);
    try launch(k.gemv, s, .{ cdiv(l.n, 8), 1, 1 }, 256, &args);
}

/// A bf16 linear with fp32 accumulation: y [m, n] = x [m, K] bf16 . w [n, K]^T (F.linear of bf16 values), scaled by
/// `scale` (round_first: of the bf16-rounded sum, as a bf16 Linear's output times a constant), `groups` block-diagonal
/// groups (x + g gx, w + g gw, y + g gy elements).
/// With `z` (a decode launch: one group, its rows within z) K is split as bfSplits(kdim, n), the partials summed in
/// split order; without it the K loop runs whole (prompt chunks).
pub fn bf16Gemm(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: u64, ldw: usize, m: usize, n: usize, kdim: usize, y: u64, ldy: usize, y_dtype: DType, scale: f32, round_first: bool, groups: usize, gx: usize, gw: usize, gy: usize, z: ?Z) !void {
    if (n % 32 != 0 or kdim % 16 != 0) return error.UnsupportedShape;
    if (z != null and groups == 1 and m > decode_rows) {
        // a decode launch past decode_rows rows (the drafter's router at 13+ streams): slabs of decode_rows rows, each
        // as a launch of its own rows (the split partials are sized for decode_rows rows; a row's bits are its own)
        const ysz: usize = if (y_dtype == .f32) 4 else 2;
        var r0: usize = 0;
        while (r0 < m) : (r0 += decode_rows)
            try bf16Gemm(k, s, x + r0 * ldx * 2, ldx, w, ldw, @min(decode_rows, m - r0), n, kdim, y + r0 * ldy * ysz, ldy, y_dtype, scale, round_first, 1, gx, gw, gy, z);
        return;
    }
    if (z != null and groups == 1 and decTakes(k, 2, n, kdim)) return decDense(k, s, 2, .{
        .q = x,
        .ldq = @intCast(ldx),
        .w = w,
        .ldw = @intCast(ldw * 2),
        .fmt = 2,
        .M = @intCast(m),
        .N = @intCast(n),
        .K = @intCast(kdim),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @intFromEnum(y_dtype),
        .scale = scale,
        .round_first = @intFromBool(round_first),
    }, z);
    var a: BfArgs = .{ .x = x, .ldx = @intCast(ldx), .gx = @intCast(gx), .w = w, .ldw = @intCast(ldw), .gw = @intCast(gw), .M = @intCast(m), .N = @intCast(n), .K = @intCast(kdim), .y = y, .ldy = @intCast(ldy), .gy = @intCast(gy), .y_dtype = @intFromEnum(y_dtype), .scale = scale, .round_first = @intFromBool(round_first) };
    const sp = if (z != null) bfSplits(kdim, n) else 1;
    if (sp > 1) {
        const zz = z.?;
        if (groups != 1) return error.UnsupportedShape;
        // the partials' buffer bounds the rows (a round's window, or the drafter's pass of every stream's block)
        if (sp * m * n > zz.floats) return error.ScratchTooSmall;
        a.splits = @intCast(sp);
        a.ksteps = @intCast(cdiv(kdim / 16, sp));
        a.z = zz.ptr;
    }
    var args: cuda.Args = .{};
    args.add(a);
    if (sp == 1 and !k.pf_off and m >= pf_min_rows and n % 128 == 0 and kdim % 32 == 0 and aligned(x, 16) and ldx % 8 == 0 and
        gx % 8 == 0 and aligned(w, 16) and ldw % 8 == 0 and gw % 8 == 0)
    {
        // the prefill bf16 GEMM: the same MMA chain (the tile kernel's bits), staged tiles
        return launchShared(k.pf_bf16, s, .{ cdiv(m, 128), n / 128, groups }, pf_bf16_smem, &args);
    }
    try launch(k.bf16, s, .{ cdiv(n / 32, 4), @min(cdiv(m, 16), 64), if (sp > 1) sp else groups }, 128, &args);
    if (sp > 1) {
        var ra: cuda.Args = .{};
        ra.add(a);
        try launch(k.bf_reduce, s, .{ cdiv(m * n, 256), 1, 1 }, 256, &ra);
    }
}

/// A native linear on m rows of x (dtype x_dtype, row stride ldx) into y (row stride ldy, y_dtype): FP8 / FP4 through
/// act_quant and the MX GEMM, bf16 through the bf16 GEMM (x must be bf16 then). decode (a decode launch: z its split
/// partials): FP8 / FP4 through the GEMV, bf16 split as bfSplits.
pub fn linear(k: *const Kernels, s: cuda.Stream, sc: Scratch, l: weights.Linear, x: u64, x_dtype: DType, ldx: usize, m: usize, y: u64, ldy: usize, y_dtype: DType, accumulate: bool, decode: ?Z) !void {
    switch (l.fmt) {
        .fp8, .fp4 => {
            if (m > sc.rows or l.k > sc.max_k) return error.ScratchTooSmall;
            try quant(k, s, x, x_dtype, ldx, m, l.k, sc.q, l.k, sc.s, l.k / 32);
            if (decode != null) return gemv(k, s, l, sc.q, l.k, sc.s, l.k / 32, m, y, ldy, y_dtype, accumulate, decode);
            try gemm(k, s, l, sc.q, l.k, sc.s, l.k / 32, m, y, ldy, y_dtype, accumulate, sc.z);
        },
        .bf16 => {
            if (x_dtype != .bf16 or accumulate) return error.UnsupportedInput;
            try bf16Gemm(k, s, x, ldx, l.w, l.k, m, l.n, l.k, y, ldy, y_dtype, 1, false, 1, 0, 0, 0, decode);
        },
        .exl3 => return error.NotANativeLinear,
    }
}

/// A native linear's DecArgs on m rows: FP8 / FP4 of quantized rows (q, sc; row strides ldq bytes, lds floats), bf16
/// of bf16 rows at q (ldq elements).
fn decArgsOf(l: weights.Linear, q: u64, ldq: usize, sc: u64, lds: usize, m: usize, y: u64, ldy: usize, y_dtype: DType) DecArgs {
    return .{
        .q = q,
        .s = if (l.fmt == .bf16) 0 else sc,
        .ldq = @intCast(ldq),
        .lds = @intCast(lds),
        .w = l.w,
        .ws = l.s,
        .ldw = switch (l.fmt) {
            .fp4 => l.k / 2,
            .bf16 => l.k * 2,
            else => l.k,
        },
        .ldws = l.k / 32,
        .fmt = switch (l.fmt) {
            .fp4 => 1,
            .bf16 => 2,
            else => 0,
        },
        .M = @intCast(m),
        .N = @intCast(l.n),
        .K = @intCast(l.k),
        .y = y,
        .ldy = @intCast(ldy),
        .y_dtype = @intFromEnum(y_dtype),
    };
}

fn decKind(l: weights.Linear) ?u2 {
    return switch (l.fmt) {
        .fp8 => 0,
        .fp4 => 1,
        .bf16 => 2,
        .exl3 => null,
    };
}

/// Decode linears on m rows (mx.linear decode for each, bf16 x): the FP8 / FP4 ones sharing an input quantized once
/// (consecutive linears of the same x, row stride and K read the same codes: act_quant's are a function of the row),
/// and consecutive linears of one format and one input on the tensor-core kernels in one launch (each as alone).
/// Past decode_rows rows (the drafter's pass of 13 or more streams) in slabs of decode_rows rows, each slab as a call of
/// its own rows (codes and scales are functions of the row; the decode kernels' and GEMV's row bits do not depend on the
/// row count), never the prompt GEMMs.
pub fn linearsDecode(k: *const Kernels, s: cuda.Stream, sc: Scratch, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, m: usize, outs: []const u64, ldos: []const usize, types: []const DType, z: Z) !void {
    if (m > decode_rows) {
        if (ls.len > 8) return error.UnsupportedShape;
        var r0: usize = 0;
        while (r0 < m) : (r0 += decode_rows) {
            var sx: [8]u64 = undefined;
            var so: [8]u64 = undefined;
            for (0..ls.len) |i| {
                sx[i] = xs[i] + r0 * ldxs[i] * 2;
                so[i] = outs[i] + r0 * ldos[i] * @as(usize, if (types[i] == .f32) 4 else 2);
            }
            try linearsDecode(k, s, sc, ls, sx[0..ls.len], ldxs, @min(decode_rows, m - r0), so[0..ls.len], ldos, types, z);
        }
        return;
    }
    var i: usize = 0;
    while (i < ls.len) {
        const l = ls[i];
        const kind = decKind(l) orelse return error.NotANativeLinear;
        if (!decTakes(k, kind, l.n, l.k)) {
            try linear(k, s, sc, l, xs[i], .bf16, ldxs[i], m, outs[i], ldos[i], types[i], false, z);
            i += 1;
            continue;
        }
        // the run of linears on this input that the launch takes
        var j = i + 1;
        while (j < ls.len and j - i < 4) : (j += 1) {
            const o = ls[j];
            const ok = decKind(o) == kind and decTakes(k, kind, o.n, o.k) and o.k == l.k and xs[j] == xs[i] and ldxs[j] == ldxs[i];
            if (!ok) break;
            // (an earlier output over this input: not one launch)
            var clash = false;
            for (i..j) |p| clash = clash or outs[p] == xs[i];
            if (clash) break;
        }
        var as: [4]DecArgs = undefined;
        if (kind == 2) {
            for (i..j) |p| as[p - i] = decArgsOf(ls[p], xs[p], ldxs[p], 0, 0, m, outs[p], ldos[p], types[p]);
        } else {
            if (m > sc.rows or l.k > sc.max_k) return error.ScratchTooSmall;
            try quantDec(k, s, xs[i], .bf16, ldxs[i], m, l.k, sc.q, l.k, sc.s, l.k / 32);
            for (i..j) |p| as[p - i] = decArgsOf(ls[p], sc.q, l.k, sc.s, l.k / 32, m, outs[p], ldos[p], types[p]);
        }
        try decDenseMulti(k, s, kind, as[0 .. j - i], z, kind != 2);
        i = j;
    }
}

/// wo_a's groups (Attention.forward: act_quant(o, inplace) of the heads' rows, then the block-diagonal bf16 einsum):
/// o [m, groups * gk] bf16 (row stride ldo) quant-dequantized into sc.fake, each group's columns through its bf16
/// weight into u's column block (u row stride ldu, bf16).
pub fn woA(k: *const Kernels, s: cuda.Stream, sc: Scratch, wo: []const weights.Linear, o: u64, ldo: usize, m: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = wo[0].k;
    const kk = gk * wo.len;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    try fake(k, s, o, .bf16, ldo, m, kk, sc.fake, kk);
    var col: usize = 0;
    for (wo, 0..) |l, g| {
        if (l.fmt != .bf16 or l.k != gk) return error.NotANativeLinear;
        try bf16Gemm(k, s, sc.fake + g * gk * 2, kk, l.w, l.k, m, l.n, l.k, u + col * 2, ldu, .bf16, 1, false, 1, 0, 0, 0, z);
        col += l.n;
    }
}

/// wo_a of a decode launch on its FP8 rows (weights.Layer.wo_a8, every group stacked): o [m, groups * gk] bf16 (row
/// stride ldo) quantized once (act_quant: the codes and scales the reference's quant-dequant rows hold), every group's
/// columns in one launch, group g's output columns reading its input columns, into u [m, groups * o_lora] bf16 (row
/// stride ldu). The same products as woA's bf16 einsum of the quant-dequant rows (both exact), summed in fp32.
pub fn woA8(k: *const Kernels, s: cuda.Stream, sc: Scratch, w8: weights.Linear, groups: usize, o: u64, ldo: usize, m: usize, u: u64, ldu: usize, z: ?Z) !void {
    const gk: usize = w8.k;
    const kk = gk * groups;
    if (m > sc.rows or kk > sc.max_k) return error.ScratchTooSmall;
    if (!decTakes(k, 0, w8.n, gk) or (w8.n / groups) % 16 != 0) return error.UnsupportedShape;
    try quantDec(k, s, o, .bf16, ldo, m, kk, sc.q, kk, sc.s, kk / 32);
    try decDenseMulti(k, s, 0, &.{.{
        .q = sc.q,
        .s = sc.s,
        .ldq = @intCast(kk),
        .lds = @intCast(kk / 32),
        .w = w8.w,
        .ws = w8.s,
        .ldw = @intCast(gk),
        .ldws = @intCast(gk / 32),
        .fmt = 0,
        .M = @intCast(m),
        .N = @intCast(w8.n),
        .K = @intCast(gk),
        .y = u,
        .ldy = @intCast(ldu),
        .y_dtype = @intFromEnum(DType.bf16),
        .gn = @intCast(w8.n / groups),
        .gq = @intCast(gk),
        .gs = @intCast(gk / 32),
    }}, z, true);
}

/// The experts' scratch (the routed rows' groups, their quantized inputs and intermediate, the slots' outputs) for
/// up to `rows` rows of `slots` slots, `e` experts, D inputs, I the whole intermediate a node's down reads, N outputs.
pub const ExpertScratch = struct {
    rows: usize,
    slots: usize,
    q: u64, // u8 [rows, D]
    s: u64, // f32 [rows, D / 32]
    cnt: u64, // i32 [E]
    off: u64, // i32 [E + 1]
    fill: u64, // i32 [E]
    act: u64, // i32 [E]
    nact: u64, // i32 [1]
    ent: u64, // i32 [rows * slots]
    erow: u64,
    hq: u64, // u8 [rows * slots, I]
    hs: u64, // f32 [rows * slots, I / 32]
    out: u64, // f32 [rows * slots, N]
    z: Z = .{ .ptr = 0, .floats = 0 }, // decode split partials (decodeZFloats; none: no decode launches split)
    // the prefill tiles' list (int2 [rows * slots / pf_bm + E]) and its length (i32 [1]): none, prompt chunks take the
    // tile kernels
    tiles: u64 = 0,
    ntiles: u64 = 0,
    // NVFP4 routed experts (none otherwise): each entry's codes and E4M3 scales, gate / up's and down's
    nq: u64 = 0, // u8 [rows * slots, D / 2]
    ns: u64 = 0, // u8 [rows * slots, D / 16]
    hnq: u64 = 0, // u8 [rows * slots, I / 2]
    hns: u64 = 0, // u8 [rows * slots, I / 16]

    /// Bytes of NVFP4's entry codes and scales: nq, ns, hnq, hns.
    pub fn nvSizes(rows: usize, slots: usize, d: usize, i: usize) [4]usize {
        const p = rows * slots;
        return .{ p * d / 2, p * d / 16, p * i / 2, p * i / 16 };
    }

    /// Bytes of the prefill tile list for `rows` rows of `slots` slots and `e` experts.
    pub fn tileBytes(rows: usize, slots: usize, e: usize) usize {
        return (rows * slots / pf_bm + e) * 8;
    }

    pub fn sizes(rows: usize, slots: usize, e: usize, d: usize, i: usize, n: usize) [12]usize {
        const p = rows * slots;
        return .{ rows * d, rows * d / 32 * 4, 4 * e, 4 * (e + 1), 4 * e, 4 * e, 4, 4 * p, 4 * p, p * i, p * i / 32 * 4, p * n * 4 };
    }
};

/// The routed rows grouped by expert (picks pick [r, slots]; ids past the experts are another node's), and x's rows
/// quantized: what gate / up and down read.
fn group(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, r: usize, one_cta: bool, nv: bool) !void {
    const e: usize = ex.count;
    const rs = r * sc.slots;
    if (one_cta and rs <= 4096) {
        // a decode launch: the count, prefix and places in one CTA (the same groups, entries in entry order)
        var g: cuda.Args = .{};
        g.add(pick);
        g.add(@as(c_int, @intCast(rs)));
        g.add(@as(c_int, @intCast(sc.slots)));
        g.add(@as(c_int, @intCast(e)));
        g.add(sc.off);
        g.add(sc.act);
        g.add(sc.nact);
        g.add(sc.ent);
        g.add(sc.erow);
        try cuda.launch.launch(k.dec_group, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 }, .shared = @intCast((2 * e + 1) * 4) }, s, &g);
        // x's codes last: gate / up (a programmatic dependent launch) reads the groups before it waits for them
        // (NVFP4: the entries' codes are last, tf_dsn_nv_quant; these are the shared expert's alone)
        if (nv) return quant(k, s, x, .bf16, ldx, r, ex.dims, sc.q, ex.dims, sc.s, ex.dims / 32);
        return quantDec(k, s, x, .bf16, ldx, r, ex.dims, sc.q, ex.dims, sc.s, ex.dims / 32);
    }
    try quant(k, s, x, .bf16, ldx, r, ex.dims, sc.q, ex.dims, sc.s, ex.dims / 32);
    try d.check(d.api.cuMemsetD32Async(sc.cnt, 0, e, s.handle), "cuMemsetD32Async");
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(@as(c_int, @intCast(rs)));
    a.add(@as(c_int, @intCast(e)));
    a.add(sc.cnt);
    try launch(k.count, s, .{ cdiv(rs, 256), 1, 1 }, 256, &a);
    a = .{};
    a.add(sc.cnt);
    a.add(@as(c_int, @intCast(e)));
    a.add(sc.off);
    a.add(sc.fill);
    a.add(sc.act);
    a.add(sc.nact);
    try launch(k.scan, s, .{ 1, 1, 1 }, 32, &a);
    a = .{};
    a.add(pick);
    a.add(@as(c_int, @intCast(rs)));
    a.add(@as(c_int, @intCast(sc.slots)));
    a.add(@as(c_int, @intCast(e)));
    a.add(sc.off);
    a.add(sc.fill);
    a.add(sc.ent);
    a.add(sc.erow);
    try launch(k.place, s, .{ cdiv(rs, 256), 1, 1 }, 256, &a);
}

/// NVFP4's activations: every routed entry's codes and scales of x [rows, kdim] (by_entry: x [p, kdim], its own row; else
/// x's row entry / slots) under its expert's g, tf_dsn_nv_quant.
fn nvQuant(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, pick: u64, slots: usize, by_entry: bool, p: usize, kdim: usize, g: u64, routed: usize, nq: u64, ns: u64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(ldx)));
    a.add(pick);
    a.add(@as(c_int, @intCast(slots)));
    a.add(@as(c_int, @intFromBool(by_entry)));
    a.add(@as(c_int, @intCast(p)));
    a.add(@as(c_int, @intCast(kdim)));
    a.add(g);
    a.add(@as(c_int, @intCast(routed)));
    a.add(nq);
    a.add(@as(i64, @intCast(kdim / 2)));
    a.add(ns);
    a.add(@as(i64, @intCast(kdim / 16)));
    try launch(k.nv_quant, s, .{ cdiv(p * (kdim / 16), 128), 1, 1 }, 128, &a);
}

/// The active experts' down weights (down's order) into L2 on stream `s`, up to `budget` bytes, paced at about
/// 230 GB/s: the 2D exchange between gate / up and down leaves DRAM idle.
pub fn prefetchDown(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, budget: usize) !void {
    const nx = ex.native orelse return;
    const i: usize = ex.down_k;
    const n: usize = ex.down_n;
    var a: cuda.Args = .{};
    a.add(sc.act);
    a.add(sc.nact);
    a.add(nx.w2);
    a.add(@as(c_longlong, @intCast(n * i / 2)));
    a.add(@as(c_longlong, @intCast(n * i)));
    a.add(@as(c_int, @intCast(ex.count - 1)));
    a.add(@as(c_longlong, @intCast(budget)));
    a.add(@as(c_longlong, @intFromFloat(128.0 * 8192.0 / 230.0)));
    try cuda.launch.launch(k.dec_l2pf, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// The decode tensor-core kernels' warps a tile (and matrix) of the experts' gate / up and down (fixed: the experts'
/// shapes are the layer's).
pub const dec_gu_ks = 4;
pub const dec_dn_ks = 2;

/// Gate / up of a native layer's experts on r rows: x [r, D] bf16 (row stride ldx) routed by pick / wts [r, slots]
/// (the shared expert the last id, every row's last slot), each slot's h = bf16(silu(min(g, limit)) * clamp(u) * w)
/// into the pack h as the 2D exchange carries it: every slot's main columns [r * slots, ldh] (ldh: the main blocks'
/// width), then with a rest block every slot's rest columns [r * slots, rest width].
/// split: a decode launch, the GEMV kernels (a warp a column, each row its own arithmetic: its bits at any row count);
/// prompt chunks: false, the MMA tiles.
pub fn gateUp(k: *const Kernels, s: cuda.Stream, d: *const cuda.Driver, ex: weights.Experts, sc: ExpertScratch, x: u64, ldx: usize, pick: u64, wts: u64, r: usize, limit: f32, h: u64, ldh: usize, split: bool) !void {
    const nx = ex.native orelse return error.NotNativeExperts;
    if (r > sc.rows) return error.ScratchTooSmall;
    try group(k, s, d, ex, sc, x, ldx, pick, r, split and k.dec_on, nx.nv);
    const dd: usize = ex.dims;
    if (nx.nv) {
        // the block-scaled kernels are the decode tensor-core and prefill tile ones (a prompt chunk's scratch has both)
        if (sc.nq == 0 or dd % 128 != 0 or (split and !k.dec_on) or (!split and !pfExperts(k, sc, dd, nx.tiles * 32, (nx.tiles - nx.rest_tiles) * 32))) return error.NvfpNeedsTileKernels;
        try nvQuant(k, s, x, ldx, pick, sc.slots, false, r * sc.slots, dd, nx.g13, ex.count - 1, sc.nq, sc.ns);
    }
    const a: GuArgs = .{
        .q = sc.q,
        .s = sc.s,
        .ldq = @intCast(dd),
        .lds = @intCast(dd / 32),
        .ent = sc.ent,
        .erow = sc.erow,
        .off = sc.off,
        .act = sc.act,
        .nact = sc.nact,
        .w1 = nx.w1,
        .w3 = nx.w3,
        .s1 = nx.s1,
        .s3 = nx.s3,
        .ldw = @intCast(dd / 2),
        .ldws = @intCast(dd / 32),
        .fmt = if (nx.nv) 2 else 1,
        .nq = sc.nq,
        .ns = sc.ns,
        .ldnq = @intCast(dd / 2),
        .ldns = @intCast(dd / 16),
        .a1 = nx.a1,
        .a3 = nx.a3,
        .D = @intCast(dd),
        .shared = @intCast(ex.count - 1),
        .sldw = @intCast(dd),
        .sldws = @intCast(dd / 32),
        .owner = nx.owner,
        .n_tiles = @intCast(nx.tiles),
        .rest_tiles = @intCast(nx.rest_tiles),
        .wts = wts,
        .limit = limit,
        .h = h,
        .ldh = @intCast(ldh),
        .P = @intCast(r * sc.slots),
    };
    const p = r * sc.slots;
    const ny = @min(@as(usize, ex.count), p);
    if (split) {
        var ga: cuda.Args = .{};
        if (k.dec_on and dd % 128 == 0) {
            // the decode tensor-core kernel: a tile's gate and up each on decGuSplit warps
            var b = a;
            b.splits = @intCast(dec_gu_ks);
            b.early_exit = @intFromBool(k.expert_exit);
            ga.add(b);
            const tpc = dec_warps / (2 * dec_gu_ks);
            const rg = decRg(r);
            return cuda.launch.launch(k.dec[3][rg], .{ .grid = .{ .x = @intCast(cdiv(nx.tiles * 2, tpc)), .y = @intCast(ny), .z = 1 }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_gu_cfg, @as(usize, 1) << @intCast(rg)), .pdl = k.dec_pdl }, s, &ga);
        }
        ga.add(a);
        if (k.sab) return launch(k.gu_sab, s, .{ cdiv(nx.tiles * 32, 16), ny, 1 }, 256, &ga);
        return launch(k.gu_gemv, s, .{ cdiv(nx.tiles * 32, 8), ny, 1 }, 256, &ga);
    }
    var args: cuda.Args = .{};
    args.add(a);
    const main = (nx.tiles - nx.rest_tiles) * 32;
    if (pfExperts(k, sc, dd, nx.tiles * 32, main) and dd % 128 == 0) {
        try tileList(k, s, sc);
        args.add(sc.tiles);
        args.add(sc.ntiles);
        return launchShared(if (nx.nv) k.pf_gu_nv else k.pf_gu, s, .{ nx.tiles * 32 / 128, p / pf_bm + ex.count, 1 }, pf_gu_smem, &args);
    }
    const t = @min(tiles(r), 2);
    try launch(k.gu[t], s, .{ cdiv(nx.tiles, 4), ny, 1 }, 128, &args);
}

/// Whether a prompt chunk's grouped experts take the prefill tiles: a tile list, K (`kdim`) a multiple of 128, the
/// output columns (`n`) and the rest block's start (`main`) multiples of 128, aligned rows.
fn pfExperts(k: *const Kernels, sc: ExpertScratch, kdim: usize, n: usize, main: usize) bool {
    if (k.pf_off or sc.tiles == 0) return false;
    return kdim % 128 == 0 and n % 128 == 0 and main % 128 == 0;
}

/// The prefill tiles of the current grouping (sc.off / act / nact) into sc.tiles, sc.ntiles.
fn tileList(k: *const Kernels, s: cuda.Stream, sc: ExpertScratch) !void {
    var a: cuda.Args = .{};
    a.add(sc.off);
    a.add(sc.act);
    a.add(sc.nact);
    a.add(sc.tiles);
    a.add(sc.ntiles);
    try launch(k.pf_tiles, s, .{ 1, 1, 1 }, 32, &a);
}

/// Down of a native layer's experts after gate / up (the same grouping): h [r * slots, ldh] bf16, the node's whole
/// intermediate part (down_k columns), quantized, through each expert's down part, the slots' outputs summed in expert
/// id order into out [r, down_n] fp32 (row stride ldo; accumulate: added to it).
/// split: a decode launch, the GEMV kernel (as gateUp).
pub fn down(k: *const Kernels, s: cuda.Stream, ex: weights.Experts, sc: ExpertScratch, h: u64, ldh: usize, pick: u64, r: usize, out: u64, ldo: usize, accumulate: bool, split: bool) !void {
    const nx = ex.native orelse return error.NotNativeExperts;
    const i: usize = ex.down_k;
    const n: usize = ex.down_n;
    const p = r * sc.slots;
    if (nx.nv) {
        if (sc.hnq == 0 or i % 128 != 0 or n % 16 != 0 or (split and !k.dec_on) or (!split and !pfExperts(k, sc, i, n, 0))) return error.NvfpNeedsTileKernels;
        // the shared expert's entries (every row's last slot) in FP8, in their entry rows; the routed ones' own codes last
        const last = sc.slots - 1;
        try quant(k, s, h + last * ldh * 2, .bf16, sc.slots * ldh, r, i, sc.hq + last * i, sc.slots * i, sc.hs + last * (i / 32) * 4, sc.slots * (i / 32));
        try nvQuant(k, s, h, ldh, pick, sc.slots, true, p, i, nx.g2, ex.count - 1, sc.hnq, sc.hns);
    } else if (split) try quantDec(k, s, h, .bf16, ldh, p, i, sc.hq, i, sc.hs, i / 32) else try quant(k, s, h, .bf16, ldh, p, i, sc.hq, i, sc.hs, i / 32);
    const a: DnArgs = .{
        .q = sc.hq,
        .s = sc.hs,
        .ldq = @intCast(i),
        .lds = @intCast(i / 32),
        .ent = sc.ent,
        .off = sc.off,
        .act = sc.act,
        .nact = sc.nact,
        .w2 = nx.w2,
        .s2 = nx.s2,
        .ldw = @intCast(i / 2),
        .ldws = @intCast(i / 32),
        .fmt = if (nx.nv) 2 else 1,
        .nq = sc.hnq,
        .ns = sc.hns,
        .ldnq = @intCast(i / 2),
        .ldns = @intCast(i / 16),
        .a2 = nx.a2,
        .shared = @intCast(ex.count - 1),
        .sldw = @intCast(i),
        .sldws = @intCast(i / 32),
        .I = @intCast(i),
        .N = @intCast(n),
        .round_bf16 = 0,
        .out = sc.out,
        .ldo = @intCast(n),
    };
    var args: cuda.Args = .{};
    args.add(a);
    const ny = @min(@as(usize, ex.count), p);
    if (split and k.dec_on and i % 128 == 0 and n % 16 == 0) {
        var b = a;
        b.splits = @intCast(dec_dn_ks);
        b.early_exit = @intFromBool(k.expert_exit);
        var da: cuda.Args = .{};
        da.add(b);
        const tpc = dec_warps / dec_dn_ks;
        const rg = decRg(r);
        try cuda.launch.launch(k.dec[4][rg], .{ .grid = .{ .x = @intCast(cdiv(n / 16, tpc)), .y = @intCast(ny), .z = 1 }, .block = .{ .x = 32 * dec_warps }, .shared = decShared(dec_experts_cfg, @as(usize, 1) << @intCast(rg)), .pdl = k.dec_pdl }, s, &da);
    } else if (split) {
        if (k.sab) try launch(k.dn_sab, s, .{ cdiv(n, 16), ny, 1 }, 256, &args) else try launch(k.dn_gemv, s, .{ cdiv(n, 8), ny, 1 }, 256, &args);
    } else if (pfExperts(k, sc, i, n, 0)) {
        try tileList(k, s, sc);
        args.add(sc.tiles);
        args.add(sc.ntiles);
        try launchShared(if (nx.nv) k.pf_dn_nv else k.pf_dn, s, .{ n / 128, p / pf_bm + ex.count, 1 }, pf_dn_smem, &args);
    } else {
        const t = @min(tiles(r), 2);
        try launch(k.dn[t], s, .{ cdiv(n / 32, 4), ny, 1 }, 128, &args);
    }
    var ca: cuda.Args = .{};
    ca.add(sc.out);
    ca.add(@as(i64, @intCast(n)));
    ca.add(pick);
    ca.add(@as(c_int, @intCast(r)));
    ca.add(@as(c_int, @intCast(sc.slots)));
    ca.add(@as(c_int, @intCast(ex.count)));
    ca.add(@as(c_int, @intCast(n)));
    ca.add(out);
    ca.add(@as(i64, @intCast(ldo)));
    ca.add(@as(c_int, @intFromBool(accumulate)));
    try launch(k.combine, s, .{ cdiv(r * n, 256), 1, 1 }, 256, &ca);
}

test "splits depend on the shape alone and fill the GPU" {
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 1280)); // wq_a: 10 CTAs, 10 K blocks a split
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 512)); // wkv
    try std.testing.expectEqual(@as(usize, 2), splits(1280, 8192)); // a 2D node's wq_b
    try std.testing.expectEqual(@as(usize, 1), splits(3072, 12800)); // a 2D node's Engram wkv
    try std.testing.expectEqual(@as(usize, 4), splits(1280, 4096)); // the indexer's wq_b
    try std.testing.expectEqual(@as(usize, 16), splits(5120, 640)); // a 2D node's expert gate / up pack
    try std.testing.expectEqual(@as(usize, 16), bfSplits(5120, 384)); // the router
    try std.testing.expectEqual(@as(usize, 16), bfSplits(4096, 1024)); // a wo_a group
}
