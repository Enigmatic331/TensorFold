//! One rank's weights on the GPU in the served engine's layouts (weights.py, exl3/linear.py, exl3/experts.py): EXL3
//! linears in strips, each layer's experts in one trellis buffer with pointer, width and scale tables, the rest as the
//! loader keeps it. Every device tensor is listed by the Python engine's name for digests against its weights.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const rank_cache = @import("rank_cache.zig");
const checkpoint = @import("checkpoint.zig");

/// A linear's storage: EXL3 (the strips below), or DeepSeek's own (Format.native: native.zig computes them): FP8 E4M3
/// [N, K] with UE8M0 scales [N / 32, K / 32], MXFP4 [N, K / 2] with UE8M0 scales [N, K / 32], or bf16 [N, K].
pub const Fmt = enum(u8) { exl3, fp8, fp4, bf16 };

/// An EXL3 linear in the strips layout: words int32 [N/128, K/16, 8, 8 * bits], fp16 suh [K] and svh [N].
/// plan_n: the N its decode plan is made for, when not n: a 2D node's column part of a TP2 slice keeps that slice's
/// plan (linear.py pinned plans), so its columns sum in the slice's K order.
/// A native linear (fmt not exl3): its weight `w` and scales `s` (none for bf16); `words` is `w` (a linear's identity:
/// its counters, the prefetch's first byte).
pub const Linear = struct { words: u64, suh: u64 = 0, svh: u64 = 0, k: u32, n: u32, k2: u32 = 0, plan_n: u32 = 0, fmt: Fmt = .exl3, w: u64 = 0, s: u64 = 0 };

/// A native layer's experts (Format.native), the routed ones MXFP4 and the shared one (the last id) FP8: each expert's
/// gate and up rows as this node computes them, packed (its main blocks, then its rest block when its rest is this
/// pair's: the order of the 2D exchanges' packs), and its down part [down_n, down_k]. Pointer tables u64 [E] on the
/// device, the strides of each format.
pub const NativeExperts = struct {
    w1: u64, // u64 [E]: gate rows [gu rows, D] (MXFP4: D / 2 bytes a row; the shared expert FP8: D)
    w3: u64,
    s1: u64, // u64 [E]: their scales (MXFP4 [rows, D / 32]; FP8 [rows / 32, D / 32])
    s3: u64,
    w2: u64, // u64 [E]: down [down_n, down_k] (/ 2 for MXFP4)
    s2: u64,
    owner: u64 = 0, // int32 [E]: 1 when this node computes e's rest block (the pack's last tile), else 0; 0: no rest
    tiles: u32, // the pack's 32-column tiles (main, plus one rest block's 4 when there is a rest)
    rest_tiles: u32 = 0,
    // NVFP4 routed experts (s1 / s3 / s2: one E4M3 byte a block of 32): fp32 [E] each, the shared expert's unused
    nv: bool = false,
    g13: u64 = 0, // 1 / input_scale of gate and up (the same input)
    g2: u64 = 0, // 1 / input_scale of down
    a1: u64 = 0, // input_scale * weight_scale_2 of gate
    a3: u64 = 0,
    a2: u64 = 0,
};

/// A layer's routed experts and its shared expert (the last entry), as exl3_experts.prepare lays them out. On a 2D node
/// (exl3/experts2d.py prepare) gate / up hold its intermediate blocks (width of the half's down_k) and down its
/// output columns (down_n of dims) over the half's whole intermediate.
pub const Experts = struct {
    trellis: u64, // gate trellises of every expert, then up, then down (pack_trellises)
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]: half-bits a value
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]: down's input signs on the gate / up blocks here
    svh_d: u64, // fp16 [E, down_n]
    count: u32,
    dims: u32,
    width: u32,
    k2_gu: [2]u32,
    k2_d: [2]u32,
    down_k: u32, // down's K: the rank's whole intermediate (width but on a 2D node)
    down_n: u32, // down's output columns here (dims but on a 2D node)
    shared_gate: [2]u64 = .{ 0, 0 }, // the shared expert's gate trellis: its first byte and length (the paced L2 prefetch)
    shared_up: [2]u64 = .{ 0, 0 }, // its up and down trellises the same way (the served "moe" fork reads all three)
    shared_down: [2]u64 = .{ 0, 0 },
    rest: ?Rest = null, // TF_DS_2D_GU=parity: the ninth gate / up block of the experts whose parity is this pair's
    native: ?NativeExperts = null, // Format.native: the experts' own tensors (the EXL3 fields then unused)
};

/// experts2d.Rest2D: a parity split's rest columns (one 128-block of the rank's half, after both pairs' main blocks) of
/// the experts this node computes them for; every table [E] or [E, width] with zeros (no trellis) elsewhere.
pub const Rest = struct {
    gate_ptr: u64, // int64 [E]: trellis [D/16, width/16, *], 0 for an expert another node computes
    up_ptr: u64,
    gate_k2: u64, // int32 [E]
    up_k2: u64,
    svh_g: u64, // fp16 [E, width]
    svh_u: u64,
    suh_d: u64, // fp16 [E, width]: down's input signs on the rest block
    remap: u64, // int32 [E + 1]: e -> e when this node computes e's rest, else E (skipped by the grouping)
    owner: u64, // int32 [E + 1]: the pair computing e's rest (E -> 0)
    width: u32,
    k2_gu: [2]u32,
    mine: u32, // this node's pair
};

pub const Layer = struct {
    idx: u32,
    ratio: u8,
    hc_attn: [3]u64 = .{ 0, 0, 0 }, // fn [24, hc * d] f32, scale [3], base [24]
    hc_ffn: [3]u64 = .{ 0, 0, 0 },
    attn_norm: u64 = 0,
    ffn_norm: u64 = 0,
    wq_a: Linear = undefined,
    q_norm: u64 = 0,
    wq_b: Linear = undefined, // this rank's heads' columns
    wkv: Linear = undefined,
    kv_norm: u64 = 0,
    sink: u64 = 0, // this rank's heads, f32
    wo_a: [4]Linear = undefined, // this rank's groups
    groups: u32 = 4, // wo_a's entries loaded: the rank's groups, its pair's half of them on a 2D node
    // Format.native: wo_a's groups as the checkpoint stores them (FP8, the "exact twin" of the bf16 rows: the same
    // values), every group's rows in one tensor [groups * o_lora, K] with its scales [groups * o_lora / 32, K / 32];
    // the decode rounds' wo_a (mx.woA8)
    wo_a8: ?Linear = null,
    wo_b: Linear = undefined, // this rank's groups' input rows (a 2D node: its pair's output columns of them)
    comp_wkv: ?Linear = null,
    comp_wgate: ?Linear = null,
    comp_norm: u64 = 0,
    idx_wq_b: ?Linear = null,
    idx_proj: u64 = 0, // [idx_heads, d] f32
    idx_proj_h: u64 = 0, // the same in fp16 (Model.__init__)
    idx_wk: ?Linear = null,
    idx_k_norm: u64 = 0,
    gate_w: u64 = 0, // [E, d] f16 (Format.native: bf16)
    gate_b: u64 = 0, // [E] f32
    gate_b_vl: u64 = 0, // [E] f32: the routing bias of image-span tokens (the checkpoint's extra gate_bias_vl)
    experts: Experts = undefined,
    engram_wkv: ?Linear = null, // this rank's hash columns' input rows
    engram_qk: u64 = 0, // q_weight * k_weight [hc, d] f32
};

pub const DSpark = struct {
    blocks: []Layer,
    main_proj: Linear,
    main_norm: u64,
    norm: u64,
    markov_embed: u64,
    markov_head: u64,
    conf: u64,
    // Format.native: the drafter's own FP8 copy of this node's head columns (TF_DS_DRAFT_HEAD unset or fp8; bf16:
    // none, the drafter reads the backbone's bf16 head). Drafts are only proposals the target verifies, so the
    // rounding moves no output token; it halves the bytes the drafter's head reads every pass.
    head8: ?Linear = null,
};

/// TF_DS_DRAFT_HEAD: the drafter's head (fp8 unless "bf16"; "fp4": MXFP4, a quarter of the bf16 bytes).
pub fn draftHeadFp8() bool {
    const v = std.c.getenv("TF_DS_DRAFT_HEAD") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "bf16");
}

fn draftHeadFp4() bool {
    const v = std.c.getenv("TF_DS_DRAFT_HEAD") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "fp4");
}

/// MXFP4 (E2M1 pairs, low nibble first, a power-of-two UE8M0 scale per 32 along K: the checkpoint's expert format) of
/// bf16 rows [n, k]: w [n, k / 2], sc [n, k / 32].
pub fn quantFp4(src: []const u8, n: usize, k: usize, w: []u8, sc: []u8) void {
    const grid = [_]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
    for (0..n) |i| for (0..k / 32) |bj| {
        var amax: f32 = 0;
        for (0..32) |j| amax = @max(amax, @abs(bf16At(src, i * k + bj * 32 + j)));
        const t = @max(amax / 6.0, 1e-30);
        const tb: u32 = @bitCast(t);
        var ex: i32 = @as(i32, @intCast((tb >> 23) & 0xff)) - 127;
        if (tb & 0x7fffff != 0) ex += 1;
        ex = std.math.clamp(ex, -127, 127);
        sc[i * (k / 32) + bj] = @intCast(ex + 127);
        const inv = std.math.ldexp(@as(f32, 1.0), -ex);
        for (0..16) |jj| {
            var byte: u8 = 0;
            for (0..2) |h| {
                const v = bf16At(src, i * k + bj * 32 + 2 * jj + h) * inv;
                const a = @min(@abs(v), 6.0);
                var best: u8 = 0;
                var bd: f32 = std.math.inf(f32);
                for (grid, 0..) |g, gi| {
                    const dd = @abs(a - g);
                    if (dd < bd) {
                        bd = dd;
                        best = @intCast(gi);
                    }
                }
                if (v < 0 and best != 0) best |= 8;
                byte |= best << @intCast(4 * h);
            }
            w[i * (k / 2) + bj * 16 + jj] = byte;
        }
    };
}

/// TF_DS_DRAFT_WOA: the drafter's wo_a (fp8 unless "bf16").
pub fn draftWoAFp8() bool {
    const v = std.c.getenv("TF_DS_DRAFT_WOA") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "bf16");
}

/// An f32 (|v| <= 448) as E4M3, rounded to nearest even (subnormals below 2^-6).
pub fn toE4m3(v: f32) u8 {
    const sign: u8 = if (v < 0) 0x80 else 0;
    const a = @abs(v);
    if (a == 0) return sign;
    const bits: u32 = @bitCast(a);
    var e: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - 127;
    if (e < -6) {
        // subnormal: a multiple of 2^-9
        const r = rne(a * 512.0);
        if (r >= 8) return sign | 0x08;
        return sign | @as(u8, @intCast(r));
    }
    var r = rne(std.math.ldexp(a, 3 - e)); // in [8, 16]
    if (r >= 16) {
        r = 8;
        e += 1;
    }
    const code: u32 = @as(u32, @intCast(e + 7)) << 3 | (r - 8);
    return sign | @as(u8, @intCast(@min(code, 0x7e)));
}

fn rne(x: f32) u32 {
    const f = @floor(x);
    const d = x - f;
    var r: u32 = @intFromFloat(f);
    if (d > 0.5 or (d == 0.5 and r % 2 == 1)) r += 1;
    return r;
}

/// FP8 (E4M3, 32 x 32 blocks with power-of-two UE8M0 scales, the checkpoint's own format) of bf16 rows [n, k].
pub fn quantFp8(src: []const u8, n: usize, k: usize, w: []u8, sc: []u8) void {
    const kb = k / 32;
    for (0..n / 32) |bi| for (0..kb) |bj| {
        var amax: f32 = 0;
        for (0..32) |i| for (0..32) |j| {
            const x = bf16At(src, (bi * 32 + i) * k + bj * 32 + j);
            amax = @max(amax, @abs(x));
        };
        // scale: the least power of two with amax / scale <= 448
        const t = @max(amax / 448.0, 1e-30);
        const tb: u32 = @bitCast(t);
        var ex: i32 = @as(i32, @intCast((tb >> 23) & 0xff)) - 127;
        if (tb & 0x7fffff != 0) ex += 1;
        ex = std.math.clamp(ex, -127, 127);
        sc[bi * kb + bj] = @intCast(ex + 127);
        const inv = std.math.ldexp(@as(f32, 1.0), -ex);
        for (0..32) |i| for (0..32) |j| {
            const at = (bi * 32 + i) * k + bj * 32 + j;
            w[at] = toE4m3(std.math.clamp(bf16At(src, at) * inv, -448.0, 448.0));
        };
    };
}

fn bf16At(b: []const u8, i: usize) f32 {
    const h = std.mem.readInt(u16, b[2 * i ..][0..2], .little);
    return @bitCast(@as(u32, h) << 16);
}

test "e4m3 encode round trips and rounds to nearest even" {
    var b: u16 = 0;
    while (b < 256) : (b += 1) {
        const x: u8 = @intCast(b);
        if (x & 0x7f == 0x7f or x == 0x80) continue; // NaN, -0
        try std.testing.expectEqual(x, toE4m3(e4m3(x)));
    }
    try std.testing.expectEqual(@as(u8, 0x38), toE4m3(1.0625)); // between 1 and 1.125: ties to even (1)
    try std.testing.expectEqual(@as(u8, 0x3a), toE4m3(1.1875)); // between 1.125 and 1.25: ties to even (1.25)
    try std.testing.expectEqual(@as(u8, 0x7e), toE4m3(448.0));
}

/// A device tensor by the Python engine's dotted name.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    embed: u64 = 0, // [V, d] bf16
    norm: u64 = 0,
    head: Linear = undefined, // this rank's vocabulary columns
    vocab_lo: usize = 0,
    vocab_hi: usize = 0,
    layers: []Layer = &.{},
    dspark: ?DSpark = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        if (w.dspark) |d| w.gpa.free(d.blocks);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

/// Where entries come from: the lane's rank cache file, or the checkpoint read by the split.
pub const Source = union(enum) {
    cache: struct { file: std.Io.File, index: *const rank_cache.Index, io: std.Io, pf: ?*Prefetch = null },
    shards: *const checkpoint.Shards,

    pub fn size(s: Source, key: []const u8) !usize {
        return switch (s) {
            .cache => |c| (c.index.get(key) orelse return error.MissingEntry).bytes,
            .shards => |sh| (try checkpoint.describe(sh, key)).bytes,
        };
    }

    pub fn read(s: Source, key: []const u8, out: []u8) !void {
        switch (s) {
            .cache => |c| {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.bytes != out.len) return error.BadSize;
                if (try c.file.readPositionalAll(c.io, out, e.offset) != out.len) return error.ShortRead;
                if (c.pf) |pf| _ = pf.pos.fetchMax(e.offset + e.bytes, .monotonic);
            },
            .shards => |sh| try checkpoint.read(sh, key, out),
        }
    }
};

/// Trellis int16 [kt, nt, w16] as int32 words [nt / 8, kt, 8, w16 / 2] (linear.py strips): each 128-column block's
/// tiles in k order. `tile` is one tile's bytes (w16 * 2).
pub fn strips(src: []const u8, out: []u8, kt: usize, nt: usize, tile: usize) void {
    for (0..nt / 8) |j| for (0..kt) |i| {
        const from = (i * nt + j * 8) * tile;
        const to = (j * kt + i) * 8 * tile;
        @memcpy(out[to..][0 .. 8 * tile], src[from..][0 .. 8 * tile]);
    };
}

/// The rank cache read ahead of the loader (TF_DS_LOAD_PREFETCH=0 turns it off). The file
/// holds the entries in the loader's order and the loader reads them one at a time, so every read waited on the disk:
/// ~1 GB/s from an NVMe that reads 9 GB/s in parallel, 83 s for a node's 81 GB. A few threads pread the file in chunks
/// up to `lead` bytes past the loader's furthest read, so its reads come from the page cache (27 s). The threads'
/// bytes are dropped: the loader's own reads, and so the weights, are unchanged. A bounded lead matters: with the
/// whole file cached first the load took 48 s, the page cache and the landing weights competing for the same memory.
const Prefetch = struct {
    io: std.Io,
    fd: std.posix.fd_t,
    size: u64,
    pos: std.atomic.Value(u64) = .init(0), // the end of the loader's furthest read
    next: std.atomic.Value(u64) = .init(0), // the next chunk to read ahead
    stop: std.atomic.Value(bool) = .init(false),
    threads: [threads_n]std.Thread = undefined,
    started: usize = 0,

    const threads_n = 8;
    const chunk: u64 = 32 << 20;
    const lead: u64 = 4 << 30;

    fn on() bool {
        const v = std.c.getenv("TF_DS_LOAD_PREFETCH") orelse return true;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    fn start(p: *Prefetch) void {
        for (&p.threads) |*th| {
            th.* = std.Thread.spawn(.{}, run, .{p}) catch break;
            p.started += 1;
        }
    }

    fn finish(p: *Prefetch) void {
        p.stop.store(true, .monotonic);
        for (p.threads[0..p.started]) |th| th.join();
        p.started = 0;
    }

    fn run(p: *Prefetch) void {
        const buf = std.heap.page_allocator.alloc(u8, chunk) catch return;
        defer std.heap.page_allocator.free(buf);
        while (!p.stop.load(.monotonic)) {
            const at = p.next.fetchAdd(chunk, .monotonic);
            if (at >= p.size) return;
            while (at > p.pos.load(.monotonic) + lead) {
                if (p.stop.load(.monotonic)) return;
                std.Io.sleep(p.io, .fromMilliseconds(2), .awake) catch {};
            }
            const n: usize = @intCast(@min(chunk, p.size - at));
            var done: usize = 0;
            while (done < n) {
                const r = std.c.pread(p.fd, buf.ptr + done, n - done, @intCast(at + done));
                if (r < 0) {
                    if (std.c.errno(r) == .INTR) continue;
                    return; // the loader reads it itself
                }
                if (r == 0) return;
                done += @intCast(r);
            }
        }
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    src: Source,
    w: *Weights,
    host: std.ArrayList(u8) = .empty,
    aux: std.ArrayList(u8) = .empty,
    wo_a_fp8: bool = false, // blockNative: wo_a's groups as their FP8 rows (the drafter's blocks), not bf16

    fn upload(L: *Loader, name: []const u8, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.alloc(L.d, bytes.len);
        errdefer b.free();
        try b.upload(0, bytes);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = b.ptr, .len = bytes.len });
        L.w.bytes += bytes.len;
        return b.ptr;
    }

    fn entry(L: *Loader, key: []const u8) ![]u8 {
        try L.host.resize(L.gpa, try L.src.size(key));
        try L.src.read(key, L.host.items);
        return L.host.items;
    }

    /// plain(name, dtype).cuda(): the entry as it is.
    fn plain(L: *Loader, tensor: []const u8, dtype: ?rank_cache.DType) !u64 {
        var kb: [256]u8 = undefined;
        return L.upload(tensor, try L.entry(try plan.plainKey(&kb, tensor, dtype)));
    }

    /// linear(prefix, cols, rows): Exl3Linear.from_tensors in strips.
    fn linear(L: *Loader, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const tr = try L.entry(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const t = try L.shape(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const kt = t[0];
        const nt = t[1];
        const w16 = t[2];
        if (w16 % 8 != 0 or nt % 8 != 0) return error.UnexpectedTensor;
        try L.aux.resize(L.gpa, tr.len);
        strips(tr, L.aux.items, kt, nt, w16 * 2);
        const words = try L.upload(try std.fmt.bufPrint(&nb, "{s}.words", .{prefix}), L.aux.items);
        const suh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "suh")));
        const svh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "svh")));
        return .{ .words = words, .suh = suh, .svh = svh, .k = @intCast(kt * 16), .n = @intCast(nt * 16), .k2 = @intCast(w16 / 8) };
    }

    /// A native part's [rows, cols] (its entry's shape).
    fn shape2(L: *Loader, key: []const u8) ![2]usize {
        return switch (L.src) {
            .cache => |c| blk: {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.rank != 2) return error.UnexpectedTensor;
                break :blk .{ e.shape[0], e.shape[1] };
            },
            .shards => |sh| blk: {
                const s = try checkpoint.describe(sh, key);
                if (s.rank != 2) return error.UnexpectedTensor;
                break :blk .{ s.shape[0], s.shape[1] };
            },
        };
    }

    /// A native linear (plan.Native) of prefix's [cols, rows] part: its weight and scales uploaded as stored.
    fn native(L: *Loader, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize, kind: plan.Native) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const wpart: []const u8 = switch (kind) {
            .fp8 => "w8",
            .fp4 => "w4",
            .nv4 => return error.UnexpectedTensor, // the routed experts' alone (expertsNative)
            .bf16 => "bf",
        };
        const wkey = try plan.partKey(&kb, prefix, cols, rows, wpart);
        const sh = try L.shape2(wkey);
        const w = try L.upload(try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}), try L.entry(wkey));
        var s: u64 = 0;
        if (kind != .bf16) {
            const skey = try plan.partKey(&kb, prefix, cols, rows, if (kind == .fp8) "s8" else "s4");
            s = try L.upload(try std.fmt.bufPrint(&nb, "{s}.scale", .{prefix}), try L.entry(skey));
        }
        const k = if (kind == .fp4) sh[1] * 2 else sh[1];
        if (k % 32 != 0 or sh[0] % 32 != 0) return error.UnexpectedTensor;
        return .{ .words = w, .w = w, .s = s, .k = @intCast(k), .n = @intCast(sh[0]), .fmt = switch (kind) {
            .fp8 => .fp8,
            .fp4 => .fp4,
            .nv4 => unreachable,
            .bf16 => .bf16,
        } };
    }

    /// The drafter's FP8 copy of the head's columns `cols` (quantFp8 of the bf16 rows on the host).
    fn headFp8(L: *Loader, cols: [2]usize) !Linear {
        var kb: [256]u8 = undefined;
        const wkey = try plan.partKey(&kb, "head", cols, null, "bf");
        const sh = try L.shape2(wkey);
        const n = sh[0];
        const k = sh[1];
        if (n % 32 != 0 or k % 32 != 0) return error.UnexpectedTensor;
        const src = try L.entry(wkey);
        if (draftHeadFp4()) {
            const q4 = try L.gpa.alloc(u8, n * k / 2);
            defer L.gpa.free(q4);
            const s4 = try L.gpa.alloc(u8, n * (k / 32));
            defer L.gpa.free(s4);
            quantFp4(src, n, k, q4, s4);
            const w4 = try L.upload("dspark.head4.weight", q4);
            const ss4 = try L.upload("dspark.head4.scale", s4);
            return .{ .words = w4, .w = w4, .s = ss4, .k = @intCast(k), .n = @intCast(n), .fmt = .fp4 };
        }
        const q = try L.gpa.alloc(u8, n * k);
        defer L.gpa.free(q);
        const sc = try L.gpa.alloc(u8, (n / 32) * (k / 32));
        defer L.gpa.free(sc);
        quantFp8(src, n, k, q, sc);
        const w = try L.upload("dspark.head8.weight", q);
        const s = try L.upload("dspark.head8.scale", sc);
        return .{ .words = w, .w = w, .s = s, .k = @intCast(k), .n = @intCast(n), .fmt = .fp8 };
    }

    /// wo_a's group `g` (Format.native): its FP8 rows dequantized to bf16 on the host (convert.py: weight * scale, then
    /// bfloat16; exact, an E4M3 value times a power of two).
    fn woANative(L: *Loader, prefix: []const u8, g: usize, o_lora: usize) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const cols: [2]usize = .{ g * o_lora, (g + 1) * o_lora };
        const wkey = try plan.partKey(&kb, prefix, cols, null, "w8");
        const sh = try L.shape2(wkey);
        const n = sh[0];
        const k = sh[1];
        var sb: [256]u8 = undefined; // (its own buffer: wkey lives in kb)
        const sc = try L.gpa.dupe(u8, try L.entry(try plan.partKey(&sb, prefix, cols, null, "s8")));
        defer L.gpa.free(sc);
        const w = try L.entry(wkey);
        try L.aux.resize(L.gpa, n * k * 2);
        for (0..n) |i| for (0..k) |j| {
            const scale: f32 = @bitCast(@as(u32, sc[(i / 32) * (k / 32) + j / 32]) << 23);
            const v = e4m3(w[i * k + j]) * scale;
            std.mem.writeInt(u16, L.aux.items[(i * k + j) * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
        };
        const p = try L.upload(try std.fmt.bufPrint(&nb, "{s}.weight.{d}", .{ prefix, g }), L.aux.items);
        return .{ .words = p, .w = p, .k = @intCast(k), .n = @intCast(n), .fmt = .bf16 };
    }

    /// wo_a's groups ga[0] .. ga[0] + ga[1] - 1 (Format.native) as stored, FP8 rows and their UE8M0 scales, the groups
    /// stacked in one tensor each (a group's rows are whole 32-row scale blocks: o_lora % 32 == 0).
    fn woA8(L: *Loader, prefix: []const u8, ga: [2]usize, o_lora: usize) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        var wbytes: std.ArrayList(u8) = .empty;
        defer wbytes.deinit(L.gpa);
        var sbytes: std.ArrayList(u8) = .empty;
        defer sbytes.deinit(L.gpa);
        var k: usize = 0;
        for (ga[0]..ga[0] + ga[1]) |g| {
            const cols: [2]usize = .{ g * o_lora, (g + 1) * o_lora };
            const wkey = try plan.partKey(&kb, prefix, cols, null, "w8");
            const sh = try L.shape2(wkey);
            if (sh[0] != o_lora or o_lora % 32 != 0 or (k != 0 and sh[1] != k)) return error.UnexpectedTensor;
            k = sh[1];
            try wbytes.appendSlice(L.gpa, try L.entry(wkey));
            try sbytes.appendSlice(L.gpa, try L.entry(try plan.partKey(&kb, prefix, cols, null, "s8")));
        }
        const w = try L.upload(try std.fmt.bufPrint(&nb, "{s}.weight.fp8", .{prefix}), wbytes.items);
        const sc = try L.upload(try std.fmt.bufPrint(&nb, "{s}.scale.fp8", .{prefix}), sbytes.items);
        return .{ .words = w, .w = w, .s = sc, .k = @intCast(k), .n = @intCast(ga[1] * o_lora), .fmt = .fp8 };
    }

    /// Format.native experts of block p: every expert's gate / up pack (main columns, then its rest when this node
    /// computes it) and its down part, in one buffer, with pointer tables over the experts (routed MXFP4, shared FP8).
    fn expertsNative(L: *Loader, p: []const u8, n: usize, half: [2]usize, gu: [2]usize, dcols: ?[2]usize, d: usize, rest: ?[2]usize, pair: ?u32, nv: bool) !Experts {
        const e_count = n + 1;
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        var tb: [256]u8 = undefined;
        const width = gu[1] - gu[0];
        const ir = if (rest) |rc| rc[1] - rc[0] else 0;
        const down_k = half[1] - half[0];
        const down_n = if (dcols) |dc| dc[1] - dc[0] else d;
        const owner = try L.gpa.alloc(i32, e_count);
        defer L.gpa.free(owner);
        for (0..e_count) |e| owner[e] = if (rest != null and plan.restPair(e, n) == pair.?) 1 else 0;
        // one expert's six tensors (w1, w3, s1, s3, w2, s2): a gate / up tensor is its main piece, then its rest piece
        // when this node computes the rest (one pack, no gap); every tensor starts 256-byte aligned
        const Piece = struct { proj: []const u8, part: []const u8, cols: ?[2]usize, rows: ?[2]usize };
        const pieces = struct {
            fn of(out: *[2]Piece, j: usize, fp4: bool, nv4: bool, g: [2]usize, r: ?[2]usize, own: bool, dc: ?[2]usize, h: [2]usize) []Piece {
                const proj: []const u8 = switch (j) {
                    0, 2 => "w1",
                    1, 3 => "w3",
                    else => "w2",
                };
                const scale = j == 2 or j == 3 or j == 5;
                const part: []const u8 = if (fp4) (if (scale) (if (nv4) "t4" else "s4") else "w4") else (if (scale) "s8" else "w8");
                if (j >= 4) {
                    out[0] = .{ .proj = proj, .part = part, .cols = dc, .rows = h };
                    return out[0..1];
                }
                out[0] = .{ .proj = proj, .part = part, .cols = g, .rows = null };
                if (!own) return out[0..1];
                out[1] = .{ .proj = proj, .part = part, .cols = r.?, .rows = null };
                return out[0..2];
            }
        }.of;
        const name = struct {
            fn of(buf: []u8, pp: []const u8, e: usize, nn: usize, proj: []const u8) ![]const u8 {
                return if (e < nn) std.fmt.bufPrint(buf, "{s}.ffn.experts.{d}.{s}", .{ pp, e, proj }) else std.fmt.bufPrint(buf, "{s}.ffn.shared_experts.{s}", .{ pp, proj });
            }
        }.of;
        var pbuf: [2]Piece = undefined;
        var total: usize = 0;
        for (0..e_count) |e| for (0..6) |j| {
            var len_j: usize = 0;
            for (pieces(&pbuf, j, e < n, nv, gu, rest, owner[e] == 1, dcols, half)) |pc| {
                len_j += try L.src.size(try plan.partKey(&kb, try name(&nb, p, e, n, pc.proj), pc.cols, pc.rows, pc.part));
            }
            total += std.mem.alignForward(usize, len_j, 256);
        };
        var buf = try cuda.DeviceBuffer.alloc(L.d, total);
        errdefer buf.free();
        const tabs = try L.gpa.alloc(u64, 6 * e_count); // [j][e]: w1, w3, s1, s3, w2, s2
        defer L.gpa.free(tabs);
        var at: usize = 0;
        for (0..e_count) |e| for (0..6) |j| {
            tabs[j * e_count + e] = buf.ptr + at;
            var len_j: usize = 0;
            for (pieces(&pbuf, j, e < n, nv, gu, rest, owner[e] == 1, dcols, half)) |pc| {
                const bytes = try L.entry(try plan.partKey(&kb, try name(&nb, p, e, n, pc.proj), pc.cols, pc.rows, pc.part));
                try buf.upload(at + len_j, bytes);
                len_j += bytes.len;
            }
            at += std.mem.alignForward(usize, len_j, 256);
        };
        try L.w.buffers.append(L.gpa, buf);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "{s}.ffn.experts.native", .{p}), .ptr = buf.ptr, .len = total });
        L.w.bytes += total;
        var x: Experts = undefined;
        x.trellis = buf.ptr;
        x.count = @intCast(e_count);
        x.dims = @intCast(d);
        x.width = @intCast(width);
        x.down_k = @intCast(down_k);
        x.down_n = @intCast(down_n);
        // the shared expert's (the last) gate, up and down: their first byte and length (the paced L2 prefetch)
        x.shared_gate = .{ tabs[0 * e_count + n], tabs[1 * e_count + n] - tabs[0 * e_count + n] };
        x.shared_up = .{ tabs[1 * e_count + n], tabs[2 * e_count + n] - tabs[1 * e_count + n] };
        x.shared_down = .{ tabs[4 * e_count + n], tabs[5 * e_count + n] - tabs[4 * e_count + n] };
        x.rest = null;
        var nx: NativeExperts = undefined;
        const fields = [_]*u64{ &nx.w1, &nx.w3, &nx.s1, &nx.s3, &nx.w2, &nx.s2 };
        for (fields, 0..) |f, j| f.* = try L.upload(try std.fmt.bufPrint(&tb, "{s}.ffn.experts.native_tab{d}", .{ p, j }), std.mem.sliceAsBytes(tabs[j * e_count ..][0..e_count]));
        nx.tiles = @intCast((width + ir) / 32);
        nx.rest_tiles = @intCast(ir / 32);
        nx.nv = nv;
        nx.g13 = 0;
        nx.g2 = 0;
        nx.a1 = 0;
        nx.a3 = 0;
        nx.a2 = 0;
        if (nv) {
            const fl = try L.gpa.alloc(f32, 5 * e_count); // g13, g2, a1, a3, a2 over the experts
            defer L.gpa.free(fl);
            for (0..e_count) |e| {
                for (0..5) |j| fl[j * e_count + e] = 1;
                if (e == n) continue;
                const sc = struct {
                    fn of(Lo: *Loader, nbuf: []u8, kbuf: []u8, pp: []const u8, ee: usize, w: []const u8, tail: []const u8) !f32 {
                        const tn = try std.fmt.bufPrint(nbuf, "{s}.ffn.experts.{d}.{s}.{s}", .{ pp, ee, w, tail });
                        const bytes = try Lo.entry(try plan.plainKey(kbuf, tn, .f32));
                        if (bytes.len != 4) return error.UnexpectedTensor;
                        return std.mem.bytesToValue(f32, bytes[0..4]);
                    }
                }.of;
                const in1 = try sc(L, &nb, &kb, p, e, "w1", "input_scale");
                const in3 = try sc(L, &nb, &kb, p, e, "w3", "input_scale");
                const in2 = try sc(L, &nb, &kb, p, e, "w2", "input_scale");
                // gate and up read one quantized input: a checkpoint that calibrated them apart is not this kernel's
                if (in1 != in3 or !(in1 > 0) or !(in2 > 0)) return error.UnexpectedTensor;
                fl[0 * e_count + e] = 1 / in1;
                fl[1 * e_count + e] = 1 / in2;
                fl[2 * e_count + e] = in1 * try sc(L, &nb, &kb, p, e, "w1", "weight_scale_2");
                fl[3 * e_count + e] = in3 * try sc(L, &nb, &kb, p, e, "w3", "weight_scale_2");
                fl[4 * e_count + e] = in2 * try sc(L, &nb, &kb, p, e, "w2", "weight_scale_2");
            }
            const nvf = [_]*u64{ &nx.g13, &nx.g2, &nx.a1, &nx.a3, &nx.a2 };
            for (nvf, 0..) |ptr, j| ptr.* = try L.upload(try std.fmt.bufPrint(&tb, "{s}.ffn.experts.native_nv{d}", .{ p, j }), std.mem.sliceAsBytes(fl[j * e_count ..][0..e_count]));
        }
        nx.owner = if (rest != null) try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.native_owner", .{p}), std.mem.sliceAsBytes(owner)) else 0;
        x.native = nx;
        if (rest) |rc| {
            // the parity split's Rest as the 2D exchanges read it: the width and the owners (pair of e's rest)
            var r: Rest = undefined;
            r.width = @intCast(rc[1] - rc[0]);
            r.k2_gu = .{ 0, 0 };
            r.mine = pair.?;
            const own = try L.gpa.alloc(i32, e_count + 1);
            defer L.gpa.free(own);
            for (0..e_count) |e| own[e] = @intCast(plan.restPair(e, n));
            own[e_count] = 0;
            r.owner = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_owner", .{p}), std.mem.sliceAsBytes(own));
            r.gate_ptr = 0;
            r.up_ptr = 0;
            r.gate_k2 = 0;
            r.up_k2 = 0;
            r.svh_g = 0;
            r.svh_u = 0;
            r.suh_d = 0;
            r.remap = 0;
            x.rest = r;
        }
        return x;
    }

    /// A trellis entry's [kt, nt, w16].
    fn shape(L: *Loader, key: []const u8) ![3]usize {
        return switch (L.src) {
            .cache => |c| blk: {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.rank != 3) return error.UnexpectedTensor;
                break :blk .{ e.shape[0], e.shape[1], e.shape[2] };
            },
            .shards => |sh| blk: {
                const s = try checkpoint.describe(sh, key);
                if (s.rank != 3) return error.UnexpectedTensor;
                break :blk .{ s.shape[0], s.shape[1], s.shape[2] };
            },
        };
    }

    /// load_block's experts: exl3_parts of w1 / w3 by output columns and w2 by input rows for every expert and the
    /// shared one, packed as pack_trellises packs them, with prepare's tables. `half`: the rank's intermediate; `gu`:
    /// gate / up's columns of it here (the half, or a 2D node's blocks); `dcols`: down's output columns (null: all).
    fn experts(L: *Loader, p: []const u8, n: usize, half: [2]usize, gu: [2]usize, dcols: ?[2]usize, d: usize, rest: ?[2]usize, pair: ?u32) !Experts {
        const e_count = n + 1;
        const width = gu[1] - gu[0];
        const down_k = half[1] - half[0];
        const down_n = if (dcols) |dc| dc[1] - dc[0] else d;
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const prefixes = try L.gpa.alloc([3][]u8, e_count);
        defer {
            for (prefixes) |pp| for (pp) |x| L.gpa.free(x);
            L.gpa.free(prefixes);
        }
        for (0..e_count) |e| {
            const ep = if (e < n) try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{d}", .{ p, e }) else try std.fmt.bufPrint(&nb, "{s}.ffn.shared_experts", .{p});
            for ([_][]const u8{ "w1", "w3", "w2" }, 0..) |proj, j| prefixes[e][j] = try std.fmt.allocPrint(L.gpa, "{s}.{s}", .{ ep, proj });
        }
        // exl3_parts' (cols, rows) of projection j: gate / up by output columns, down by input rows (and on a 2D
        // node by its output columns too)
        const Parts = struct {
            gu: [2]usize,
            half: [2]usize,
            dcols: ?[2]usize,
            fn cols(q: @This(), j: usize) ?[2]usize {
                return if (j < 2) q.gu else q.dcols;
            }
            fn rows(q: @This(), j: usize) ?[2]usize {
                return if (j == 2) q.half else null;
            }
        };
        const pq: Parts = .{ .gu = gu, .half = half, .dcols = dcols };
        // sizes first: one buffer, gate trellises of every expert, then up, then down
        const sizes = try L.gpa.alloc(usize, 3 * e_count);
        defer L.gpa.free(sizes);
        var total: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            sizes[j * e_count + e] = try L.src.size(key);
            total += sizes[j * e_count + e];
        };
        var buf = try cuda.DeviceBuffer.alloc(L.d, total);
        errdefer buf.free();
        const ptrs = try L.gpa.alloc(u64, 3 * e_count);
        defer L.gpa.free(ptrs);
        const k2s = try L.gpa.alloc(i32, 3 * e_count);
        defer L.gpa.free(k2s);
        var at: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            const tr = try L.entry(key);
            try buf.upload(at, tr);
            const t = try L.shape(key);
            ptrs[j * e_count + e] = buf.ptr + at;
            k2s[j * e_count + e] = @intCast(t[2] / 8);
            at += sizes[j * e_count + e];
        };
        try L.w.buffers.append(L.gpa, buf);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "{s}.ffn.experts.trellis", .{p}), .ptr = buf.ptr, .len = total });
        L.w.bytes += total;
        var x: Experts = undefined;
        x.trellis = buf.ptr;
        x.shared_gate = .{ ptrs[n], sizes[n] }; // j = 0 (gate), e = n (the shared expert)
        x.shared_up = .{ ptrs[e_count + n], sizes[e_count + n] };
        x.shared_down = .{ ptrs[2 * e_count + n], sizes[2 * e_count + n] };
        x.count = @intCast(e_count);
        x.dims = @intCast(d);
        x.width = @intCast(width);
        x.down_k = @intCast(down_k);
        x.down_n = @intCast(down_n);
        const tables = [_]*u64{ &x.gate_ptr, &x.up_ptr, &x.down_ptr };
        const k2t = [_]*u64{ &x.gate_k2, &x.up_k2, &x.down_k2 };
        for (0..3) |j| {
            tables[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.ptr{d}", .{ p, j }), std.mem.sliceAsBytes(ptrs[j * e_count ..][0..e_count]));
            k2t[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.k2{d}", .{ p, j }), std.mem.sliceAsBytes(k2s[j * e_count ..][0..e_count]));
        }
        x.k2_gu = .{ std.math.maxInt(u32), 0 };
        x.k2_d = .{ std.math.maxInt(u32), 0 };
        for (k2s[0 .. 2 * e_count]) |v| x.k2_gu = .{ @min(x.k2_gu[0], @as(u32, @intCast(v))), @max(x.k2_gu[1], @as(u32, @intCast(v))) };
        for (k2s[2 * e_count ..]) |v| x.k2_d = .{ @min(x.k2_d[0], @as(u32, @intCast(v))), @max(x.k2_d[1], @as(u32, @intCast(v))) };
        // stack(mats, j, n): every expert's scale row in order; suh_d is down's input signs on gate / up's blocks
        // (experts2d.prepare: the half's [gu - half), all of it but on a 2D node)
        const off = gu[0] - half[0];
        const stacks = [_]struct { dst: *u64, proj: usize, part: []const u8, len: usize, src_len: usize, off: usize, name: []const u8 }{
            .{ .dst = &x.suh_g, .proj = 0, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_g" },
            .{ .dst = &x.suh_u, .proj = 1, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_u" },
            .{ .dst = &x.svh_g, .proj = 0, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_g" },
            .{ .dst = &x.svh_u, .proj = 1, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_u" },
            .{ .dst = &x.suh_d, .proj = 2, .part = "suh", .len = width, .src_len = down_k, .off = off, .name = "suh_d" },
            .{ .dst = &x.svh_d, .proj = 2, .part = "svh", .len = down_n, .src_len = down_n, .off = 0, .name = "svh_d" },
        };
        for (stacks) |s| {
            try L.aux.resize(L.gpa, e_count * s.len * 2);
            for (0..e_count) |e| {
                const j = s.proj;
                const row = try L.entry(try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), s.part));
                if (row.len != s.src_len * 2) return error.UnexpectedTensor;
                @memcpy(L.aux.items[e * s.len * 2 ..][0 .. s.len * 2], row[s.off * 2 ..][0 .. s.len * 2]);
            }
            s.dst.* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{s}", .{ p, s.name }), L.aux.items);
        }
        x.rest = null;
        if (rest) |rc| x.rest = try L.expertsRest(p, prefixes, n, half, dcols, rc, d, pair.?);
        return x;
    }

    /// experts2d.prepare_rest: the rest block's gate / up parts of the experts whose rest is this pair's (their
    /// trellises in one buffer), the tables over every expert (zeros where another node computes it), the remap and
    /// the owners.
    fn expertsRest(L: *Loader, p: []const u8, prefixes: []const [3][]u8, n: usize, half: [2]usize, dcols: ?[2]usize, rc: [2]usize, d: usize, mine: u32) !Rest {
        const e_count = n + 1;
        const ir = rc[1] - rc[0];
        if (rc[0] % 128 != 0 or ir % 128 != 0 or ir == 0) return error.UnsupportedShape;
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        var total: usize = 0;
        for (0..e_count) |e| {
            if (plan.restPair(e, n) != mine) continue;
            for (0..2) |j| total += try L.src.size(try plan.partKey(&kb, prefixes[e][j], rc, null, "tr"));
        }
        if (total == 0) return error.NoRestHere;
        var buf = try cuda.DeviceBuffer.alloc(L.d, total);
        errdefer buf.free();
        const ptrs = try L.gpa.alloc(u64, 2 * e_count);
        defer L.gpa.free(ptrs);
        const k2s = try L.gpa.alloc(i32, 2 * e_count);
        defer L.gpa.free(k2s);
        @memset(ptrs, 0);
        @memset(k2s, 0);
        const svh = try L.gpa.alloc(u8, 2 * e_count * ir * 2);
        defer L.gpa.free(svh);
        const suh_d = try L.gpa.alloc(u8, e_count * ir * 2);
        defer L.gpa.free(suh_d);
        @memset(svh, 0);
        @memset(suh_d, 0);
        const remap = try L.gpa.alloc(i32, e_count + 1);
        defer L.gpa.free(remap);
        const owner = try L.gpa.alloc(i32, e_count + 1);
        defer L.gpa.free(owner);
        var k2_gu: [2]u32 = .{ std.math.maxInt(u32), 0 };
        var at: usize = 0;
        for (0..e_count) |e| {
            const o = plan.restPair(e, n);
            owner[e] = @intCast(o);
            remap[e] = if (o == mine) @intCast(e) else @intCast(e_count);
            if (o != mine) continue;
            for (0..2) |j| {
                const key = try plan.partKey(&kb, prefixes[e][j], rc, null, "tr");
                const t = try L.shape(key);
                if (t[0] * 16 != d or t[1] * 16 != ir) return error.UnexpectedTensor;
                const tr = try L.entry(key);
                try buf.upload(at, tr);
                ptrs[j * e_count + e] = buf.ptr + at;
                const k2: u32 = @intCast(t[2] / 8);
                k2s[j * e_count + e] = @intCast(k2);
                k2_gu = .{ @min(k2_gu[0], k2), @max(k2_gu[1], k2) };
                at += tr.len;
                const row = try L.entry(try plan.partKey(&kb, prefixes[e][j], rc, null, "svh"));
                if (row.len != ir * 2) return error.UnexpectedTensor;
                @memcpy(svh[(j * e_count + e) * ir * 2 ..][0 .. ir * 2], row);
            }
            // down's input signs over the half (its rows), at the rest's columns
            const drow = try L.entry(try plan.partKey(&kb, prefixes[e][2], dcols, half, "suh"));
            if (drow.len != (half[1] - half[0]) * 2) return error.UnexpectedTensor;
            @memcpy(suh_d[e * ir * 2 ..][0 .. ir * 2], drow[(rc[0] - half[0]) * 2 ..][0 .. ir * 2]);
        }
        remap[e_count] = @intCast(e_count);
        owner[e_count] = 0;
        try L.w.buffers.append(L.gpa, buf);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "{s}.ffn.experts.rest_trellis", .{p}), .ptr = buf.ptr, .len = total });
        L.w.bytes += total;
        var r: Rest = undefined;
        r.width = @intCast(ir);
        r.k2_gu = k2_gu;
        r.mine = mine;
        r.gate_ptr = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_ptr0", .{p}), std.mem.sliceAsBytes(ptrs[0..e_count]));
        r.up_ptr = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_ptr1", .{p}), std.mem.sliceAsBytes(ptrs[e_count..]));
        r.gate_k2 = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_k20", .{p}), std.mem.sliceAsBytes(k2s[0..e_count]));
        r.up_k2 = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_k21", .{p}), std.mem.sliceAsBytes(k2s[e_count..]));
        r.svh_g = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_svh_g", .{p}), svh[0 .. e_count * ir * 2]);
        r.svh_u = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_svh_u", .{p}), svh[e_count * ir * 2 ..]);
        r.suh_d = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_suh_d", .{p}), suh_d);
        r.remap = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_remap", .{p}), std.mem.sliceAsBytes(remap));
        r.owner = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.rest_owner", .{p}), std.mem.sliceAsBytes(owner));
        return r;
    }

    /// load_block of a Format.native checkpoint (plan.blockNative's entries).
    fn blockNative(L: *Loader, c: Config, s: plan.Split, p: []const u8, i: usize, n_experts: usize) !Layer {
        var nb: [256]u8 = undefined;
        var pb: [256]u8 = undefined;
        const gl = s.groups(c);
        var lay: Layer = .{ .idx = @intCast(i), .ratio = c.ratios[i] };
        for ([_][]const u8{ "fn", "scale", "base" }, 0..) |part, k| {
            lay.hc_attn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_attn_{s}", .{ p, part }), .f32);
            lay.hc_ffn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_ffn_{s}", .{ p, part }), .f32);
        }
        lay.attn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn_norm.weight", .{p}), null);
        lay.ffn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn_norm.weight", .{p}), null);
        lay.wq_a = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.wq_a", .{p}), null, null, .fp8);
        lay.q_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.q_norm.weight", .{p}), null);
        lay.wq_b = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.wq_b", .{p}), s.wqB(c), null, .fp8);
        lay.wkv = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.wkv", .{p}), null, null, .fp8);
        lay.kv_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.kv_norm.weight", .{p}), null);
        {
            var kb: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.attn_sink", .{p});
            const all = try L.entry(try plan.plainKey(&kb, name, .f32));
            const hs = s.headSpan(c);
            lay.sink = try L.upload(name, all[hs[0] * 4 ..][0 .. hs[1] * 4]);
        }
        const ga = s.woAGroups(c);
        for (0..ga[1]) |g| {
            const pre = try std.fmt.bufPrint(&pb, "{s}.attn.wo_a", .{p});
            const gc: [2]usize = .{ (ga[0] + g) * c.o_lora, (ga[0] + g + 1) * c.o_lora };
            lay.wo_a[g] = if (L.wo_a_fp8) try L.native(pre, gc, null, .fp8) else try L.woANative(pre, ga[0] + g, c.o_lora);
        }
        lay.groups = @intCast(ga[1]);
        lay.wo_a8 = try L.woA8(try std.fmt.bufPrint(&pb, "{s}.attn.wo_a", .{p}), ga, c.o_lora);
        lay.wo_b = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.wo_b", .{p}), s.woBCols(c), s.range(gl * c.o_lora), .fp8);
        if (c.kv_sources.has(i)) {
            lay.comp_wkv = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.compressor.wkv", .{p}), null, null, .bf16);
            lay.comp_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.norm.weight", .{p}), null);
            if (c.ratios[i] > 1) lay.comp_wgate = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.compressor.wgate", .{p}), null, null, .bf16);
        }
        if (c.index_sources.has(i)) {
            lay.idx_wq_b = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.indexer.wq_b", .{p}), null, null, .fp8);
            // weights_proj stays bf16 (native.zig's bf16 linear: Indexer.weights_proj is a bf16 Linear)
            const wp = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.indexer.weights_proj", .{p}), null, null, .bf16);
            lay.idx_proj = wp.w;
            lay.idx_proj_h = 0;
            if (c.kv_sources.has(i)) {
                lay.idx_wk = try L.native(try std.fmt.bufPrint(&pb, "{s}.attn.indexer.wk", .{p}), null, null, .bf16);
                lay.idx_k_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.k_norm.weight", .{p}), null);
            }
        }
        lay.gate_w = (try L.native(try std.fmt.bufPrint(&pb, "{s}.ffn.gate", .{p}), null, null, .bf16)).w;
        lay.gate_b = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias", .{p}), .f32);
        if (c.vision) lay.gate_b_vl = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias_vl", .{p}), .f32);
        const xp = s.expertParts(c);
        lay.experts = try L.expertsNative(p, n_experts, xp.half, xp.gu, xp.dcols, c.hidden, xp.rest, s.pair, c.nvfp4 and i < c.layers);
        if (c.engram_layers.has(i)) {
            lay.engram_wkv = try L.native(try std.fmt.bufPrint(&pb, "{s}.engram.wkv", .{p}), s.engramCols(c), s.range(s.engramRows(c)), .fp8);
            var kb: [256]u8 = undefined;
            const q = try L.gpa.dupe(u8, try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}), .f32)));
            defer L.gpa.free(q);
            const k = try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}), .f32));
            if (q.len != k.len) return error.UnexpectedTensor;
            // Engram.forward: q_weight.float() * k_weight.float(), one fp32 rounding a product
            for (0..q.len / 4) |j| {
                const a: f32 = @bitCast(std.mem.readInt(u32, q[4 * j ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, k[4 * j ..][0..4], .little));
                std.mem.writeInt(u32, q[4 * j ..][0..4], @bitCast(a * b), .little);
            }
            lay.engram_qk = try L.upload(try std.fmt.bufPrint(&nb, "{s}.engram.qk", .{p}), q);
        }
        return lay;
    }

    fn block(L: *Loader, c: Config, s: plan.Split, p: []const u8, i: usize, n_experts: usize) !Layer {
        if (c.format == .native) return L.blockNative(c, s, p, i, n_experts);
        var nb: [256]u8 = undefined;
        const gl = s.groups(c);
        var lay: Layer = .{ .idx = @intCast(i), .ratio = c.ratios[i] };
        for ([_][]const u8{ "fn", "scale", "base" }, 0..) |part, k| {
            lay.hc_attn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_attn_{s}", .{ p, part }), .f32);
            lay.hc_ffn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_ffn_{s}", .{ p, part }), .f32);
        }
        lay.attn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn_norm.weight", .{p}), null);
        lay.ffn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn_norm.weight", .{p}), null);
        lay.wq_a = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_a", .{p}), null, null);
        lay.q_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.q_norm.weight", .{p}), null);
        lay.wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_b", .{p}), s.wqB(c), null);
        lay.wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wkv", .{p}), null, null);
        lay.kv_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.kv_norm.weight", .{p}), null);
        {
            var kb: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.attn_sink", .{p});
            const all = try L.entry(try plan.plainKey(&kb, name, .f32));
            const hs = s.headSpan(c);
            lay.sink = try L.upload(name, all[hs[0] * 4 ..][0 .. hs[1] * 4]);
        }
        const ga = s.woAGroups(c);
        for (0..ga[1]) |g| lay.wo_a[g] = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ p, ga[0] + g }), null, null);
        lay.groups = @intCast(ga[1]);
        lay.wo_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_b", .{p}), s.woBCols(c), s.range(gl * c.o_lora));
        if (s.pair != null) {
            // the 2D column parts keep their TP2 slices' decode plans
            lay.wq_b.plan_n = @intCast(s.heads(c) * c.head_dim);
            lay.wo_b.plan_n = @intCast(c.hidden);
        }
        if (c.kv_sources.has(i)) {
            lay.comp_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wkv", .{p}), null, null);
            lay.comp_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.norm.weight", .{p}), null);
            if (c.ratios[i] > 1) lay.comp_wgate = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wgate", .{p}), null, null);
        }
        if (c.index_sources.has(i)) {
            lay.idx_wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wq_b", .{p}), null, null);
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj.weight", .{p});
            var kb: [256]u8 = undefined;
            const f32s = try L.entry(try plan.plainKey(&kb, name, .f32));
            lay.idx_proj = try L.upload(name, f32s);
            // Model.__init__: idx_proj.to(float16), the stored f16 values back (the f32 copy widened them exactly)
            try L.aux.resize(L.gpa, f32s.len / 2);
            for (0..f32s.len / 4) |k| {
                const v: f32 = @bitCast(std.mem.readInt(u32, f32s[4 * k ..][0..4], .little));
                std.mem.writeInt(u16, L.aux.items[2 * k ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
            }
            lay.idx_proj_h = try L.upload(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj_h", .{p}), L.aux.items);
            if (c.kv_sources.has(i)) {
                lay.idx_wk = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wk", .{p}), null, null);
                lay.idx_k_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.k_norm.weight", .{p}), null);
            }
        }
        lay.gate_w = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.weight", .{p}), .f16);
        lay.gate_b = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias", .{p}), .f32);
        const xp = s.expertParts(c);
        lay.experts = try L.experts(p, n_experts, xp.half, xp.gu, xp.dcols, c.hidden, xp.rest, s.pair);
        if (c.engram_layers.has(i)) {
            lay.engram_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.engram.wkv", .{p}), s.engramCols(c), s.range(s.engramRows(c)));
            if (s.pair != null) lay.engram_wkv.?.plan_n = @intCast((c.hc + 1) * c.hidden);
            var kb: [256]u8 = undefined;
            const q = try L.gpa.dupe(u8, try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}), .f32)));
            defer L.gpa.free(q);
            const k = try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}), .f32));
            if (q.len != k.len) return error.UnexpectedTensor;
            // q * k on the GPU in f32: one rounding a product, the same on the host
            for (0..q.len / 4) |j| {
                const a: f32 = @bitCast(std.mem.readInt(u32, q[4 * j ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, k[4 * j ..][0..4], .little));
                std.mem.writeInt(u32, q[4 * j ..][0..4], @bitCast(a * b), .little);
            }
            lay.engram_qk = try L.upload(try std.fmt.bufPrint(&nb, "{s}.engram.qk", .{p}), q);
        }
        return lay;
    }
};

/// Every tensor of rank `s` on device `d`, from `src` (weights.py _load, in its order).
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, src: Source, c: Config, s: plan.Split, dspark: bool) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .d = d, .src = src, .w = &w };
    var pf: Prefetch = undefined;
    var pf_on = false;
    if (src == .cache and Prefetch.on()) {
        const rc = src.cache;
        pf = .{ .io = rc.io, .fd = rc.file.handle, .size = rc.index.data_end };
        L.src.cache.pf = &pf;
        pf.start();
        pf_on = true;
    }
    defer {
        if (pf_on) pf.finish();
        L.host.deinit(gpa);
        L.aux.deinit(gpa);
    }
    w.embed = try L.plain("embed.weight", null);
    w.norm = try L.plain("norm.weight", null);
    const vr = s.headCols(c);
    if (c.format == .native) {
        w.head = try L.native("head", vr, null, .bf16);
    } else {
        w.head = try L.linear("head", vr, null);
        if (s.pair != null) w.head.plan_n = @intCast(s.vocab(c)); // the vocabulary half's plan
    }
    w.vocab_lo = vr[0];
    w.vocab_hi = vr[1];
    w.layers = try gpa.alloc(Layer, c.layers);
    var nb: [64]u8 = undefined;
    for (0..c.layers) |i| w.layers[i] = try L.block(c, s, try std.fmt.bufPrint(&nb, "layers.{d}", .{i}), i, c.experts);
    if (dspark and c.dspark_block > 0) {
        const blocks = try gpa.alloc(Layer, c.draft_layers);
        errdefer gpa.free(blocks);
        // DSpark stays TP2 inside the pair on a 2D node (weights.py _load_2d)
        const tp2: plan.Split = .{ .rank = s.rank, .world = s.world };
        // the drafter's wo_a reads its FP8 rows: act_quant(o) times them is the reference's quant-dequantized o times
        // the dequantized bf16 rows, product for product (TF_DS_DRAFT_WOA=bf16: the bf16 rows, as the backbone's)
        L.wo_a_fp8 = c.format == .native and draftWoAFp8();
        defer L.wo_a_fp8 = false;
        for (0..c.draft_layers) |j| blocks[j] = try L.block(c, tp2, try std.fmt.bufPrint(&nb, "mtp.{d}", .{j}), c.layers + j, c.draft_experts);
        var lb: [64]u8 = undefined;
        const last = try std.fmt.bufPrint(&lb, "mtp.{d}", .{c.draft_layers - 1});
        var tb: [128]u8 = undefined;
        w.dspark = .{
            .blocks = blocks,
            .main_proj = if (c.format == .native) try L.native("mtp.0.main_proj", null, null, .fp8) else try L.linear("mtp.0.main_proj", null, null),
            .main_norm = try L.plain("mtp.0.main_norm.weight", null),
            .norm = try L.plain(try std.fmt.bufPrint(&tb, "{s}.norm.weight", .{last}), null),
            .markov_embed = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.embed.weight", .{last}), null),
            .markov_head = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.head.weight", .{last}), null),
            .conf = try L.plain(try std.fmt.bufPrint(&tb, "{s}.confidence_head.proj.weight", .{last}), if (c.format == .native) .f32 else null),
        };
        if (c.format == .native and draftHeadFp8()) w.dspark.?.head8 = try L.headFp8(vr);
    }
    return w;
}

/// An E4M3 byte's value (no NaN in weights).
fn e4m3(b: u8) f32 {
    const sign: f32 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xf;
    const m: f32 = @floatFromInt(b & 7);
    if (e == 0) return sign * m / 8 * std.math.pow(f32, 2, -6);
    return sign * (1 + m / 8) * std.math.pow(f32, 2, @floatFromInt(e - 7));
}

test "e4m3 values" {
    try std.testing.expectEqual(@as(f32, 1.0), e4m3(0x38));
    try std.testing.expectEqual(@as(f32, -448.0), e4m3(0xfe));
    try std.testing.expectEqual(@as(f32, 0.5), e4m3(0x30));
    try std.testing.expectEqual(@as(f32, 0.001953125), e4m3(0x01)); // 2^-9: the smallest subnormal
}

test "strips: each 128-column block's tiles in k order" {
    // kt 2, nt 16 (two blocks of 8 tiles), one byte a tile
    var src: [32]u8 = undefined;
    for (&src, 0..) |*x, i| x.* = @intCast(i);
    var out: [32]u8 = undefined;
    strips(&src, &out, 2, 16, 1);
    // block 0: k row 0 tiles 0..7, k row 1 tiles 16..23; block 1: tiles 8..15, then 24..31
    const want = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    try std.testing.expectEqualSlices(u8, &want, &out);
}
