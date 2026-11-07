//! The DSpark drafter of the served lane's concurrent decoder (dspark.py's Drafter, DraftPool and BatchDraftGraph as
//! multi.py runs them) on one TP rank: the streams' window rings in one pool, the target's taps absorbed into them
//! (absorb after a prompt chunk; absorb_many of every drafting stream's window rows after a round's forward, the served
//! TF_DS_EAGER_ABSORB), and N streams' drafts and confidences from one pass: BatchDraftGraph._body with
//! TF_DS_DRAFT_FUSED and the hc switch (the three stages, each sublayer's post fused into the next mixes; q's RMSNorm
//! writing wq_b's rotated rows and q's RoPE in wq_b's epilogue; the block rows' keys normed and roped, stored nowhere;
//! the sparse attention over the stream's 128 newest absorbed positions and its own block rows; wo_a's epilogue
//! rotating wo_b's rows; the MoE at top 3 of the routed experts and the shared one), the Markov loop (markov.py: the
//! vocabulary split over the ranks, the bias rows of 256 frequent tokens cached, each step's bests gathered) and the
//! confidence head (one fp32 GEMV as torch issues it). The served L2 prefetch forks move no bytes and are left out; the
//! RDMA gathers are NCCL all-gathers (the same bytes).
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const plan = @import("plan.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri = @import("tri.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_linear = @import("exl3_linear.zig");
const mx = @import("mx.zig");
const exl3_experts = @import("exl3_experts.zig");
const markov_tokens = @import("markov_tokens.zig");
const draft2d = @import("draft2d.zig");

const Engine = prompt.Engine;
const Chunk = prompt.Chunk;

/// MultiDecoder's stream slots in the served lane (--parallel 4): the pool's streams.
pub const max_streams = 16; // the most streams any stream setting (Model.streams) gives the pool
/// The served pool's streams when unset (--parallel 4); the four-node lane serves 16.
pub const default_streams = 4;
/// A pass's rows at most: every stream's block (four streams: 20). Past kernels.decode_rows() (16 rows: four streams) the MoE gate is a
/// cuBLAS matmul with the plain routing and the attention one split (its pick list padded to 16 columns of -1).
pub const max_rows = max_streams * 5; // every stream of the pool, a five-row block each
/// kernels.decode_rows(): the row-invariant decode kernels' rows at most.
/// sparse_attn's pick list of a single-split call: padded with -1 to a multiple of 16 entries.
const pick_pad = 16;
/// dspark.py: ring_size = window + 16.
pub const ring_extra = 16;
/// The stages (DSpark blocks) and a stream's block rows at most.
pub const max_stages = 4;
pub const max_block = 8;
/// absorb_many's rows at most: every drafting stream's verify window (16 rows a round).
pub const max_absorb = 48; // a round's drafting rows at most (round.max_rows)

/// DraftPool: one ring plane a stage, bf16 [slots * ring, head_dim]; slot s's rows [s * ring, (s + 1) * ring). Positions
/// below `absorbed[s]` are in slot s's rings.
pub const Pool = struct {
    rings: [max_stages]u64 = @splat(0),
    stages: usize,
    slots: usize,
    ring: usize,
    hd: usize,
    absorbed: [max_streams]usize = @splat(0),

    /// Zeros (torch.zeros), as the served pool starts.
    pub fn init(e: *const Engine, a: *prompt.Arena, slots: usize) !Pool {
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or slots > max_streams) return error.DrafterTooLarge;
        var p: Pool = .{ .stages = ds.blocks.len, .slots = slots, .ring = e.c.window + ring_extra, .hd = e.c.head_dim };
        for (0..p.stages) |j| {
            p.rings[j] = try a.take(p.planeBytes());
            try prompt.fill(e, p.rings[j], 0, p.planeBytes());
        }
        return p;
    }

    pub fn planeBytes(p: *const Pool) usize {
        return p.slots * p.ring * p.hd * 2;
    }

    /// Slot s's rows of stage j's plane (DraftPool.views[s].rings[j]).
    pub fn view(p: *const Pool, j: usize, s: usize) u64 {
        return p.rings[j] + s * p.ring * p.hd * 2;
    }
};

/// The points of a pass a checker sees, in the order the served build's recorder takes them (stage j: layer
/// layers + j): each stage's attention input rows and its stage's ring plane, its partial and gather, the MoE's input
/// rows, partial and gather; the streams and pre after the stages; the head's logits (this rank's columns); the drafts.
pub const Point = enum { attn_in, attn_ring, attn_out, attn_gather, moe_in, moe_out, moe_gather, stages_h, stages_pre, local, markov };

/// A checker of a pass's buffers between its steps (synchronized first by the checker): false stops the pass.
pub const Probe = struct {
    ctx: *anyopaque,
    at: *const fn (ctx: *anyopaque, what: Point, stage: usize, dev: u64) anyerror!bool,

    fn check(pb: ?Probe, what: Point, stage: usize, dev: u64) !void {
        const p = pb orelse return;
        if (!try p.at(p.ctx, what, stage, dev)) return error.DraftMismatch;
    }
};

/// TF_DS_DRAFT_DUMP=DIR (a debugging aid, off in service): node 0 appends every absorbed taps row (DIR/taps.bin bf16
/// [n, taps * D], DIR/tpos.bin int64 positions) until pass TF_DS_DRAFT_DUMP_PASS (default 2), then runs that pass
/// eagerly and writes its intermediates (DIR/<name>-<stage>.bin, raw device bytes) and DIR/meta.txt.
pub var ddump: ?[:0]const u8 = null;
pub var ddump_pass: usize = 2;
pub var ddump_node: u32 = 0;
var ddump_count: usize = 0;
var ddump_on: bool = false; // inside the dumped pass

fn ddWrite(e: *const Engine, name: []const u8, stage: usize, ptr: u64, bytes: usize, append: bool) void {
    const dir = ddump orelse return;
    if (ddump_node != 0 or bytes == 0) return;
    e.s.synchronize() catch return;
    const host = std.heap.page_allocator.alloc(u8, bytes) catch return;
    defer std.heap.page_allocator.free(host);
    e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH") catch return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/{s}-{d}.bin", .{ dir, name, stage }, 0) catch return;
    const f = std.c.fopen(path.ptr, if (append) "ab" else "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(host.ptr, 1, bytes, f);
}

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// A host buffer of the dumped pass.
fn ddHost(name: []const u8, bytes: []const u8) void {
    const dir = ddump orelse return;
    if (ddump_node != 0) return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/{s}-0.bin", .{ dir, name }, 0) catch return;
    const f = std.c.fopen(path.ptr, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, f);
}

/// A dumped pass's buffer (only inside the dumped pass).
fn dd(e: *const Engine, name: []const u8, stage: usize, ptr: u64, bytes: usize) void {
    if (ddump_on) ddWrite(e, name, stage, ptr, bytes, false);
}

/// Absorbed taps rows and their positions (host), appended until the dumped pass.
fn ddTaps(e: *const Engine, dr: *const Drafter, taps: u64, pos: []const i64) void {
    if (ddump == null or ddump_count > ddump_pass) return;
    ddWrite(e, "taps", 0, taps, pos.len * dr.taps_w * 2, true);
    const dir = ddump.?;
    if (ddump_node != 0) return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/tpos-0.bin", .{dir}, 0) catch return;
    const f = std.c.fopen(path.ptr, "ab") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(std.mem.sliceAsBytes(pos).ptr, 8, pos.len, f);
}

const DumpProbe = struct {
    e: *const Engine,
    dr: *const Drafter,
    pool: *const Pool,
    R: usize,
    N: usize,
    steps: usize,

    fn at(ctx: *anyopaque, what: Point, stage: usize, dev: u64) anyerror!bool {
        const p: *DumpProbe = @ptrCast(@alignCast(ctx));
        const e = p.e;
        const d = e.c.hidden;
        const R = p.R;
        const bytes: usize = switch (what) {
            .attn_in, .moe_in => R * d * 2,
            .attn_ring => p.pool.slots * p.pool.ring * p.pool.hd * 2,
            .attn_out, .moe_out => R * d * 4,
            .attn_gather, .moe_gather => e.world * R * d * 4,
            .stages_h => R * e.c.hc * d * 2,
            .stages_pre => R * e.c.hc * 4,
            .local => R * e.w.head.n * 4,
            .markov => p.N * (p.steps + 1) * 8,
        };
        ddWrite(e, @tagName(what), stage, dev, bytes, false);
        return true;
    }
};

/// Debug (TF_DS_DRAFT_DUMP): a round's tap rows and the streams they came from (the first rounds), appended.
var tapcheck_n: usize = 0;
pub fn ddTapCheck(e: *const Engine, h: u64, tap: u64, ld: usize, rows: usize) void {
    if (ddump == null or tapcheck_n >= 12) return;
    tapcheck_n += 1;
    const d = e.c.hidden;
    ddWrite(e, "tc_h", 0, h, rows * e.c.hc * d * 2, true);
    // the tap rows: [rows, d] of a [rows, ld] buffer
    for (0..rows) |r| ddWrite(e, "tc_t", 0, tap + r * ld * 2, d * 2, true);
}

/// TF_DS_DRAFT_PROF=1 (a profile, off in service): passes run eagerly with a synchronize at each mark; each mark's
/// host ms since the previous one summed by name (printed by printProf).
pub var dprof: bool = false;
/// TF_DS_DRAFT_LOG=FILE (a measurement, off in service): node 0 appends each pass's drafts and confidences.
pub var dlog: ?[:0]const u8 = null;
const prof_names = [_][]const u8{ "start", "embed", "mix_a", "qkv_a", "q_b", "kv_rope", "sattn", "wo_a", "wo_b", "gather_a", "mix_f", "gate", "experts_gu", "experts_d", "gather_m", "post", "head", "markov", "conf", "conf_a", "conf_b", "gather_a0", "gather_m0" };
const ProfMark = enum { start, embed, mix_a, qkv_a, q_b, kv_rope, sattn, wo_a, wo_b, gather_a, mix_f, gate, experts_gu, experts_d, gather_m, post, head, markov, conf, conf_a, conf_b, gather_a0, gather_m0 };
var prof_ns: [prof_names.len]u64 = @splat(0);
var prof_last: u64 = 0;
var prof_passes: u64 = 0;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn pmark(e: *const Engine, m: ProfMark) void {
    if (!dprof) return;
    e.s.synchronize() catch {};
    const t = nowNs();
    if (m != .start) prof_ns[@intFromEnum(m)] += t - prof_last;
    prof_last = t;
}

pub fn printProf() void {
    if (!dprof or prof_passes == 0) return;
    var buf: [2048]u8 = undefined;
    var n: usize = 0;
    n += (std.fmt.bufPrint(buf[n..], "{{\"follow\": \"draft\", \"draft_profile_ms\": {{\"passes\": {d}", .{prof_passes}) catch return).len;
    var tot: u64 = 0;
    for (prof_names, 0..) |nm, i| {
        if (i == 0) continue;
        tot += prof_ns[i];
        n += (std.fmt.bufPrint(buf[n..], ", \"{s}\": {d:.3}", .{ nm, @as(f64, @floatFromInt(prof_ns[i])) / @as(f64, @floatFromInt(prof_passes)) / 1e6 }) catch return).len;
    }
    n += (std.fmt.bufPrint(buf[n..], ", \"total\": {d:.3}}}}}\n", .{@as(f64, @floatFromInt(tot)) / @as(f64, @floatFromInt(prof_passes)) / 1e6}) catch return).len;
    std.debug.print("{s}", .{buf[0..n]});
}

/// An absorb_many item: a drafting stream's slot, its window rows' taps (bf16 [n, taps * D], contiguous rows) and the
/// window's first position.
pub const Item = struct { slot: usize, taps: u64, n: usize, start: usize };

/// TF_DS_DRAFT_LINK (unset or 1; 0: off): a 2D node's drafter gathers its sublayer partials from its pair partner over
/// their two-rank link communicator (the ring neighbours' TF_DS_LINK_GRAPHS links) instead of the four-rank
/// all-gather's pair half: the same bytes, one hop.
pub var pair_link: bool = true;

/// The drafter's sublayer gather: fp32 [R, width] partials -> dst [world, R, width] in TP2 rank order (a 2D node: its
/// pair's two, over the partner link when there is one).
fn pairGather(e: *const Engine, src: u64, dst: u64, R: usize, width: usize) !void {
    const tw = e.two orelse return e.gatherF32(src, dst, R * width);
    if (pair_link and tw.rings == null) {
        const partner = tw.nccl[2 * tw.p + (1 - tw.r)];
        if (e.comm.linkTo(partner)) |l| {
            const count = R * width;
            if (l.me == tw.r) return e.comm.linkGather(l.comm, src, dst, count, .f32, e.s);
            // the link's order is the other way round: through the parts, swapped
            if (2 * count * 4 > 4 * tw.part_bytes) return error.PartTooBig;
            try e.comm.linkGather(l.comm, src, tw.parts, count, .f32, e.s);
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, tw.parts + count * 4, count * 4, e.s.handle), "cuMemcpyDtoDAsync");
            return e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst + count * 4, tw.parts, count * 4, e.s.handle), "cuMemcpyDtoDAsync");
        }
    }
    return tw.pairGather(e, src, dst, R, width, 4);
}

/// The per-step gather of the Markov loop's bests: [world, rows, 4] fp32 into dst.
const MarkovGather = struct {
    e: *const Engine,
    dst: u64,

    fn run(ctx: *anyopaque, send: u64, rows: usize) anyerror!u64 {
        const g: *MarkovGather = @ptrCast(@alignCast(ctx));
        try g.e.gatherF32(send, g.dst, rows * 4);
        return g.dst;
    }
};

/// The drafter pass's CUDA graphs (BatchDraftGraph), one a (streams, steps): every pointer it reads is a fixed
/// buffer and every per-pass value (ids, positions, slots, the pick lists) is uploaded ahead of the replay.
pub const PassGraphs = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(Key, cuda.graph.Exec) = .empty,
    captures: u64 = 0, // captures attempted (TF_DS_GRAPH_FAIL_TEST=RANK:N:pass counts them)

    pub const Key = struct { streams: u32, steps: u32 };

    pub fn init(gpa: std.mem.Allocator) PassGraphs {
        return .{ .gpa = gpa };
    }

    pub fn deinit(g: *PassGraphs) void {
        var it = g.map.valueIterator();
        while (it.next()) |x| x.deinit();
        g.map.deinit(g.gpa);
        g.* = undefined;
    }

    pub fn count(g: *const PassGraphs) usize {
        return g.map.count();
    }

    fn run(g: *PassGraphs, dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize) !void {
        const key: Key = .{ .streams = @intCast(N), .steps = @intCast(steps) };
        if (g.map.getPtr(key)) |x| return x.launchOn(e.s);
        // G0.3, as round.Graphs.run: every rank captures and instantiates (the map's room reserved first), then all
        // agree before any launch; one failure fails the pass on every rank
        var failed: ?anyerror = null;
        var exec_got: ?cuda.graph.Exec = null;
        // the map's room first, agreed before anyone records: NCCL needs every rank to capture the same collectives
        // (a rank that skipped the capture would leave the others' recorded exchanges unmatched)
        const room = g.map.ensureUnusedCapacity(g.gpa, 1);
        if (!try e.agreeAll(if (room) |_| true else |_| false)) {
            try room;
            return error.GraphCaptureFailed;
        }
        capture: {
            cuda.graph.beginCapture(e.s, .thread_local) catch |err| {
                failed = err;
                break :capture;
            };
            dr.body(e, ch, pool, N, steps, null) catch |err| {
                if (cuda.graph.endCapture(e.s)) |got| {
                    var x = got;
                    x.deinit();
                } else |_| {}
                failed = err;
                break :capture;
            };
            var graph = cuda.graph.endCapture(e.s) catch |err| {
                failed = err;
                break :capture;
            };
            defer graph.deinit();
            exec_got = graph.instantiate() catch |err| {
                failed = err;
                break :capture;
            };
        }
        g.captures += 1;
        if (failed == null and prompt.injectedCaptureFailure(.pass, e.comm.rank, g.captures)) {
            if (exec_got) |*x| x.deinit();
            exec_got = null;
            failed = error.GraphInjectedFailure;
        }
        const all_ok = e.agreeAll(failed == null) catch |err| {
            if (exec_got) |*x| x.deinit();
            return err;
        };
        if (!all_ok) {
            if (exec_got) |*x| x.deinit();
            if (failed) |err| {
                std.log.err("drafter pass graph of {d} streams, {d} steps failed on rank {d}: {s}; the pass fails on every rank", .{ N, steps, e.comm.rank, @errorName(err) });
                return err;
            }
            std.log.err("drafter pass graph of {d} streams, {d} steps failed on another rank; the pass fails on every rank", .{ N, steps });
            return error.GraphCaptureFailed;
        }
        const exec = exec_got.?;
        // The cache owns exec from here, including when the first launch fails.
        g.map.putAssumeCapacity(key, exec);
        try exec.launchOn(e.s);
    }
};

pub const Drafter = struct {
    ds: weights.DSpark,
    n: usize, // a stream's block rows: [token, noise x (n - 1)]
    noise: i64,
    topk: usize,
    slots_moe: usize, // the MoE's slots a row: top-k routed and the shared expert
    taps_w: usize, // the taps' row width: taps * D
    // the pass's index rows (int64, max_rows a row: ids, positions, window positions, ring bases, zeros, -1), bidx
    // [max_rows, n] (each row's stream's block rows; a single-split pass's padded to 16 columns of -1), wlo int64 [1] =
    // 0 (Model._zero)
    rows: u64,
    bidx: u64,
    wlo: u64,
    // the streams and the mHC state
    h: u64,
    h_alt: u64,
    x: u64,
    part: u64,
    pre: u64,
    pre_a: u64,
    pre_f: u64,
    post: u64,
    comb: u64,
    // attention
    qa: u64,
    ykv: u64,
    qr: u64,
    xhq: u64, // fp16 [R, q_lora]: wq_b's rotated input rows (the RMSNorm writes them)
    q: u64,
    kvb: u64, // bf16 [R, head_dim]: the block rows' keys
    o: u64,
    pm_: u64,
    pl_: u64,
    po_: u64,
    u: u64,
    xb: u64, // fp16 [R, wo_b.k]: wo_b's rotated input rows (wo_a's epilogue writes them)
    pa: u64,
    ga: u64,
    // MoE
    gl: u64,
    xf: u64, // fp32 [R, D]: a wide pass's rows for the gate's cuBLAS matmul
    gate_f: [max_stages]u64, // fp32 [experts, D]: each stage's gate widened (gate_w.float())
    pick: u64,
    mw: u64,
    pm: u64,
    gm: u64,
    xsd: exl3_experts.DecodeScratch,
    xs: exl3_experts.Scratch, // a pass of EXACT_ROWS rows or more (13+ streams): the chunk path on a scratch of its own (d48)
    mxe: mx.ExpertScratch = undefined, // Format.native: the stages' experts' grouping and rows (mx.zig)
    nxh: u64 = 0, // Format.native: the stages' intermediate rows, bf16 [max_rows * slots, width]
    // the head
    xc: u64, // bf16 [R, D]: the collapsed streams (pre-norm: the confidence head reads them)
    xn: u64,
    local: u64, // fp32 [R, this rank's vocabulary columns]
    // the Markov loop
    mk: tri_markov.Markov,
    msc: tri_markov.Scratch,
    mg: u64, // fp32 [world, max_streams, 4]: a step's gathered bests
    out: u64, // int64 [max_streams, n + 1]: each stream's last token, then its drafts
    // the confidence head
    embs: u64, // bf16 [max_rows, rank]: the Markov embedding rows of the tokens each step read
    xcf: u64, // fp32 [max_rows, D]
    ef: u64, // fp32 [max_rows, rank]
    conf_in: u64, // fp32 [max_rows, D + rank]
    confw: u64, // fp32 [D + rank]: the head's weight widened (conf.float())
    conf: u64, // fp32 [max_rows]
    confw_host: []f32 = &.{}, // the head's fp32 weight on the host: the confidence dot runs there (pass)
    xc_host: []u16 = &.{}, // bf16 [max_rows, D]: the collapsed rows read back
    em_host: []u16 = &.{}, // bf16 [max_rows, rank]: the embedding rows read back
    // absorbs: the taps rows (one item's or the items' cat), main_proj's rows, main_x, each stage's wkv rows, the
    // keys' normed rows (kv_norm_rope's out, not read), positions and ring rows
    at: u64,
    ay: u64,
    ax: u64,
    ak: [max_stages]u64,
    ao: u64,
    apos: u64,
    arows: u64,
    // the pass's graphs (BatchDraftGraph: one a streams and steps; null: eager)
    graphs: ?*PassGraphs = null,
    // the last pass on the host: each stream's drafts and confidences (`steps` of them)
    streams: usize = 0,
    steps: usize = 0,
    drafts: [max_streams][max_block]i64 = undefined,
    confs: [max_streams][max_block]f32 = undefined,

    /// Every buffer for passes of up to max_rows rows and absorbs of up to a ring of rows; the Markov loop's cached
    /// bias rows (Markov.fill, at load as served) for rank s.rank of s.world.
    pub fn init(e: *const Engine, a: *prompt.Arena, sp: plan.Split) !Drafter {
        const c = e.c;
        const ds = e.w.dspark orelse return error.NoDrafter;
        if (ds.blocks.len > max_stages or c.dspark_block > max_block or c.dspark_block == 0) return error.DrafterTooLarge;
        const d = c.hidden;
        const hc = c.hc;
        const hd = c.head_dim;
        const R = max_rows;
        const b0 = ds.blocks[0];
        const hl = b0.wq_b.n / hd; // the blocks' heads (TP2's on a 2D node too: TP2 inside the pair)
        var dr: Drafter = undefined;
        dr.ds = ds;
        dr.n = c.dspark_block;
        dr.noise = c.dspark_noise;
        dr.topk = c.draft_top_k;
        dr.slots_moe = c.draft_top_k + 1;
        dr.taps_w = c.dspark_taps.slice().len * d;
        dr.streams = 0;
        dr.steps = 0;
        dr.graphs = null;
        dr.rows = try a.take(6 * R * 8);
        dr.bidx = try a.take(R * pick_pad * 8);
        dr.wlo = try a.take(8);
        try prompt.fill(e, dr.wlo, 0, 8);
        dr.h = try a.take(R * hc * d * 2);
        dr.h_alt = try a.take(R * hc * d * 2);
        dr.x = try a.take(R * d * 2);
        dr.part = try a.take(R * tri_basic.hc_blocks * 32 * 4);
        dr.pre = try a.take(R * hc * 4);
        dr.pre_a = try a.take(R * hc * 4);
        dr.pre_f = try a.take(R * hc * 4);
        dr.post = try a.take(R * hc * 4);
        dr.comb = try a.take(R * hc * hc * 4);
        dr.qa = try a.take(R * b0.wq_a.n * 2);
        dr.ykv = try a.take(R * hd * 2);
        dr.qr = try a.take(R * b0.wq_a.n * 2);
        dr.xhq = try a.take(R * b0.wq_b.k * 2);
        dr.q = try a.take(R * hl * hd * 2);
        dr.kvb = try a.take(R * hd * 2);
        dr.o = try a.take(R * hl * hd * 2);
        dr.pm_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.pl_ = try a.take(R * hl * tri_attn.attn_splits * 4);
        dr.po_ = try a.take(R * hl * tri_attn.attn_splits * hd * 4);
        dr.u = try a.take(R * b0.groups * b0.wo_a[0].n * 2);
        dr.xb = try a.take(R * b0.wo_b.k * 2);
        dr.pa = try a.take(R * d * 4);
        dr.ga = try a.take(e.world * R * d * 4);
        const ne: usize = b0.experts.count - 1;
        dr.gl = try a.take(@max(16 * (d / 256), R) * ne * 4);
        dr.xf = try a.take(R * d * 4);
        dr.gate_f = @splat(0);
        if (e.mx == null) for (ds.blocks, 0..) |lay, j| {
            dr.gate_f[j] = try a.take(ne * d * 4);
            try e.ops.f16ToF32(e.s, lay.gate_w, dr.gate_f[j], ne * d);
        };
        dr.pick = try a.take(R * dr.slots_moe * 4);
        dr.mw = try a.take(R * dr.slots_moe * 4);
        dr.pm = try a.take(R * d * 4);
        dr.gm = try a.take(e.world * R * d * 4);
        // Model._moe_scratch(("moe", top-k + 1, the stages' experts, decode)): one decode scratch of their own
        if (e.mx != null) {
            const ex = b0.experts;
            const es = mx.ExpertScratch.sizes(R, dr.slots_moe, ex.count, ex.dims, ex.down_k, ex.down_n);
            dr.mxe.rows = R;
            dr.mxe.slots = dr.slots_moe;
            dr.mxe.nq = 0; // the stages' experts are MXFP4 in every checkpoint
            dr.mxe.ns = 0;
            dr.mxe.hnq = 0;
            dr.mxe.hns = 0;
            inline for (.{ "q", "s", "cnt", "off", "fill", "act", "nact", "ent", "erow", "hq", "hs", "out" }, 0..) |f, j| {
                @field(dr.mxe, f) = try a.take(es[j]);
            }
            // its decode launches' split partials (mx.decodeZFloats: the stages' experts, router and wo_a)
            var zf: usize = 0;
            for (ds.blocks) |blk| zf = @max(zf, mx.decodeZFloats(blk, c.hidden, ne, c.index_heads, R, dr.slots_moe));
            dr.mxe.z = .{ .ptr = try a.take(zf * 4), .floats = zf };
            dr.nxh = try a.take(R * dr.slots_moe * ex.width * 2);
            dr.xsd = std.mem.zeroes(exl3_experts.DecodeScratch);
            dr.xs = std.mem.zeroes(exl3_experts.Scratch);
        } else dr.xsd = try prompt.decodeScratch(e, a, b0.experts, dr.slots_moe);
        if (e.mx == null) {
            const ex = b0.experts;
            const sz = exl3_experts.Scratch.sizes(R, dr.slots_moe, ex.dims, ex.width, ex.count);
            dr.xs.rows = R;
            dr.xs.slots = dr.slots_moe;
            inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
                @field(dr.xs, f) = try a.take(sz[j]);
            }
        }
        dr.xc = try a.take(R * d * 2);
        dr.xn = try a.take(R * d * 2);
        dr.local = try a.take(R * e.w.head.n * 4);
        // the Markov loop: every token's slot -1 (none), then the cached rows of the first 256 frequent tokens
        const none = try a.take(c.vocab * 4);
        try e.d.check(e.d.api.cuMemsetD32Async(none, 0xffffffff, c.vocab, e.s.handle), "cuMemsetD32Async");
        // the Markov loop's kernels read the head as fp16 (the EXL3 lane's checkpoint); DeepSeek's own holds it in bf16
        var mhead = ds.markov_head;
        if (e.mx) |mk| {
            const n = c.vocab * c.markov_rank;
            mhead = try a.take(n * 2);
            try mx.bf16ToF16(mk, e.s, ds.markov_head, mhead, n);
        }
        dr.mk = if (e.two) |tw| try draft2d.markov(c.*, tw, mhead, ds.markov_embed, none) else try tri_markov.Markov.init(c.*, sp, mhead, ds.markov_embed, none);
        const k = markov_tokens.cached.len;
        const tok = try a.take(k * 8);
        try prompt.upload(e, tok, &markov_tokens.cached, k * 8);
        const cache = try a.take(k * dr.mk.cols * 2);
        const slot = try a.take(c.vocab * 4);
        {
            const host = try std.heap.page_allocator.alloc(i32, c.vocab);
            defer std.heap.page_allocator.free(host);
            @memset(host, -1);
            for (markov_tokens.cached, 0..) |t, j| host[@intCast(t)] = @intCast(j);
            try prompt.upload(e, slot, host.ptr, c.vocab * 4);
            try e.s.synchronize(); // (the host table is freed next)
        }
        try dr.mk.fill(e.t, tok, k, cache, slot);
        dr.msc = .{
            .pv = try a.take(max_streams * dr.mk.n_part * 4),
            .pi = try a.take(max_streams * dr.mk.n_part * 4),
            .send = try a.take(max_streams * 4 * 4),
            .stage = try a.take(max_streams * dr.mk.cols * 2),
        };
        try prompt.fill(e, dr.msc.send, 0, max_streams * 4 * 4);
        dr.mg = try a.take(dr.mk.world * max_streams * 4 * 4);

        dr.out = try a.take(max_streams * (max_block + 1) * 8);
        // the confidence head
        const rank = c.markov_rank;
        dr.embs = try a.take(R * rank * 2);
        dr.xcf = try a.take(R * d * 4);
        dr.ef = try a.take(R * rank * 4);
        dr.conf_in = try a.take(R * (d + rank) * 4);
        dr.confw = try a.take((d + rank) * 4);
        // the confidence head's fp32 weights: the EXL3 lane's checkpoint holds them in fp16, DeepSeek's own in bf16,
        // widened to fp32 when loaded (plan.zig)
        if (e.mx != null) {
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dr.confw, ds.conf, (d + rank) * 4, e.s.handle), "cuMemcpyDtoDAsync");
        } else try e.ops.f16ToF32(e.s, ds.conf, dr.confw, d + rank);
        dr.conf = try a.take(R * 4);
        // the confidence head runs on the host after the pass's one read (cuBLAS's 5 x 5376 fp32 product took ~2 ms of
        // a ~11 ms pass on a GB10): its weight read back once
        dr.confw_host = &.{};
        dr.xc_host = &.{};
        dr.em_host = &.{};
        errdefer dr.deinit();
        dr.confw_host = try std.heap.page_allocator.alloc(f32, d + rank);
        dr.xc_host = try std.heap.page_allocator.alloc(u16, R * d);
        dr.em_host = try std.heap.page_allocator.alloc(u16, R * rank);
        try e.s.synchronize();
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(dr.confw_host.ptr, dr.confw, (d + rank) * 4), "cuMemcpyDtoH");
        // absorbs
        const ring = c.window + ring_extra;
        dr.at = try a.take(ring * dr.taps_w * 2);
        dr.ay = try a.take(ring * d * 2);
        dr.ax = try a.take(ring * d * 2);
        dr.ak = @splat(0);
        for (0..ds.blocks.len) |j| dr.ak[j] = try a.take(ring * hd * 2);
        dr.ao = try a.take(ring * hd * 2);
        dr.apos = try a.take(ring * 8);
        dr.arows = try a.take(ring * 8);
        return dr;
    }

    /// The host buffers (the device ones are the arena's).
    pub fn deinit(dr: *Drafter) void {
        const pa = std.heap.page_allocator;
        if (dr.confw_host.len > 0) pa.free(dr.confw_host);
        if (dr.xc_host.len > 0) pa.free(dr.xc_host);
        if (dr.em_host.len > 0) pa.free(dr.em_host);
        dr.confw_host = &.{};
        dr.xc_host = &.{};
        dr.em_host = &.{};
    }

    fn row(dr: *const Drafter, i: usize) u64 {
        return dr.rows + i * max_rows * 8;
    }

    /// Drafter.absorb into slot `slot`'s view: the target's taps rows of positions start .. start + n - 1 (bf16
    /// [n, taps * D], contiguous) into every stage's ring, the last ring rows of them: main_x = RMSNorm(main_proj @
    /// taps), each stage's wkv of it normed and roped into its ring row (position % ring).
    pub fn absorb(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *Pool, slot: usize, taps: u64, n: usize, start: usize) !void {
        if (n == 0 or slot >= pool.slots) return error.BadAbsorb;
        const keep = @min(n, pool.ring);
        if (keep > 128) return notPorted(@src()); // main_proj's prompt GEMM (the served lane's replay keeps 128 rows)
        const first = start + n - keep;
        var pos: [max_absorb * 9]i64 = undefined;
        var rows: [max_absorb * 9]i64 = undefined;
        for (0..keep) |r| {
            pos[r] = @intCast(first + r);
            rows[r] = @intCast((first + r) % pool.ring);
        }
        try prompt.upload(e, dr.apos, &pos, keep * 8);
        try prompt.upload(e, dr.arows, &rows, keep * 8);
        ddTaps(e, dr, taps + (n - keep) * dr.taps_w * 2, pos[0..keep]);
        try dr.mainKeys(e, ch, taps + (n - keep) * dr.taps_w * 2, keep);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.view(j, slot), pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, keep, e.c.head_dim);
        }
        pool.absorbed[slot] = start + n;
    }

    /// Drafter.absorb_many: every item's rows in one pass over the pool's planes (ring row slot * ring + position %
    /// ring); the caller sets each slot's `absorbed` (eager: the kept length, known after sampling).
    pub fn absorbMany(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, items: []const Item) !void {
        var n: usize = 0;
        var pos: [max_absorb]i64 = undefined;
        var rows: [max_absorb]i64 = undefined;
        for (items) |it| {
            if (it.slot >= pool.slots or n + it.n > max_absorb) return error.BadAbsorb;
            try e.ops.copyRows(e.s, it.taps, dr.taps_w * 2, dr.at + n * dr.taps_w * 2, dr.taps_w * 2, dr.taps_w * 2, it.n);
            for (0..it.n) |r| {
                const p = it.start + r;
                pos[n + r] = @intCast(p);
                rows[n + r] = @intCast(it.slot * pool.ring + p % pool.ring);
            }
            n += it.n;
        }
        if (n == 0) return;
        try prompt.upload(e, dr.apos, &pos, n * 8);
        try prompt.upload(e, dr.arows, &rows, n * 8);
        ddTaps(e, dr, dr.at, pos[0..n]);
        try dr.mainKeys(e, ch, dr.at, n);
        for (dr.ds.blocks, 0..) |lay, j| {
            try tri_norm.kvNormRope(e.t, dr.ak[j], lay.kv_norm, e.plain.cos, e.plain.sin, dr.apos, pool.rings[j], pool.slots * pool.ring, dr.arows, e.c.eps, true, e.c.rope_dim, dr.ao, n, e.c.head_dim);
        }
    }

    /// main_x = RMSNorm(main_proj @ taps) of n rows, then every stage's wkv of it (stage_kv: one group).
    fn mainKeys(dr: *Drafter, e: *const Engine, ch: *const Chunk, taps: u64, n: usize) !void {
        const c = e.c;
        const d = c.hidden;
        const hd = c.head_dim;
        // (decode-size absorbs, a round's window rows: the GEMV; a prompt chunk's: the GEMM)
        try prompt.groupedDecode(e, ch, n, &.{dr.ds.main_proj}, &.{taps}, &.{dr.taps_w}, &.{dr.ay}, &.{d}, &.{.bf16});
        try tri_basic.rmsnorm(e.t, dr.ay, d, dr.ds.main_norm, dr.ax, d, c.eps, n, d);
        const S = dr.ds.blocks.len;
        var ls: [max_stages]weights.Linear = undefined;
        var xs: [max_stages]u64 = undefined;
        var ldxs: [max_stages]usize = undefined;
        var ldos: [max_stages]usize = undefined;
        var types: [max_stages]exl3_linear.DType = undefined;
        for (dr.ds.blocks, 0..) |lay, j| {
            ls[j] = lay.wkv;
            xs[j] = dr.ax;
            ldxs[j] = d;
            ldos[j] = hd;
            types[j] = .bf16;
        }
        try prompt.groupedDecode(e, ch, n, ls[0..S], xs[0..S], ldxs[0..S], dr.ak[0..S], ldos[0..S], types[0..S]);
    }

    /// BatchDraftGraph.run for N streams (tokens at positions q0 in slots): each stream's `steps` drafts and their
    /// confidences into drafts / confs. Each slot's rings hold its positions before q0 (absorbed).
    pub fn pass(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, probe: ?Probe) !void {
        const N = tokens.len;
        const n = dr.n;
        const R = N * n;
        if (N == 0 or N > max_streams or R > max_rows or q0.len != N or slots.len != N) return notPorted(@src());
        if (steps == 0 or steps > n) return error.BadDraftPass;
        // _body's index rows: ids [token, noise ...], positions q0 + j, window positions q0 - 1, ring bases
        var hv: [6 * max_rows]i64 = @splat(0);
        var hb: [max_rows * pick_pad]i64 = @splat(-1);
        const bw: usize = if (R > tri.decode_rows) pick_pad else n; // the pick list's columns
        for (0..N) |i| {
            if (slots[i] < 0 or slots[i] >= pool.slots) return error.BadSlot;
            for (0..n) |j| {
                const r = i * n + j;
                hv[0 * max_rows + r] = if (j == 0) tokens[i] else dr.noise;
                hv[1 * max_rows + r] = q0[i] + @as(i64, @intCast(j));
                hv[2 * max_rows + r] = q0[i] - 1;
                hv[3 * max_rows + r] = slots[i] * @as(i64, @intCast(pool.ring));
                hv[4 * max_rows + r] = 0;
                hv[5 * max_rows + r] = -1;
                for (0..n) |k| hb[r * bw + k] = @intCast(i * n + k);
            }
        }
        try prompt.upload(e, dr.rows, &hv, hv.len * 8);
        try prompt.upload(e, dr.bidx, &hb, R * bw * 8);
        // the Markov loop's out [N, steps + 1]: each stream's last token first
        const ts = steps + 1;
        var ho: [max_streams * (max_block + 1)]i64 = @splat(0);
        for (0..N) |i| ho[i * ts] = tokens[i];
        try prompt.upload(e, dr.out, &ho, N * ts * 8);
        var dprobe: DumpProbe = .{ .e = e, .dr = dr, .pool = pool, .R = R, .N = N, .steps = steps };
        const dumping = ddump != null and ddump_count == ddump_pass and probe == null;
        if (ddump != null) ddump_count += 1;
        if (dumping) {
            ddump_on = true;
            defer ddump_on = false;
            dd(e, "rows", 0, dr.rows, hv.len * 8);
            try dr.body(e, ch, pool, N, steps, .{ .ctx = &dprobe, .at = DumpProbe.at });
            dd(e, "out", 0, dr.out, N * ts * 8);
            if (ddump_node == 0) {
                var mb: [512]u8 = undefined;
                const path = std.mem.printSentinel(&mb, "{s}/meta.txt", .{ddump.?}, 0) catch unreachable;
                if (std.c.fopen(path.ptr, "wb")) |f| {
                    defer _ = std.c.fclose(f);
                    var tb: [1024]u8 = undefined;
                    const s = std.fmt.bufPrint(&tb, "N {d}\nsteps {d}\nn {d}\nslots {d}\nring {d}\nworld {d}\nhead_n {d}\nvocab_lo {d}\ntoken {d}\nq0 {d}\nslot {d}\n", .{ N, steps, n, pool.slots, pool.ring, e.world, e.w.head.n, e.w.vocab_lo, tokens[0], q0[0], slots[0] }) catch unreachable;
                    _ = std.c.fwrite(s.ptr, 1, s.len, f);
                }
            }
        } else if (dprof and probe == null) {
            pmark(e, .start);
            try dr.body(e, ch, pool, N, steps, null);
            prof_passes += 1;
        } else if (dr.graphs != null and probe == null) try dr.graphs.?.run(dr, e, ch, pool, N, steps) else try dr.body(e, ch, pool, N, steps, probe);
        // one host read: the drafts and the confidences
        try e.s.synchronize();
        var hc: [max_streams * max_block]f32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&ho, dr.out, N * ts * 8), "cuMemcpyDtoH");
        try dr.confidence(e, N, steps, &hc);
        if (dumping) ddHost("conf", std.mem.sliceAsBytes(hc[0 .. N * steps]));
        for (0..N) |i| for (0..steps) |j| {
            dr.drafts[i][j] = ho[i * ts + 1 + j];
            dr.confs[i][j] = hc[i * steps + j];
        };
        dr.streams = N;
        dr.steps = steps;
        if (dlog) |path| if (ddump_node == 0) {
            // TF_DS_DRAFT_LOG: a line a stream of the pass: slot q0 token, its drafts, its confidences
            if (std.c.fopen(path.ptr, "ab")) |f| {
                defer _ = std.c.fclose(f);
                for (0..N) |i| {
                    var lb: [512]u8 = undefined;
                    var at: usize = 0;
                    at += (std.fmt.bufPrint(lb[at..], "{d} {d} {d}", .{ slots[i], q0[i], tokens[i] }) catch break).len;
                    for (0..steps) |j| at += (std.fmt.bufPrint(lb[at..], " {d}", .{dr.drafts[i][j]}) catch break).len;
                    for (0..steps) |j| at += (std.fmt.bufPrint(lb[at..], " {d:.4}", .{dr.confs[i][j]}) catch break).len;
                    lb[at] = '\n';
                    _ = std.c.fwrite(&lb, 1, at + 1, f);
                }
            }
        };
    }

    /// DSparkConfidenceHead on the host: conf[i * steps + j] = [xc row (i, j), the Markov embedding row of the token step
    /// j read] . w, fp32 products summed in fp64 (the collapsed rows and the embedding rows read back as bf16).
    fn confidence(dr: *const Drafter, e: *const Engine, N: usize, steps: usize, out: []f32) !void {
        const d = e.c.hidden;
        const rank = e.c.markov_rank;
        const n = dr.n;
        const xc = dr.xc_host;
        const em = dr.em_host;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(xc.ptr, dr.xc, N * n * d * 2), "cuMemcpyDtoH");
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(em.ptr, dr.embs, N * steps * rank * 2), "cuMemcpyDtoH");
        const wv = dr.confw_host;
        for (0..N) |i| for (0..steps) |j| {
            var acc: f64 = 0;
            const xr = xc[(i * n + j) * d ..][0..d];
            for (xr, wv[0..d]) |x, wk| acc += @as(f64, bf(x) * wk);
            const er = em[(i * steps + j) * rank ..][0..rank];
            for (er, wv[d..][0..rank]) |x, wk| acc += @as(f64, bf(x) * wk);
            out[i * steps + j] = @floatCast(acc);
        };
    }

    /// The pass's device work for N streams of `steps` drafts, from the rows' embeddings to the confidence head (its
    /// inputs uploaded: the index rows, the pick lists, the Markov loop's first column): what a pass graph captures.
    fn body(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, N: usize, steps: usize, probe: ?Probe) !void {
        const c = e.c;
        const t = e.t;
        const n = dr.n;
        const R = N * n;
        const d = c.hidden;
        // the streams: every row's token embedding in each stream, pre (1, 0, 0, 0)
        try tri_basic.embedInit(t, e.w.embed, dr.row(0), dr.h, dr.pre, R, d, c.hc);
        pmark(e, .embed);
        // _stages_fused
        var h = dr.h;
        var spare = dr.h_alt;
        var pre = dr.pre;
        var pre_f = dr.pre_f;
        var pending: ?u64 = null;
        for (dr.ds.blocks, 0..) |lay, j| {
            h = try dr.mix(e, h, &spare, &pending, lay.hc_attn, pre, lay.attn_norm, dr.pre_a, R);
            pmark(e, .mix_a);
            dd(e, "h_a", j, h, R * c.hc * d * 2);
            dd(e, "pre_in", j, pre, R * c.hc * 4);
            dd(e, "pre_a", j, dr.pre_a, R * c.hc * 4);
            try Probe.check(probe, .attn_in, j, dr.x);
            try Probe.check(probe, .attn_ring, j, pool.rings[j]);
            try dr.attention(e, ch, pool, j, R);
            try Probe.check(probe, .attn_out, j, dr.pa);
            try pairGather(e, dr.pa, dr.ga, R, d);
            pmark(e, if (j == 0) .gather_a0 else .gather_a);
            try Probe.check(probe, .attn_gather, j, dr.ga);
            pending = dr.ga;
            h = try dr.mix(e, h, &spare, &pending, lay.hc_ffn, dr.pre_a, lay.ffn_norm, pre_f, R);
            pmark(e, .mix_f);
            dd(e, "h_f", j, h, R * c.hc * d * 2);
            dd(e, "pre_f", j, pre_f, R * c.hc * 4);
            try Probe.check(probe, .moe_in, j, dr.x);
            try dr.moe(e, j, R);
            try Probe.check(probe, .moe_out, j, dr.pm);
            try pairGather(e, dr.pm, dr.gm, R, d);
            pmark(e, if (j == 0) .gather_m0 else .gather_m);
            try Probe.check(probe, .moe_gather, j, dr.gm);
            pending = dr.gm;
            std.mem.swap(u64, &pre, &pre_f);
        }
        try tri_basic.hcPost(t, dr.gm, h, dr.post, dr.comb, h, e.world, R, d);
        pmark(e, .post);
        const S = dr.ds.blocks.len;
        try Probe.check(probe, .stages_h, S, h);
        try Probe.check(probe, .stages_pre, S, pre);
        // the head on every row (this rank's columns: the Markov loop scores them without a gather)
        try tri_basic.collapse(t, h, pre, dr.xc, R, d);
        try tri_basic.rmsnorm(t, dr.xc, d, dr.ds.norm, dr.xn, d, c.eps, R, d);
        dd(e, "xc", 0, dr.xc, R * d * 2);
        dd(e, "xn", 0, dr.xn, R * d * 2);
        const head = dr.ds.head8 orelse e.w.head; // the drafter's FP8 copy (weights.DSpark.head8), else the backbone's
        try prompt.groupedDecode(e, ch, R, &.{head}, &.{dr.xn}, &.{d}, &.{dr.local}, &.{head.n}, &.{.f32});
        try Probe.check(probe, .local, S, dr.local);
        pmark(e, .head);
        // the Markov loop: out [N, steps + 1], each stream's last token first
        const ts = steps + 1;
        var g: MarkovGather = .{ .e = e, .dst = dr.mg };
        try dr.mk.steps(t, dr.local, dr.out, ts, N, n, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run });
        try Probe.check(probe, .markov, S, dr.out);
        pmark(e, .markov);
        // the confidence head: [xc (fp32), the embedding rows the steps read (fp32)] @ conf^T, a stream's first steps rows
        const rank = c.markov_rank;
        const w = d + rank;
        try e.d.check(e.d.api.cuMemsetD8Async(ch.invalid, 0, 4, e.s.handle), "cuMemsetD8Async");
        // (the product itself on the host: pass)
        _ = w;
        for (0..N) |i| {
            try e.ops.gatherRows(e.s, dr.ds.markov_embed, c.vocab, dr.out + i * ts * 8, dr.embs + i * steps * rank * 2, rank * 2, steps, ch.invalid);
        }
        pmark(e, .conf);
    }

    /// pre_mix of _stages_fused: hc_pre2 of the streams h with the pending gathered partials posted in (into the spare
    /// buffer, which becomes the streams; no side stream in the drafter); returns the streams.
    fn mix(dr: *Drafter, e: *const Engine, h: u64, spare: *u64, pending: *?u64, params: [3]u64, pre_in: u64, norm: u64, pre_out: u64, R: usize) !u64 {
        const c = e.c;
        if (pending.*) |gp| {
            const out = try tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, .{ .gathered = gp, .world = e.world, .h_out = spare.* }, null, R, c.hidden);
            pending.* = null;
            spare.* = h;
            return out;
        }
        return tri_hc.hcPre2(e.t, h, params[0], params[1], params[2], pre_in, norm, c.eps, c.hc_eps, c.hc_iters, dr.x, pre_out, dr.post, dr.comb, dr.part, null, null, R, c.hidden);
    }

    /// BatchDraftGraph._attention (fused): stage j's attention partial pa [R, D] fp32 from its mixed rows x.
    fn attention(dr: *Drafter, e: *const Engine, ch: *const Chunk, pool: *const Pool, j: usize, R: usize) !void {
        const c = e.c;
        const t = e.t;
        const lay = dr.ds.blocks[j];
        const hd = c.head_dim;
        const rd = c.rope_dim;
        const hl = lay.wq_b.n / hd; // the block's heads (TP2's on a 2D node too)
        const rope = e.plain; // a window-only stage (Model._cs: ratio 0)
        const pos = dr.row(1);
        // attn_in(comp=False): wq_a and wkv, one group
        try prompt.groupedDecode(e, ch, R, &.{ lay.wq_a, lay.wkv }, &.{ dr.x, dr.x }, &.{ c.hidden, c.hidden }, &.{ dr.qa, dr.ykv }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
        pmark(e, .qkv_a);
        if (e.mx != null) {
            // Format.native: q's RMSNorm as plain rows, wq_b on them, q's RoPE
            try tri_basic.rmsnorm(t, dr.qa, lay.wq_a.n, lay.q_norm, dr.qr, lay.wq_a.n, c.eps, R, lay.wq_a.n);
            try prompt.groupedDecode(e, ch, R, &.{lay.wq_b}, &.{dr.qr}, &.{lay.wq_a.n}, &.{dr.q}, &.{lay.wq_b.n}, &.{.bf16});
            try tri_norm.ropeHeads(t, dr.q, rope.cos, rope.sin, pos, rd, false, R, hl, hd);
            pmark(e, .q_b);
            dd(e, "qa", j, dr.qa, R * lay.wq_a.n * 2);
            dd(e, "ykv", j, dr.ykv, R * hd * 2);
            dd(e, "qr", j, dr.qr, R * lay.wq_a.n * 2);
            dd(e, "q", j, dr.q, R * hl * hd * 2);
        } else {
            // q_proj with rope (rot_q): q's RMSNorm writes wq_b's rotated rows; wq_b's epilogue applies q's RoPE
            const rot = [_]tri_norm.Rot{.{ .suh = lay.wq_b.suh, .h = dr.xhq }};
            try tri_norm.rmsnormRot(t, dr.qa, lay.wq_a.n, lay.q_norm, c.eps, &rot, dr.qr, lay.wq_a.n, R, lay.wq_a.n);
            var qc = [_]exl3_linear.Call{.{ .layer = lay.wq_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xhq, .y = dr.q, .ldy = @intCast(lay.wq_b.n), .y_dtype = .bf16, .counters = 0, .rope = .{ .cos = rope.cos, .sin = rope.sin, .pos = pos, .hd = @intCast(hd), .rd = @intCast(rd) } }};
            try prompt.groupedRotated(e, ch, R, &qc);
        }
        // the block rows' keys: normed and roped, stored nowhere (ring row -1)
        const plane_rows = pool.slots * pool.ring;
        try tri_norm.kvNormRope(t, dr.ykv, lay.kv_norm, rope.cos, rope.sin, pos, pool.rings[j], plane_rows, dr.row(5), c.eps, true, rd, dr.kvb, R, hd);
        pmark(e, .kv_rope);
        // each row sees its stream's 128 newest absorbed positions (to q0 - 1) and every block row of its stream
        try tri_attn.sparseAttn(t, .{
            .q = dr.q,
            .out = dr.o,
            .rows = R,
            .h = hl,
            .hd = hd,
            .sink = lay.sink,
            .wsrc = pool.rings[j],
            .wsrc_rows = plane_rows,
            .wlo = dr.wlo,
            .ring = true,
            .comp = .{ .bf16 = dr.kvb },
            .idx = dr.bidx,
            .n_idx = dr.n,
            .pos = dr.row(2),
            .scale = prompt.scale(hd),
            .window = c.window,
            .wbase = dr.row(3),
            .cbase = dr.row(4),
            .ring_rows = pool.ring,
            .parts = .{ .pm = dr.pm_, .pl = dr.pl_, .po = dr.po_ },
        });
        pmark(e, .sattn);
        dd(e, "kvb", j, dr.kvb, R * hd * 2);
        dd(e, "o_rope", j, dr.o, R * hl * hd * 2);
        try tri_norm.ropeHeads(t, dr.o, rope.cos, rope.sin, pos, rd, true, R, hl, hd);
        dd(e, "o", j, dr.o, R * hl * hd * 2);
        const uw = lay.groups * lay.wo_a[0].n;
        if (e.mx) |k| {
            // Format.native: wo_a's groups (o quant-dequantized, bf16 einsum; the FP8 rows (weights.draftWoAFp8): each
            // group's act_quant(o) through its FP8 GEMV, the same products), then wo_b (fp32)
            if (if (k.dec_on) lay.wo_a8 else null) |w8| {
                // the decode kernels' FP8 copy of wo_a (weights.zig wo_a8): one launch for the groups
                try mx.woA8(k, e.s, ch.mxs, w8, lay.groups, dr.o, hl * hd, R, dr.u, uw, dr.mxe.z);
            } else if (lay.wo_a[0].fmt == .fp8) {
                const gk: usize = lay.wo_a[0].k;
                for (lay.wo_a[0..lay.groups], 0..) |wo, g| try mx.linear(k, e.s, ch.mxs, wo, dr.o + g * gk * 2, .bf16, hl * hd, R, dr.u + g * wo.n * 2, uw, .bf16, false, dr.mxe.z);
            } else try mx.woA(k, e.s, ch.mxs, lay.wo_a[0..lay.groups], dr.o, hl * hd, R, dr.u, uw, dr.mxe.z);
            dd(e, "u", j, dr.u, R * uw * 2);
            pmark(e, .wo_a);
            try prompt.groupedDecode(e, ch, R, &.{lay.wo_b}, &.{dr.u}, &.{uw}, &.{dr.pa}, &.{lay.wo_b.n}, &.{.f32});
            pmark(e, .wo_b);
            return;
        }
        // wo_ab: wo_a's inputs rotated from o's column blocks, its epilogue writing wo_b's rotated rows; wo_b (fp32)
        var calls: [exl3_linear.gmax]exl3_linear.Call = undefined;
        const gw = hl * hd / lay.groups;
        var col: usize = 0;
        var off: usize = 0;
        for (lay.wo_a[0..lay.groups], 0..) |wo, gi| {
            calls[gi] = .{ .layer = wo, .x = dr.o + gi * gw * 2, .ldx = @intCast(hl * hd), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = dr.u + col * 2, .ldy = @intCast(uw), .y_dtype = .bf16, .counters = try prompt.counterOf(ch, wo), .rot = .{ .suh = lay.wo_b.suh, .xh = dr.xb, .ldr = @intCast(lay.wo_b.k), .off = @intCast(col) } };
            col += wo.n;
            off += R * wo.k;
        }
        try exl3_linear.rotMany(e.lin, e.s, calls[0..lay.groups], R, true);
        try exl3_linear.glinear(e.lin, e.s, calls[0..lay.groups], R, ch.gz, true, true);
        var cb = [_]exl3_linear.Call{.{ .layer = lay.wo_b, .x = 0, .ldx = 0, .x_dtype = .bf16, .xh = dr.xb, .y = dr.pa, .ldy = @intCast(lay.wo_b.n), .y_dtype = .f32, .counters = 0 }};
        try prompt.groupedRotated(e, ch, R, &cb);
    }

    /// Model.moe(lay, x, topk) of a pass's rows: the gate's chunk sums (rowmm_gate; past decode_rows rows the gate's fp32
    /// matmul), the routing at the stages' top-k with the shared expert in every row's last slot, the routed experts'
    /// decode path on the stages' own scratch.
    fn moe(dr: *Drafter, e: *const Engine, j: usize, R: usize) !void {
        const c = e.c;
        const lay = dr.ds.blocks[j];
        const ne: usize = lay.experts.count - 1;
        if (e.mx) |k| {
            // Format.native: Gate.forward's fp32 logits, the routing at the stages' top-k, the experts
            try mx.bf16Gemm(k, e.s, dr.x, c.hidden, lay.gate_w, c.hidden, R, ne, c.hidden, dr.gl, ne, .f32, 1, false, 1, 0, 0, 0, dr.mxe.z);
            try tri_norm.route(e.t, dr.gl, 0, lay.gate_b, dr.topk, c.routed_scaling, lay.experts.count - 1, dr.pick, dr.mw, R, ne, dr.slots_moe);
            dd(e, "gl", j, dr.gl, R * ne * 4);
            dd(e, "pick", j, dr.pick, R * dr.slots_moe * 4);
            dd(e, "mw", j, dr.mw, R * dr.slots_moe * 4);
            pmark(e, .gate);
            try mx.gateUp(k, e.s, e.d, lay.experts, dr.mxe, dr.x, c.hidden, dr.pick, dr.mw, R, c.swiglu_limit, dr.nxh, lay.experts.width, true);
            pmark(e, .experts_gu);
            try mx.down(k, e.s, lay.experts, dr.mxe, dr.nxh, lay.experts.width, dr.pick, R, dr.pm, c.hidden, false, true);
            pmark(e, .experts_d);
            return;
        }
        var kc: usize = 0;
        if (R <= tri.decode_rows) {
            kc = try tri_norm.rowmmGate(e.t, dr.x, c.hidden, lay.gate_w, dr.gl, R, c.hidden, ne);
        } else {
            // x.float() @ gate_w.float().t() through cuBLAS, then the plain routing
            try e.ops.toF32(e.s, dr.x, dr.xf, R * c.hidden);
            try e.blas.xwT(dr.xf, dr.gate_f[j], dr.gl, R, c.hidden, ne);
        }
        try tri_norm.route(e.t, dr.gl, kc, lay.gate_b, dr.topk, c.routed_scaling, lay.experts.count - 1, dr.pick, dr.mw, R, ne, dr.slots_moe);
        // routed(): fewer than EXACT_ROWS rows take the decode window's fused path, more the chunk's (13+ streams)
        if (R < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, dr.xsd, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
        try exl3_experts.prompt(e.ex, e.s, lay.experts, dr.xs, dr.x, c.hidden, dr.pick, dr.mw, dr.pm, R, c.swiglu_limit);
    }
};

/// MultiDecoder._depth: the most drafts a stream verifies in a round of `live` streams (TF_DS_PARALLEL_DEPTH unset:
/// 5, 5, 3, 3), at most the engine's drafts and the rows a stream may take of a round (max_rows / live - 1).
pub fn depth(live: usize, drafts: usize, round_rows: usize) usize {
    const by = [_]usize{ 5, 5, 3, 3 };
    const k = by[@min(live, by.len) - 1];
    return @min(@min(k, drafts), (round_rows / @max(live, 1)) -| 1);
}

/// The batched pass's Markov steps for N drafting streams (BatchDraftGraph's `steps` under the served "conf" policy):
/// as deep as any round with N or more live streams may verify.
pub fn passSteps(n_streams: usize, slots: usize, drafts: usize, round_rows: usize, block: usize) usize {
    var s: usize = 0;
    var live = n_streams;
    while (live <= slots) : (live += 1) s = @max(s, depth(live, drafts, round_rows));
    return @max(1, @min(block, s));
}

/// The served ROUND_MS table (TF_DS_ROUND_MS unset): a round's forward ms by its rows, and DRAFT_MS.
pub const round_ms = [_]f64{ 30.0, 35.2, 39.9, 44.9, 48.6, 52.1, 55.6, 59.1, 62.7, 66.3, 69.6, 72.9, 75.5, 78.2, 81.3, 84.4, 87.5, 90.6, 93.7, 96.8, 99.9, 103.0, 106.1, 109.2 };
pub const draft_ms: f64 = 3.5;

extern "c" fn exp(x: f64) f64;

/// TF_DS_CONF_CAL="a,b": a draft's landing chance (once the earlier drafts landed) is sigmoid(a conf + b) (a 1, b 0:
/// the served rule, the confidence's own sigmoid). Unset: a 1, b 0.19, the logistic fit of 399 logged rounds of the
/// four-node native lane (prose and code, all five drafts verified; the head is well calibrated, a shade low).
pub var conf_cal: [2]f64 = .{ 1.0, 0.19 };
/// TF_DS_DRAFT_MS: a round's ms past its forward that chooseK adds (unset: the four-node native lane's measured pass
/// and absorb, ~6.0 + 0.55 ms after this work; the served EXL3 lane's DRAFT_MS was 3.5).
pub var cost_ms: f64 = 6.6;

/// A draft's landing chance from its confidence (conf_cal).
pub fn landing(c: f32) f64 {
    return 1.0 / (1.0 + exp(-(conf_cal[0] * @as(f64, c) + conf_cal[1])));
}

/// MultiDecoder._choose_k: the k in 0 .. kmax with the most expected tokens a millisecond (1 + the prefix survivals'
/// sum, survival the sigmoid of each confidence, over the round's table cost with the other streams at their table
/// depth), in Python's float64 arithmetic (libm's exp, as Python's math.exp calls it); the first best wins.
pub fn chooseK(conf: []const f32, kmax: usize, live: usize, drafts: usize, round_rows: usize) usize {
    return chooseKWith(conf, kmax, live, drafts, round_rows, &round_ms, draft_ms, 0.0);
}

/// chooseK over a round table by rows (`table`, the lane's TF_DS_ROUND_MS) and a round's ms past its forward `dms`,
/// each draft's survival its landing chance (conf_cal). draft_row_ms > 0 (A3, TF_DS_DRAFT_ROW_MS): the forward is the
/// table at the live streams plus draft_row_ms for every row past them, as the lane core's allocation prices it.
pub fn chooseKWith(conf: []const f32, kmax: usize, live: usize, drafts: usize, round_rows: usize, table: []const f64, dms: f64, draft_row_ms: f64) usize {
    const others = (live - 1) * (depth(live, drafts, round_rows) + 1);
    var best: usize = 0;
    var best_v: f64 = -1.0;
    var surv: f64 = 1.0;
    var exp_tokens: f64 = 1.0;
    for (0..kmax + 1) |kk| {
        if (kk > 0) {
            surv *= landing(conf[kk - 1]);
            exp_tokens += surv;
        }
        const rows = @min(table.len, others + kk + 1);
        const fwd = if (draft_row_ms > 0.0)
            table[@min(table.len, live) - 1] + draft_row_ms * @as(f64, @floatFromInt(others + kk + 1 - @min(others + kk + 1, live)))
        else
            table[rows - 1];
        const v = exp_tokens * @as(f64, @floatFromInt(live)) / (fwd + dms);
        if (v > best_v) {
            best = kk;
            best_v = v;
        }
    }
    return best;
}

test "draft depths and pass steps as multi.py takes them" {
    // 16-row rounds, 5 drafts: 5 alone or beside one other stream, 3 at three or four
    try std.testing.expectEqual(@as(usize, 5), depth(1, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), depth(2, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(3, 5, 16));
    try std.testing.expectEqual(@as(usize, 3), depth(4, 5, 16));
    try std.testing.expectEqual(@as(usize, 5), passSteps(1, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 5), passSteps(2, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(3, 4, 5, 16, 5));
    try std.testing.expectEqual(@as(usize, 3), passSteps(4, 4, 5, 16, 5));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}
