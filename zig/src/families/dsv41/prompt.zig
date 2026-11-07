//! A prompt chunk's forward on one TP rank, the served build's Model._forward_k (model.py) step by step on the served
//! build's kernels: the streams of the chunk's tokens (embed_init), then a layer at a time the attention mixes (hc_pre, or
//! hc_pre_pf with the previous layer's gathered MoE partials posted first), attention (attention_k), the gather of its
//! partials, the FFN mixes with that post fused in (hc_pre_pf), the MoE and its gather, whose post goes into the next
//! layer's mixes (switch "hc_pf2"); before an Engram layer's mixes, its Engram (engram_apply). Compressed layers
//! (ratio 1, 2): a kv source's compressor and index keys into the caches (kv_source_update), the indexer's top-k
//! (prompt_keys: scored keys and the selection; from the candidate source on: fp32 scores and topk_indices), and the
//! sparse attention over the window (a later chunk's first keys from the ring) and the selected latents. Decode-sized
//! rows (1..128) take the grouped EXL3 linears. With replay (bounded replay prefill) the decoder's first layer cuts the
//! chunk to its rows at or past replay (replayCut); DSpark taps (tap) and the head on the last row (head) end it.
//! Not yet: ring mode (verify windows), a pool that leaves blocks out, a chunk starting inside a compressor group;
//! those return error.NotPortedYet.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const weights = @import("weights.zig");
const tri = @import("tri.zig");
const tri_basic = @import("tri_basic.zig");
const tri_hc = @import("tri_hc.zig");
const tri_norm = @import("tri_norm.zig");
const tri_attn = @import("tri_attn.zig");
const tri_markov = @import("tri_markov.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ops = @import("ops.zig");
const cublas = @import("cublas.zig");
const comm = @import("comm.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_rows = @import("engram_rows.zig");
const exact = @import("exact.zig");
const tri_index = @import("tri_index.zig");
const exl3_linear = @import("exl3_linear.zig");
const mx = @import("mx.zig");
const prompt2d = @import("prompt2d.zig");
const rdma = @import("rdma.zig");
const round = @import("round.zig");

/// A profile's (--profile 2) chunk timer (model.zig sets it while it times a chunk): the attention's and the MoE's
/// steps marked inside their phases (a_in, a_q, a_idx, a_core, a_woa; m_rt, m_gu, m_x), the rest of each in its own.
pub var ptimer: ?*round.PhaseTimer = null;

/// The monotonic clock (ns), for the engramWorker thread and TF_DS_LOG_TIMING's diagnostic prints (WP0 4.5): no
/// std.Io handle reaches that thread, so this calls libc directly rather than going through Model.now's std.Io path.
fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * 1_000_000_000 + ts.nsec;
}

fn pmark(e: *const Engine, p: round.PhaseTimer.Phase) void {
    if (ptimer) |t| t.mark(e.s, p) catch {};
}

/// The indexer's [rows, n_comp] keys a row block holds at most (model.py: rb * n_comp <= 2^26 / 4 past 16 rows).
const key_elems: usize = 1 << 24;
/// A row block's rows at most where the layers before the candidate source prune (prunes: n_comp > 32,768, so rb =
/// 2^26 / (4 n_comp) < 512).
const pruned_rows: usize = 512;

/// model.py's RING_EXTRA: window ring slots beyond the window (a verify window never clobbers a key it reads).
pub const ring_extra = 16;
/// model.py's RAW: per-position compressor inputs kept (a ratio-2 group's earlier row for the next chunk).
pub const raw_rows = 64;

/// Device memory carved in 256-byte aligned pieces from blocks allocated as it grows, all freed together: the first
/// block init's size, then blocks of `block` bytes; a piece of `own` bytes or more takes a block of its own (the
/// pool's caches), so a pool of any size costs what its pieces take plus at most one block. `used`: the pieces'
/// bytes; `reserved`: the blocks'.
pub const Arena = struct {
    d: *const cuda.Driver,
    blocks: [max_blocks]cuda.DeviceBuffer = undefined,
    n: usize = 0,
    cur: ?usize = null, // the block small pieces come from
    at: usize = 0, // its next free byte
    used: usize = 0,
    reserved: usize = 0,

    const max_blocks = 1024;
    const block: usize = 256 << 20;
    const own: usize = 64 << 20;

    pub fn init(d: *const cuda.Driver, bytes: usize) !Arena {
        var a: Arena = .{ .d = d };
        if (bytes > 0) {
            a.cur = try a.grow(bytes);
            a.at = 0;
        }
        return a;
    }

    fn grow(a: *Arena, bytes: usize) !usize {
        if (a.n == max_blocks) return error.ArenaFull;
        a.blocks[a.n] = try cuda.DeviceBuffer.alloc(a.d, bytes);
        a.reserved += bytes;
        a.n += 1;
        return a.n - 1;
    }

    pub fn deinit(a: *Arena) void {
        for (a.blocks[0..a.n]) |*b| b.free();
        a.n = 0;
    }

    pub fn take(a: *Arena, bytes: usize) !u64 {
        if (bytes >= own) {
            const i = try a.grow(std.mem.alignForward(usize, bytes, 256));
            a.used += bytes;
            return a.blocks[i].ptr;
        }
        var at = std.mem.alignForward(usize, a.at, 256);
        if (a.cur == null or at + bytes > a.blocks[a.cur.?].len) {
            a.cur = try a.grow(block);
            at = 0;
        }
        a.at = at + bytes;
        a.used += bytes;
        return a.blocks[a.cur.?].ptr + at;
    }
};

/// TF_DS_GRAPH_FAIL_TEST=RANK:N (round graphs) or RANK:N:pass (the drafter's pass graphs): a test of G0.3, never in
/// service: rank RANK's Nth capture of that kind reports a failed instantiate (its graph dropped). Unset: never.
pub fn injectedCaptureFailure(kind: enum { round, pass }, rank: u32, nth: u64) bool {
    const v = std.mem.span(std.c.getenv("TF_DS_GRAPH_FAIL_TEST") orelse return false);
    var it = std.mem.splitScalar(u8, v, ':');
    const r = std.fmt.parseInt(u32, it.next() orelse return false, 10) catch return false;
    const n = std.fmt.parseInt(u64, it.next() orelse return false, 10) catch return false;
    const want: @TypeOf(kind) = if (it.next()) |x| (if (std.mem.eql(u8, x, "pass")) .pass else return false) else .round;
    return r == rank and n == nth and want == kind;
}

/// A RoPE kind's tables on the device: cos, sin fp32 [rows, rope_dim / 2] (ops.RopeTables' rows).
pub const Rope = struct { cos: u64, sin: u64 };

/// What a chunk's steps run on: one rank's weights, kernels, stream and gather.
pub const Engine = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    t: tri.Tri,
    blas: *const cublas.Blas,
    comm: *const comm.Comm,
    pf: *const exl3_prefill.Kernels,
    lin: *const exl3_linear.Kernels,
    ex: *const exl3_experts.Kernels,
    ops: *const ops.Ops,
    exact: *const exact.Exact,
    c: *const Config,
    w: *const weights.Weights,
    world: usize,
    plain: Rope,
    compressed: Rope,
    round_rows: usize = 16, // a concurrent round's rows at most (TF_DS_ROUND_ROWS; round.max_rows at most)
    two: ?*const prompt2d.Two = null, // a node of the four-node 2D split (prompt2d.zig); `world` stays TP2's
    ring: ?*rdma.Ring = null, // TP2: the decode-size all-gathers over the RDMA ring (the served lane's), else NCCL
    par: ?*const exl3_experts2d.ParKernels = null, // the parity split's gate / up in one launch (null: the served two)
    mx: ?*const mx.Kernels = null, // Format.native: DeepSeek's own FP8 / MXFP4 checkpoint's kernels (mx.zig)
    agree_flag: u64 = 0, // a device byte for agreeAll (Model.init, layer_main); 0 at world > 1: agreeAll refuses

    /// G0.3 (docs/UNIFIED-DECODE-IMPROVEMENTS.md): whether `ok` holds on every rank, after a graph capture and
    /// instantiate and before any launch of it: one byte a rank set on the stream, the ranks' largest by NCCL (1: a
    /// failure), read after a synchronize. Called in the same order on every rank (a new graph key comes up at the same
    /// round and stretch everywhere). World 1: this rank's own. A failure in the middle of recording on one rank only
    /// is not covered: the others' recorded collectives stay unmatched (no out-of-band abort exists).
    pub fn agreeAll(e: *const Engine, ok: bool) !bool {
        if (e.comm.world <= 1) return ok;
        if (e.agree_flag == 0) return error.NoAgreeFlag;
        try e.d.check(e.d.api.cuMemsetD8Async(e.agree_flag, @intFromBool(!ok), 1, e.s.handle), "cuMemsetD8Async");
        try e.comm.maxBytes(e.agree_flag, 1, e.s);
        try e.s.synchronize();
        var v: [1]u8 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&v, e.agree_flag, 1), "cuMemcpyDtoH");
        return v[0] == 0;
    }

    /// A decode-size all-gather of `count` fp32 a rank into `dst`, rank after rank: the RDMA ring when the engine has
    /// one and the slice fits a slot (whole float4s), else NCCL. The same bytes either way.
    pub fn gatherF32(e: *const Engine, src: u64, dst: u64, count: usize) !void {
        if (e.ring) |r| if (count % 4 == 0 and count * 4 <= r.s.max_bytes) return r.gather(e.s, src, @intCast(count / 4), dst);
        return e.comm.allGather(src, dst, count, .f32, e.s);
    }

    pub fn heads(e: *const Engine) usize {
        if (e.two) |t| return t.heads;
        return e.c.heads / e.world;
    }

    pub fn slots(e: *const Engine) usize {
        return e.c.top_k + 1;
    }

    /// A sequence's window ring for one layer: bf16 [window + RING_EXTRA, head_dim].
    pub fn ringBytes(e: *const Engine) usize {
        return (e.c.window + ring_extra) * e.c.head_dim * 2;
    }
};

/// A prompt chunk's buffers for up to `cap` rows (model.py _forward_k's and its steps' torch.empty's).
/// An image span of the prompt: positions [at, at + len), its rows from row `row` of the image rows buffer.
pub const Span = struct { at: usize, len: usize, row: usize };

/// The rows [r0, r1) of the chunk (relative to its start) a span covers, if any.
fn overlap(sp: Span, start: usize, n: usize) ?[2]usize {
    const lo = @max(sp.at, start);
    const hi = @min(sp.at + sp.len, start + n);
    return if (lo < hi) .{ lo - start, hi - start } else null;
}

pub const Chunk = struct {
    cap: usize,
    spans: []const Span = &.{}, // the prompt's image spans (absolute positions)
    img_rows: u64 = 0, // bf16 [span tokens, D]: their rows
    pick_vl: u64 = 0, // int32 [cap, slots]: the routing of span rows with the image bias, before it joins pick
    wts_vl: u64 = 0, // fp32 [cap, slots]
    n: usize = 0,
    start: usize = 0,
    ids: u64, // int64 [cap]
    pos: u64, // int64 [n]: start .. start + n (after a replay cut a view into pos0, pos[first:])
    pos0: u64, // int64 [cap]: the chunk's positions as begin() wrote them
    pos1: u64 = 0, // int64 [cap]: a replay cut's positions moved to a 16-byte boundary (replayCut)
    h: u64, // bf16 [cap, hc, D]: the streams
    h_alt: u64, // their second buffer (hc_pre_pf writes the posted streams there)
    x: u64, // bf16 [cap, D]: a block's input rows
    part: u64, // fp32 [cap * HC_BLOCKS * 32]
    pre: u64, // fp32 [cap, hc]: the layer's input pre (the previous layer's FFN pre_out)
    pre_a: u64, // fp32 [cap, hc]
    pre_f: u64, // fp32 [cap, hc]
    post: u64, // fp32 [cap, hc]
    comb: u64, // fp32 [cap, hc, hc]
    // attention
    qa: u64, // bf16 [cap, q_lora]
    y: u64, // bf16 [cap, head_dim]
    qr: u64, // bf16 [cap, q_lora]
    q: u64, // bf16 [cap, heads, head_dim]
    wsrc: u64, // bf16 [window - 1 + cap, head_dim]
    wlo: u64, // int64 [1]
    o: u64, // bf16 [cap, heads, head_dim]
    u: u64, // bf16 [cap, groups * o_lora]
    pa: u64, // fp32 [cap, D]: the attention partial
    ga: u64, // fp32 [world, cap, D]: its gather
    neg: u64, // int64 [cap], -1: no ring slot (model.py _neg)
    ring_idx: u64, // int64 [ring]
    rslot: u64 = 0, // int64 [RING_EXTRA]: ring mode's slots, pos % R
    pm_: u64 = 0, // fp32 [16, heads, 8]: the split attention's partials (ring mode's rows)
    pl_: u64 = 0,
    po_: u64 = 0, // fp32 [16, heads, 8, head_dim]
    invalid: u64, // uint32 [1]
    // compressed layers
    kvc: u64 = 0, // fp32 [cap, head_dim]: the compressor's kv projection
    scc: u64 = 0, // fp32 [cap, head_dim]: its gate scores
    lat: u64 = 0, // bf16 [cap, head_dim]: the compressed latents
    lat2: u64 = 0, // bf16 [cap, head_dim]: their RoPE'd copy (the cache's)
    ik: u64 = 0, // bf16 [cap, index_head_dim]: the index keys
    groups: u64 = 0, // int64 [cap]: the new latents' groups
    gpos: u64 = 0, // int64 [cap]: their positions (group * ratio)
    raw_idx: u64 = 0, // int64 [RAW]
    iq: u64 = 0, // bf16 [cap, index_heads * index_head_dim]
    iq4: u64 = 0, // bf16 [cap, index_heads * index_head_dim]: fp4_qd's bytes
    wl: u64 = 0, // fp32 [cap, index_heads]
    iw: u64 = 0, // bf16 [cap, index_heads]: the heads' weights
    vis: u64 = 0, // int64 [cap]: compressed entries a row sees
    keys: u64 = 0, // int64 [rows of a block, n_comp]: at most key_elems
    score: u64 = 0, // fp32 [rows of a block, n_comp]: the candidate pool's layers' index scores
    tmax: u64 = 0, // int64 [rows of a block, cdiv(n_comp, 64)]
    cidx: u64 = 0, // int64 [cap, index_topk]: the selected latents
    max_comp: usize = 0,
    // the candidate pool past candidate_blocks blocks (model.py _candidates, apply_candidates): the chunk's block mask
    // (u8 [cap, pool_ms]), a row block's block maxima (fp32) and their top blocks (int64 [rows, candidate_blocks]), a
    // row's every_vis (no mask); the layers before the candidate source's pruned selection (tpos, cand: up to
    // pruned_rows rows)
    pool: u64 = 0,
    pool_ms: usize = 0,
    bmax: u64 = 0,
    pidx: u64 = 0,
    every: u64 = 0,
    pbufs: tri_index.PrunedBufs = .{ .tpos = 0, .cand = 0, .every = 0 },
    // MoE
    xf: u64, // fp32 [cap, D]
    gate_f: u64, // fp32 [experts, D]
    logits: u64, // fp32 [cap, experts]
    pick: u64, // int32 [cap, slots]
    wts: u64, // fp32 [cap, slots]
    pm: u64, // fp32 [cap, D]: the MoE partial
    gm: u64, // fp32 [world, cap, D]: its gather (the pending post)
    // WP4 (docs/prefill-plan.md, quarter-HC): a repack of ga's or gm's owned-row slice, tightly strided [world, q, D]
    // (q = cap / world) -- _hc_mix_pf / _hc_post derive the gathered buffer's per-plane stride from the launch's own
    // `rows` (RS = rows * D), not a separate cap constant, so a pointer offset into ga/gm's cap-strided planes would
    // read the wrong bytes once `rows` is q instead of cap. Zero size (and never touched) when there is no Two.
    qhc_g: u64 = 0,
    // WP4: `part` is pure intermediate scratch for the Sinkhorn/mix iteration (written by _hc_mix_pf, read back by
    // _hc_finish within the SAME call, never carrying state between calls) whose layout is documented only as a
    // flat element count, not a dimension order (unlike h/pre/post/comb, which ARE plainly per-row and safe to
    // pointer-offset -- tf-dsv41-hc-replay's own working test proves it for exactly those). A pointer offset into
    // the full-cap ch.part assumes a row-major layout that was never established; give it its own buffer, sized
    // tight for q = cap / world rows, exactly as the real engine would size ch.part for a q-row chunk.
    qhc_part: u64 = 0,
    xs: exl3_experts.Scratch,
    // Model._moe_scratch's decode scratch: windows under 64 rows (a prompt's small chunk, every decode round), one for
    // every layer of the same slots and experts, its epoch carried from call to call
    xsd: exl3_experts.DecodeScratch,
    gl: u64, // fp32 [16, D / 256, experts]: rowmm_gate's chunk sums (chunks of up to 16 rows)
    cidxp: u64 = 0, // int64 [cap, index_topk + 16]: the selection padded with -1 to whole 16-column tiles
    // Engram (zero when no layer has it)
    eb: u64 = 0, // bf16 [cap, cols * engram_head_dim]: the rows read
    ek: u64 = 0, // fp32 [cap, hc * D + D]: their projection, this rank's columns
    ekg: u64 = 0, // fp32 [world, cap, hc * D + D]: its gather
    kv: u64 = 0, // bf16 [cap, hc * D + D]: the ranks' sum
    ws: exl3_prefill.Workspace,
    blas_ws: u64, // cuBLAS workspace
    // the grouped EXL3 linears of decode-sized rows (1..128): rotated rows, split-K partials, each layer's counters
    gxh: u64 = 0, // fp16 [128, the largest group's sum of K]
    gz: u64 = 0, // fp32 [the largest group's split-K partials at 128 rows]
    counters: [640]Counter = undefined,
    n_counters: usize = 0,
    ktop: u64 = 0, // int64 [cap, index_topk]: torch's top-k values for _topk_finish
    // the DSpark taps and the head
    taps: u64 = 0, // bf16 [taps, cap, D]: each tap layer's mean of the streams (Model.forward's taps)
    head_x: u64 = 0, // bf16 [1, D]: collapse_norm of the last row
    head_l: u64 = 0, // fp32 [1, this rank's vocabulary columns]: its logits
    head_g: u64 = 0, // fp32 [world, 1, columns]: their gather (the prompt's logits, rank after rank)
    pending: bool = false, // gm holds a MoE gather whose post is not in h yet
    // Format.native (mx.zig): the linears' quantized rows and split partials, and the experts' scratch (the
    // intermediate's rows: xs.xd for a prompt chunk's, xsd.xd for a decode window's)
    mxs: mx.Scratch = undefined,
    mxe: mx.ExpertScratch = undefined,

    /// Every buffer for `cap` rows from `a`, the compressed layers' for chunks ending by position `tokens`; the scratch
    /// the served build zeros (or fills with -1) is set the same.
    pub fn init(e: *const Engine, a: *Arena, cap: usize, tokens: usize) !Chunk {
        const c = e.c;
        const d = c.hidden;
        const hc = c.hc;
        const l0 = e.w.layers[0];
        const hl = e.heads();
        const hd = c.head_dim;
        const groups = l0.groups;
        const o_lora = l0.wo_a[0].n;
        const ex = l0.experts;
        const sl = e.slots();
        var ch: Chunk = undefined;
        ch.spans = &.{};
        ch.img_rows = 0;
        ch.cap = cap;
        ch.n = 0;
        ch.start = 0;
        ch.pending = false;
        ch.ids = try a.take(cap * 8);
        ch.pos0 = try a.take(cap * 8);
        ch.pos1 = try a.take(cap * 8);
        ch.pos = ch.pos0;
        ch.h = try a.take(cap * hc * d * 2);
        ch.h_alt = try a.take(cap * hc * d * 2);
        ch.x = try a.take(cap * d * 2);
        ch.part = try a.take(cap * tri_basic.hc_blocks * 32 * 4);
        ch.pre = try a.take(cap * hc * 4);
        ch.pre_a = try a.take(cap * hc * 4);
        ch.pre_f = try a.take(cap * hc * 4);
        ch.post = try a.take(cap * hc * 4);
        ch.comb = try a.take(cap * hc * hc * 4);
        ch.qa = try a.take(cap * l0.wq_a.n * 2);
        ch.y = try a.take(cap * hd * 2);
        ch.qr = try a.take(cap * l0.wq_a.n * 2);
        ch.q = try a.take(cap * hl * hd * 2);
        ch.wsrc = try a.take((c.window - 1 + cap) * hd * 2);
        ch.wlo = try a.take(8);
        ch.o = try a.take(cap * hl * hd * 2);
        ch.u = try a.take(cap * groups * o_lora * 2);
        ch.pa = try a.take(cap * d * 4);
        ch.ga = try a.take(e.world * cap * d * 4);
        ch.neg = try a.take(cap * 8);
        ch.ring_idx = try a.take((c.window + ring_extra) * 8);
        ch.rslot = try a.take(ring_extra * 8);
        ch.pm_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.pl_ = try a.take(ring_extra * hl * tri_attn.attn_splits * 4);
        ch.po_ = try a.take(ring_extra * hl * tri_attn.attn_splits * hd * 4);
        ch.invalid = try a.take(4);
        ch.xf = try a.take(cap * d * 4);
        ch.gate_f = try a.take(c.experts * d * 4);
        ch.logits = try a.take(cap * c.experts * 4);
        ch.pick = try a.take(cap * sl * 4);
        ch.wts = try a.take(cap * sl * 4);
        ch.pick_vl = try a.take(cap * sl * 4);
        ch.wts_vl = try a.take(cap * sl * 4);
        ch.pm = try a.take(cap * d * 4);
        ch.gm = try a.take(e.world * cap * d * 4);
        // WP4 quarter-HC scratch: only when a Two exists and cap divides evenly by the physical world (today always
        // 4): sized for the owned-row-slice repack, world * q * d, q = cap / world.
        if (e.two != null and e.comm.world != 0 and cap % e.comm.world == 0) {
            ch.qhc_g = try a.take(e.world * (cap / e.comm.world) * d * 4);
            ch.qhc_part = try a.take((cap / e.comm.world) * tri_basic.hc_blocks * 32 * 4);
        }
        // compressed layers: the smallest ratio sets the most compressed entries a row can see
        var min_ratio: usize = 0;
        for (e.w.layers) |lay| {
            if (lay.ratio != 0 and (min_ratio == 0 or lay.ratio < min_ratio)) min_ratio = lay.ratio;
        }
        if (min_ratio != 0) {
            const ih = c.index_heads * c.index_head_dim;
            ch.max_comp = tokens / min_ratio;
            ch.kvc = try a.take(cap * c.head_dim * 4);
            ch.scc = try a.take(cap * c.head_dim * 4);
            ch.lat = try a.take(cap * c.head_dim * 2);
            ch.lat2 = try a.take(cap * c.head_dim * 2);
            ch.ik = try a.take(cap * c.index_head_dim * 2);
            ch.groups = try a.take(cap * 8);
            ch.gpos = try a.take(cap * 8);
            ch.raw_idx = try a.take(raw_rows * 8);
            ch.iq = try a.take(cap * ih * 2);
            ch.iq4 = try a.take(cap * ih * 2);
            ch.wl = try a.take(cap * c.index_heads * 4);
            ch.iw = try a.take(cap * c.index_heads * 2);
            ch.vis = try a.take(cap * 8);
            // the indexer's row blocks keep rows * n_comp under 2^24 (model.py rb), so their keys under key_elems
            const ke: usize = @min(cap * ch.max_comp, key_elems); // (typed: @min with a comptime bound narrows)
            ch.keys = try a.take(ke * 8);
            ch.score = try a.take(ke * 4);
            ch.tmax = try a.take((ke / tri_index.tile + cap) * 8);
            ch.cidx = try a.take(cap * c.index_topk * 8);
            ch.ktop = try a.take(cap * c.index_topk * 8);
            ch.cidxp = try a.take(cap * (c.index_topk + 16) * 8);
            // the pool: only past candidate_blocks blocks, so a row block holds at most key_elems / 2048 rows there
            const cb = c.candidate_block;
            ch.pool_ms = (ch.max_comp + cb - 1) / cb;
            if (ch.pool_ms > c.candidate_blocks) {
                ch.pool = try a.take(cap * ch.pool_ms);
                ch.bmax = try a.take(@max(16 * ch.pool_ms, key_elems / cb + cap) * 4);
                ch.pidx = try a.take(cap * c.candidate_blocks * 8);
            }
            ch.every = try a.take(cap * 8);
            {
                const ev = try std.heap.page_allocator.alloc(i64, cap);
                defer std.heap.page_allocator.free(ev);
                @memset(ev, tri_index.every_vis);
                try upload(e, ch.every, ev.ptr, cap * 8);
                try e.s.synchronize(); // (the host copy is freed next)
            }
            if (tri_index.prunes(ch.max_comp, c.index_topk)) {
                ch.pbufs = .{ .tpos = try a.take(pruned_rows * c.index_topk * 8), .cand = try a.take(pruned_rows * c.index_topk * tri_index.tile * 8), .every = ch.every };
            }
        }
        for (e.w.layers) |lay| {
            const ew = lay.engram_wkv orelse continue;
            const en = if (e.two) |t| t.engramWidth() else ew.n; // the summed projection's width (2D: both pairs')
            ch.eb = try a.take(cap * ew.k * 2);
            ch.ek = try a.take(cap * ew.n * 4);
            ch.ekg = try a.take(e.world * cap * en * 4);
            ch.kv = try a.take(cap * en * 2);
            break;
        }
        const sz = exl3_experts.Scratch.sizes(cap, sl, ex.dims, ex.width + if (ex.rest) |r| r.width else 0, ex.count); // parity: the pack
        var xs: exl3_experts.Scratch = undefined;
        xs.rows = cap;
        xs.slots = sl;
        if (e.mx != null) {
            // Format.native: only the intermediate's rows (bf16 [cap * slots, pack]), xs.xd for chunks and xsd.xd for
            // decode windows; the grouping and the quantized rows live in ch.mxe
            const pack = ex.width + if (ex.rest) |r| r.width else 0;
            xs = std.mem.zeroes(exl3_experts.Scratch);
            xs.rows = cap;
            xs.slots = sl;
            xs.xd = try a.take(cap * sl * pack * 2);
            ch.xs = xs;
            ch.xsd = std.mem.zeroes(exl3_experts.DecodeScratch);
            ch.xsd.rows = exl3_experts.exact_rows;
            ch.xsd.slots = sl;
            ch.xsd.xd = xs.xd;
            try mxScratch(e, &ch, a, cap);
        } else {
            inline for (.{ "xg", "xu", "xd", "z", "no_y", "ids", "count", "counts", "members", "work_gu", "work_d" }, 0..) |f, j| {
                @field(xs, f) = try a.take(sz[j]);
            }
            ch.xs = xs;
            ch.xsd = try decodeScratch(e, a, ex, sl);
            if (e.two) |t| try t.clearRest(e); // parity: the rest launches' grouping starts empty
        }
        ch.gl = try a.take(16 * (d / 256) * c.experts * 4);
        // prefill.Workspace: the largest prompt GEMM's rotated input and W_q (fp16) over every layer, and the Hadamard
        var max_xk: usize = 0;
        var max_kn: usize = 0;
        for (e.w.layers) |lay| {
            var ls: [16]?weights.Linear = @splat(null);
            ls[0] = lay.wq_a;
            ls[1] = lay.wkv;
            ls[2] = lay.wq_b;
            ls[3] = lay.wo_b;
            for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
            ls[8] = lay.comp_wkv;
            ls[9] = lay.comp_wgate;
            ls[10] = lay.idx_wq_b;
            ls[11] = lay.idx_wk;
            ls[12] = lay.engram_wkv;
            for (ls) |ol| {
                const l = ol orelse continue;
                max_xk = @max(max_xk, l.k);
                max_kn = @max(max_kn, @as(usize, l.k) * l.n);
            }
        }
        ch.ws = .{ .xh = try a.take(cap * max_xk * 2), .w = try a.take(max_kn * 2), .h = try a.take(128 * 128 * 2) };
        ch.blas_ws = try a.take(cublas.Blas.workspace_bytes);
        // grouped linears: every layer's counters (int32 [8 N / 128], zeros, left zero by each launch), the largest
        // group's rotated rows and partials at 128 rows
        var max_gk: usize = 0;
        var max_gz: usize = 0;
        ch.n_counters = 0;
        for (e.w.layers) |lay| try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
        {
            // the head: one row through its own group
            const l = e.w.head;
            try addLinear(e, &ch, a, l, &max_gk, &max_gz);
            ch.head_x = try a.take(d * 2);
            ch.head_l = try a.take(l.n * 4);
            const hn = if (e.two) |t| t.hw[0] + t.hw[1] else l.n; // a rank's vocabulary half (2D: both pairs' parts)
            ch.head_g = try a.take(e.world * hn * 4);
        }
        if (e.w.dspark) |ds| {
            // the drafter (draft.zig): its stages' linears as a layer's, main_proj, and the stages' wkv as one group
            // (Drafter.stage_kv)
            var kv_k: usize = 0;
            var kv: [8]weights.Linear = undefined;
            if (ds.blocks.len > kv.len) return error.TooManyStages;
            for (ds.blocks, 0..) |lay, j| {
                try layerLinears(e, &ch, a, lay, &max_gk, &max_gz);
                kv[j] = lay.wkv;
                kv_k += lay.wkv.k;
            }
            max_gk = @max(max_gk, kv_k);
            max_gz = @max(max_gz, exl3_linear.zFloats(kv[0..ds.blocks.len], 128));
            try addLinear(e, &ch, a, ds.main_proj, &max_gk, &max_gz);
        }
        ch.taps = try a.take(@max(c.dspark_taps.slice().len, 1) * cap * d * 2);
        ch.gxh = try a.take(128 * max_gk * 2);
        ch.gz = try a.take(@max(max_gz, 1) * 4);
        // the scratch as experts.py makes it: zeros, the member lists -1
        if (e.mx == null) {
            for ([_]u64{ xs.xg, xs.xu, xs.xd, xs.z, xs.no_y, xs.ids, xs.count, xs.counts }, [_]usize{ sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sz[6], sz[7] }) |p, n| try fill(e, p, 0, n);
            try fill32(e, xs.members, 0xffffffff, sz[8] / 4);
        }
        try fill(e, ch.neg, 0xff, cap * 8);
        var had: [128 * 128]u16 = undefined;
        exl3_prefill.hadamard(&had);
        try upload(e, ch.ws.h, &had, had.len * 2);
        return ch;
    }
};

/// Format.native's scratch for chunks of `cap` rows (and rounds and drafter passes, fewer): the widest K any native
/// linear reads (wo_a's groups together), the largest split partials, and the experts' grouping for the routed rows.
fn mxScratch(e: *const Engine, ch: *Chunk, a: *Arena, cap: usize) !void {
    var max_k: usize = 0;
    var zf: usize = 0;
    var e_max: usize = 0;
    var dn_k: usize = 0;
    var dn_n: usize = 0;
    const blocks: []const weights.Layer = if (e.w.dspark) |ds| ds.blocks else &.{};
    for ([_][]const weights.Layer{ e.w.layers, blocks }) |ls| for (ls) |lay| {
        var lin: [16]?weights.Linear = @splat(null);
        lin[0] = lay.wq_a;
        lin[1] = lay.wkv;
        lin[2] = lay.wq_b;
        lin[3] = lay.wo_b;
        lin[4] = lay.comp_wkv;
        lin[5] = lay.comp_wgate;
        lin[6] = lay.idx_wq_b;
        lin[7] = lay.idx_wk;
        lin[8] = lay.engram_wkv;
        for (lin) |ol| {
            const l = ol orelse continue;
            max_k = @max(max_k, l.k);
            zf = @max(zf, mx.zFloats(l, cap));
        }
        var wk: usize = 0;
        for (lay.wo_a[0..lay.groups]) |wo| wk += wo.k;
        max_k = @max(max_k, wk);
        // the 2D split's wo_b reads the rank's whole wo_a output
        if (e.two) |t| max_k = @max(max_k, t.uw[0] + t.uw[1]);
        e_max = @max(e_max, lay.experts.count);
        dn_k = @max(dn_k, lay.experts.down_k);
        dn_n = @max(dn_n, lay.experts.down_n);
        max_k = @max(max_k, lay.experts.down_k);
    };
    if (e.w.dspark) |ds| {
        max_k = @max(max_k, ds.main_proj.k);
        zf = @max(zf, mx.zFloats(ds.main_proj, cap));
    }
    const ls = mx.Scratch.sizes(cap, max_k, zf);
    ch.mxs = .{ .q = try a.take(ls[0]), .s = try a.take(ls[1]), .z = try a.take(ls[2]), .fake = try a.take(ls[3]), .rows = cap, .max_k = max_k };
    const sl = e.slots();
    const es = mx.ExpertScratch.sizes(cap, sl, e_max, e.c.hidden, dn_k, dn_n);
    var x: mx.ExpertScratch = undefined;
    x.rows = cap;
    x.slots = sl;
    inline for (.{ "q", "s", "cnt", "off", "fill", "act", "nact", "ent", "erow", "hq", "hs", "out" }, 0..) |f, j| {
        @field(x, f) = try a.take(es[j]);
    }
    // the prefill tiles' list (mx.gateUp / down of a prompt chunk)
    x.tiles = try a.take(mx.ExpertScratch.tileBytes(cap, sl, e_max));
    x.ntiles = try a.take(4);
    x.nq = 0;
    x.ns = 0;
    x.hnq = 0;
    x.hns = 0;
    if (e.c.nvfp4) {
        const ns = mx.ExpertScratch.nvSizes(cap, sl, e.c.hidden, dn_k);
        x.nq = try a.take(ns[0]);
        x.ns = try a.take(ns[1]);
        x.hnq = try a.take(ns[2]);
        x.hns = try a.take(ns[3]);
    }
    x.z = .{ .ptr = 0, .floats = 0 };
    // decode launches' split partials (round.zig and the drafter's own: mx.decode_rows rows at most)
    var zf_dec: usize = 0;
    for ([_][]const weights.Layer{ e.w.layers, blocks }) |dls| for (dls) |lay| {
        zf_dec = @max(zf_dec, mx.decodeZFloats(lay, e.c.hidden, e.c.experts, e.c.index_heads, mx.decode_rows, sl));
    };
    if (e.w.head.fmt == .bf16) zf_dec = @max(zf_dec, mx.bfSplits(e.w.head.k, e.w.head.n) * mx.decode_rows * e.w.head.n);
    if (zf_dec > 0) x.z = .{ .ptr = try a.take(zf_dec * 4), .floats = zf_dec };
    ch.mxe = x;
}

/// A layer's linears in the chunk's grouped-linear sizes (its attention input group, its wo_a group, each alone at 128
/// rows) and each one's split-K counters (Chunk.init).
fn layerLinears(e: *const Engine, ch: *Chunk, a: *Arena, lay: weights.Layer, max_gk: *usize, max_gz: *usize) !void {
    const in_group = [_]weights.Linear{ lay.wq_a, lay.wkv };
    max_gk.* = @max(max_gk.*, lay.wq_a.k + lay.wkv.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&in_group, 128));
    var wo_k: usize = 0;
    for (lay.wo_a[0..lay.groups]) |wo| wo_k += wo.k;
    max_gk.* = @max(max_gk.*, wo_k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(lay.wo_a[0..lay.groups], 128));
    var ls: [16]?weights.Linear = @splat(null);
    ls[0] = lay.wq_a;
    ls[1] = lay.wkv;
    ls[2] = lay.wq_b;
    ls[3] = lay.wo_b;
    for (lay.wo_a[0..lay.groups], 0..) |wo, g| ls[4 + g] = wo;
    ls[8] = lay.comp_wkv;
    ls[9] = lay.comp_wgate;
    ls[10] = lay.idx_wq_b;
    ls[11] = lay.idx_wk;
    ls[12] = lay.engram_wkv;
    for (ls) |ol| {
        const l = ol orelse continue;
        try addLinear(e, ch, a, l, max_gk, max_gz);
    }
}

/// One linear alone at 128 rows in the chunk's grouped-linear sizes, and its split-K counters (int32 [8 N / 128], zeros,
/// left zero by each launch).
fn addLinear(e: *const Engine, ch: *Chunk, a: *Arena, l: weights.Linear, max_gk: *usize, max_gz: *usize) !void {
    max_gk.* = @max(max_gk.*, l.k);
    max_gz.* = @max(max_gz.*, exl3_linear.zFloats(&[_]weights.Linear{l}, 128));
    if (ch.n_counters == ch.counters.len) return error.TooManyLinears;
    const bytes = 8 * (l.n / 128) * 4;
    const ptr = try a.take(bytes);
    try fill(e, ptr, 0, bytes);
    ch.counters[ch.n_counters] = .{ .words = l.words, .ptr = ptr };
    ch.n_counters += 1;
}

/// Model._moe_scratch's decode scratch: experts.py Scratch(rows=max(n, 64), prompt=False), zeros but its member lists'
/// -1 (made at the first window under 64 rows and never replaced).
pub fn decodeScratch(e: *const Engine, a: *Arena, ex: weights.Experts, slots: usize) !exl3_experts.DecodeScratch {
    const rows = exl3_experts.exact_rows;
    const sz = if (e.two != null) try exl3_experts2d.decodeSizes(rows, slots, ex) else try exl3_experts.DecodeScratch.sizes(rows, slots, ex.dims, ex.width, ex.count); // 2D: this node's widths
    var sc: exl3_experts.DecodeScratch = undefined;
    sc.rows = rows;
    sc.slots = slots;
    inline for (.{ "xg", "xu", "xd", "z", "y", "cnt_gu", "cnt_d", "epoch", "ready", "ready_cnt", "ids", "count", "members" }, 0..) |f, j| {
        const ptr = try a.take(sz[j]);
        @field(sc, f) = ptr;
        if (j == 12) try fill32(e, ptr, 0xffffffff, sz[j] / 4) else try fill(e, ptr, 0, sz[j]);
    }
    return sc;
}

/// Host bytes to the device in the stream's order (the stream is non-blocking: the legacy stream's synchronous copies
/// would not wait for its kernels). Pageable host memory is staged before the call returns.
pub fn upload(e: *const Engine, dst: u64, src: *const anyopaque, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, src, bytes, e.s.handle), "cuMemcpyHtoDAsync");
}

/// cuMemsetD8 in the stream's order.
pub fn fill(e: *const Engine, dst: u64, value: u8, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemsetD8Async(dst, value, bytes, e.s.handle), "cuMemsetD8Async");
}

/// cuMemsetD32 in the stream's order.
pub fn fill32(e: *const Engine, dst: u64, value: u32, words: usize) !void {
    try e.d.check(e.d.api.cuMemsetD32Async(dst, value, words, e.s.handle), "cuMemsetD32Async");
}

/// A layer's split-K counters (Exl3Linear.counters) by its trellis words.
pub const Counter = struct { words: u64, ptr: u64 };

pub fn counterOf(ch: *const Chunk, l: weights.Linear) !u64 {
    for (ch.counters[0..ch.n_counters]) |c| if (c.words == l.words) return c.ptr;
    return error.NoCounters;
}

/// Exl3Group(ls) on m (1..128) rows (linear.py): one rot_many launch rotating every layer's input rows xs[i] (bf16,
/// row stride ldxs[i]) into the group's buffer, then a glinear launch for each (bits, warps) set, outputs into outs[i]
/// (row stride ldos[i], dtype types[i]); programmatic dependent launches, split-K partials dropped from L2 (the served
/// TF_EXL3_PDL and TF_EXL3_L2_DISCARD).
pub fn grouped(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    if (m == 0 or m > 128 or ls.len > 8) return error.NotADecodeWindow;
    if (e.mx) |k| {
        // Format.native: each linear on its own (its rows quantized, the MX or bf16 GEMM): the bits of any row count
        for (ls, 0..) |l, i| try mx.linear(k, e.s, ch.mxs, l, xs[i], .bf16, ldxs[i], m, outs[i], ldos[i], types[i], false, null);
        return;
    }
    var calls: [8]exl3_linear.Call = undefined;
    var off: usize = 0;
    for (ls, 0..) |l, i| {
        calls[i] = .{ .layer = l, .x = xs[i], .ldx = @intCast(ldxs[i]), .x_dtype = .bf16, .xh = ch.gxh + off * 2, .y = outs[i], .ldy = @intCast(ldos[i]), .y_dtype = types[i], .counters = try counterOf(ch, l) };
        off += m * l.k;
    }
    try exl3_linear.rotMany(e.lin, e.s, calls[0..ls.len], m, true);
    try exl3_linear.glinear(e.lin, e.s, calls[0..ls.len], m, ch.gz, true, true);
}

/// Exl3Group.rotated on m rows already rotated into each call's xh (a kernel that folds the rotation in wrote them):
/// the glinear launches alone, each call's counters and folds as given.
/// grouped for a decode launch (a round's rows, the drafter's): Format.native's FP8 / FP4 linears through the GEMV
/// (mx.linear decode), each row its own bits at any row count; EXL3 as grouped.
pub fn groupedDecode(e: *const Engine, ch: *const Chunk, m: usize, ls: []const weights.Linear, xs: []const u64, ldxs: []const usize, outs: []const u64, ldos: []const usize, types: []const exl3_linear.DType) !void {
    const k = e.mx orelse return grouped(e, ch, m, ls, xs, ldxs, outs, ldos, types);
    // past mx.decode_rows rows (the drafter's pass of 13+ streams) linearsDecode runs slabs of the same kernels
    if (m == 0 or m > 128 or ls.len > 8) return error.NotADecodeWindow;
    try mx.linearsDecode(k, e.s, ch.mxs, ls, xs, ldxs, m, outs, ldos, types, ch.mxe.z);
}

pub fn groupedRotated(e: *const Engine, ch: *const Chunk, m: usize, calls: []exl3_linear.Call) !void {
    if (m == 0 or m > 128 or calls.len > exl3_linear.gmax) return error.NotADecodeWindow;
    if (e.mx != null) return error.RotatedRowsOnNative; // Format.native: the callers pass plain rows (grouped)
    for (calls) |*c| c.counters = try counterOf(ch, c.layer);
    try exl3_linear.glinear(e.lin, e.s, calls, m, ch.gz, true, true);
}

/// The pool's compressed-attention caches (model.py PoolCache: comp, index_k, comp_raw), by kv-source layer: the
/// compressed latents as packed FP4 (codes [rows, head_dim / 2], E8M0 scales [rows, head_dim / 16]), the index keys
/// (codes [rows, index_head_dim / 2], scales [rows, index_head_dim / 32], 127 when unwritten) over `cap` positions of
/// streams' extents (rows = cap / ratio + 2), and, at ratio 2, each slot's positional store of the compressor's raw kv
/// and scores (fp32 [RAW, head_dim] each, `slots` of them). One stream's caches (SeqCache) are a view.
pub const Caches = struct {
    comp_codes: [64]u64 = @splat(0),
    comp_scales: [64]u64 = @splat(0),
    idx_codes: [64]u64 = @splat(0),
    idx_scales: [64]u64 = @splat(0),
    raw_kv: [64]u64 = @splat(0),
    raw_score: [64]u64 = @splat(0),
    cap: usize = 0, // positions
    slots: usize = 1, // the positional stores' slots

    pub fn init(e: *const Engine, a: *Arena, cap: usize, slots: usize) !Caches {
        const c = e.c;
        var cs: Caches = .{ .cap = cap, .slots = slots };
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cap / r + 2;
            cs.comp_codes[i] = try a.take(rows * c.head_dim / 2);
            cs.comp_scales[i] = try a.take(rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                cs.idx_codes[i] = try a.take(rows * c.index_head_dim / 2);
                cs.idx_scales[i] = try a.take(rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                cs.raw_kv[i] = try a.take(slots * raw_rows * c.head_dim * 4);
                cs.raw_score[i] = try a.take(slots * raw_rows * c.head_dim * 4);
            }
        }
        try cs.clear(e);
        return cs;
    }

    /// A new pool's caches, as PoolCache makes them: zeros, the index keys' E8M0 scales 127 (1.0).
    pub fn clear(cs: *const Caches, e: *const Engine) !void {
        const c = e.c;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            const rows = cs.cap / r + 2;
            try fill(e, cs.comp_codes[i], 0, rows * c.head_dim / 2);
            try fill(e, cs.comp_scales[i], 0, rows * c.head_dim / 16);
            if (lay.idx_wk != null) {
                try fill(e, cs.idx_codes[i], 0, rows * c.index_head_dim / 2);
                try fill(e, cs.idx_scales[i], 127, rows * c.index_head_dim / 32);
            }
            if (r > 1) {
                try fill(e, cs.raw_kv[i], 0, cs.slots * raw_rows * c.head_dim * 4);
                try fill(e, cs.raw_score[i], 0, cs.slots * raw_rows * c.head_dim * 4);
            }
        }
    }

    /// The stream in pool slot `slot` with its extent from position `base` (MultiDecoder: pool_view): its compressed
    /// rows from base / ratio, its slot's positional stores (one slot's view: its prompt chunks read positions from 0).
    pub fn view(cs: *const Caches, e: *const Engine, slot: usize, base: usize) !Caches {
        const c = e.c;
        if (slot >= cs.slots or base >= cs.cap) return error.BadView;
        var v = cs.*;
        v.slots = 1;
        v.cap = cs.cap - base;
        for (e.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (base % r != 0) return error.BadView;
            const g = base / r;
            v.comp_codes[i] += g * c.head_dim / 2;
            v.comp_scales[i] += g * c.head_dim / 16;
            if (lay.idx_wk != null) {
                v.idx_codes[i] += g * c.index_head_dim / 2;
                v.idx_scales[i] += g * c.index_head_dim / 32;
            }
            if (r > 1) {
                v.raw_kv[i] += slot * raw_rows * c.head_dim * 4;
                v.raw_score[i] += slot * raw_rows * c.head_dim * 4;
            }
        }
        return v;
    }
};

/// What one forward carries from layer to layer (model.py _forward_k's `shared`): the latest kv-source layer, and the
/// latest indexer's selection (in the chunk's cidx, kk a row).
pub const Shared = struct { kv_layer: ?usize = null, kk: ?usize = null, pool: bool = false };

/// The chunk's rows: token ids at positions start .., their streams h [n, hc, D] (every stream the token's embedding
/// row) and pre [n, hc] = (1, 0, 0, 0): embed_init, the bytes of forward()'s embedding rows expanded over the streams.
pub fn begin(e: *const Engine, ch: *Chunk, ids: []const i64, start: usize, host_pos: []i64) !void {
    const n = ids.len;
    if (n > ch.cap or host_pos.len < n) return error.ChunkTooLong;
    ch.n = n;
    ch.start = start;
    ch.pending = false;
    for (host_pos[0..n], 0..) |*p, i| p.* = @intCast(start + i);
    try upload(e, ch.ids, ids.ptr, n * 8);
    ch.pos = ch.pos0;
    try upload(e, ch.pos, host_pos.ptr, n * 8);
    try tri_basic.embedInit(e.t, e.w.embed, ch.ids, ch.h, ch.pre, n, e.c.hidden, e.c.hc);
}

/// merge_image_embeddings on the chunk: each span row's embedding (in every one of the hc streams) replaced by its
/// image row (the delimiters' learned rows and the aligner's), before the first layer.
pub fn embedSpans(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const row = c.hidden * 2;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, ch.n) orelse continue;
        const src = ch.img_rows + (sp.row + (ch.start + o[0] - sp.at)) * row;
        for (0..c.hc) |s| try e.ops.copyRows(e.s, src, row, ch.h + (o[0] * c.hc + s) * row, c.hc * row, row, o[1] - o[0]);
    }
}

/// The streams' second buffer becomes the streams (hc_pre_pf wrote the posted ones there) and the old one the spare.
fn swapStreams(ch: *Chunk, src: u64) void {
    std.debug.assert(src == ch.h_alt);
    ch.h_alt = ch.h;
    ch.h = src;
}

/// A row range: `start`, the count `n`. WP4's qhcRange and WP5's row-sharded gather() both key owned-row
/// transport off the identical range, so it is named once here rather than as two matching anonymous structs.
pub const QRows = struct { start: usize, n: usize };

/// WP4 (quarter-HC, docs/prefill-plan.md): this layer's owned row range on this node, when quarter-HC applies --
/// a full-cap 2D chunk, strictly before the replay-cut boundary layer (c.layers / 2, which needs every row's h /
/// pre / x globally and is never sharded: replayCut truncates ch.n there for the decoder's own, unchanged path).
/// node g of `world` owns rows [g * q, (g + 1) * q), q = cap / world. A short chunk, a resumed chunk under cap
/// rows, a non-2D engine or the flag off (agreeQuarterHc) all fall through to today's unsharded path (null).
fn qhcRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    const t = e.two orelse return null;
    if (!t.qhc) return null;
    if (li >= e.c.layers / 2) return null;
    if (ch.n != ch.cap) return null;
    const world = e.comm.world;
    if (world == 0 or ch.cap % world != 0) return null;
    const q = ch.cap / world;
    return .{ .start = e.comm.rank * q, .n = q };
}

/// WP4: `full` (fp32 [e.world, ch.cap, D], i.e. ch.ga or ch.gm) repacked at rows [r.start, r.start + r.n) of every
/// plane into ch.qhc_g, tightly strided [e.world, r.n, D] -- see ch.qhc_g's own comment for why. A local
/// device-to-device copy (no NCCL), one per plane.
fn qhcRepackG(e: *const Engine, ch: *Chunk, full: u64, r: anytype) !void {
    const row_bytes = e.c.hidden * 4;
    for (0..e.world) |w| {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(
            ch.qhc_g + w * r.n * row_bytes,
            full + w * ch.cap * row_bytes + r.start * row_bytes,
            r.n * row_bytes,
            e.s.handle,
        ), "cuMemcpyDtoDAsync");
    }
}

/// WP4: all-gather this node's local r.n-row slice of a [cap, width] buffer (ch.x, bf16; or ch.taps' one tap's rows)
/// into every rank's full copy. An ordinary world-4 all-gather over the base communicator (not Two's column / rank
/// reconstruction: the buffer is already fully reduced and un-column-sharded at this point), in place -- the send
/// pointer is already exactly where this rank's slice belongs in the recv buffer (r.start = rank * r.n). NCCL's
/// documented in-place rule (send == recv + rank * count); a cross-communicator timing race and a missing
/// synchronize were both tested live and ruled out as the cause of WP4's real bug (see qhcResyncBoundary).
fn qhcGatherRows(e: *const Engine, buf: u64, width: usize, r: anytype) !void {
    try e.comm.allGather(buf + r.start * width * 2, buf, r.n * width, .bf16, e.s);
}

/// WP4: before the replay-cut boundary layer (li == c.layers / 2), resync ch.h, ch.pre, ch.post and ch.comb to
/// full width if the last layer (li - 1) was quarter-HC row-sharded. replayCut's attnMixes call runs at li on the
/// FULL ch.n with ch.pending still true (the boundary layer is never row-sharded itself): that is the fused
/// hc_pre_pf path, which does not just overwrite post/comb as fresh output -- it reads them first, to fuse
/// layer (li - 1)'s posted MoE gather into the streams (`_hc_mix_pf`'s POST/COMB arguments), before `_hc_finish`
/// writes the new ones. Gathering only h and pre (as this function first did) left post/comb valid for only this
/// node's own owned rows, so attnMixes(li) silently fused garbage for the other three quarters of the chunk --
/// found by two independent code reviews (Codex, Grok) after three live hypotheses (a G-buffer/PART-buffer stride
/// bug, an NCCL warm-up gap, a missing synchronize) were each fixed or tested and none resolved the live failure.
/// A plain world-4 all-gather of each rank's own correctly-computed owned rows (never of the other ranks'
/// don't-care garbage rows, which this node never wrote and never reads).
pub fn qhcResyncBoundary(e: *const Engine, ch: *Chunk, li: usize) !void {
    if (li == 0) return;
    if (qhcRange(e, ch, li - 1)) |r| {
        const c = e.c;
        try e.comm.allGather(ch.h + r.start * c.hc * c.hidden * 2, ch.h, r.n * c.hc * c.hidden, .bf16, e.s);
        try e.comm.allGather(ch.pre + r.start * c.hc * 4, ch.pre, r.n * c.hc, .f32, e.s);
        try e.comm.allGather(ch.post + r.start * c.hc * 4, ch.post, r.n * c.hc, .f32, e.s);
        try e.comm.allGather(ch.comb + r.start * c.hc * c.hc * 4, ch.comb, r.n * c.hc * c.hc, .f32, e.s);
    }
}

/// Layer `li`'s attention mixes: x = RMSNorm of the streams mixed by pre, pre_a / post / comb from their own mixes;
/// with a pending MoE gather, hc_pre_pf posts it into the streams first.
pub fn attnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (qhcRange(e, ch, li)) |r| {
        const row_h = c.hc * c.hidden * 2;
        const row_p = c.hc * 4;
        const row_c = c.hc * c.hc * 4;
        if (ch.pending) {
            try qhcRepackG(e, ch, ch.gm, r);
            _ = try tri_hc.hcPrePf(e.t, ch.h + r.start * row_h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre + r.start * row_p, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_a + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, .{ .gathered = ch.qhc_g, .world = e.world, .h_out = ch.h_alt + r.start * row_h }, r.n, c.hidden);
            // swapStreams asserts its argument is exactly ch.h_alt (the whole buffer); the kernel's `h_out` here was
            // offset to this node's own rows (correct for the write it actually did), so swap the base pointers
            // directly instead of going through that helper's assert.
            const old_h = ch.h;
            ch.h = ch.h_alt;
            ch.h_alt = old_h;
            ch.pending = false;
        } else {
            try tri_basic.hcPre(e.t, ch.h + r.start * row_h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre + r.start * row_p, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_a + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, r.n, c.hidden);
        }
        try qhcGatherRows(e, ch.x, c.hidden, r);
    } else if (ch.pending) {
        const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, .{ .gathered = ch.gm, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
        swapStreams(ch, src);
        ch.pending = false;
    } else {
        try tri_basic.hcPre(e.t, ch.h, lay.hc_attn[0], lay.hc_attn[1], lay.hc_attn[2], ch.pre, lay.attn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_a, ch.post, ch.comb, ch.part, ch.n, c.hidden);
    }
}

/// TF_DS_DUMP=DIR (a debugging aid, off in service): TF_DS_DUMP_LAYER's layers' (a comma list, default just "0")
/// intermediates of the first prompt chunk, each written as DIR/n<node>-<name>.bin (raw device bytes) after the
/// stream drains. WP1's exit gate (a tensor dump at layers 0, 8 and 19) is TF_DS_DUMP_LAYER=0,8,19.
pub var dump_dir: ?[:0]const u8 = null;
pub var dump_layers: [8]i64 = .{ 0, -1, -1, -1, -1, -1, -1, -1 };
pub var dump_done: bool = false;
pub var dump_node: u32 = 0;

fn wantLayer(li: usize) bool {
    const l: i64 = @intCast(li);
    for (dump_layers) |d| {
        if (d == l) return true;
    }
    return false;
}

pub fn dump(e: *const Engine, li: usize, name: []const u8, ptr: u64, bytes: usize) void {
    const dir = dump_dir orelse return;
    if (!wantLayer(li) or dump_done or bytes == 0) return;
    e.s.synchronize() catch return;
    const host = std.heap.page_allocator.alloc(u8, bytes) catch return;
    defer std.heap.page_allocator.free(host);
    e.d.check(e.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH") catch return;
    var buf: [512]u8 = undefined;
    const path = std.mem.printSentinel(&buf, "{s}/n{d}-{s}.bin", .{ dir, dump_node, name }, 0) catch return;
    const f = std.c.fopen(path.ptr, "wb") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(host.ptr, 1, bytes, f);
}

/// mm(layer, x, out_dtype) of m rows: the prompt GEMM beyond 128 rows, the layer's own group (GROUPED) up to 128.
pub fn mmRows(e: *const Engine, ch: *const Chunk, m: usize, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    const dt: exl3_linear.DType = switch (out_type) {
        .bf16 => .bf16,
        .fp32 => .f32,
    };
    // Format.native: one kernel for every row count (a row's bits at any m)
    if (e.mx) |k| return mx.linear(k, e.s, ch.mxs, l, x, .bf16, ldx, m, out, os, dt, false, null);
    if (m > 128) return exl3_prefill.matmul(e.pf, e.t, ch.ws, l, x, .bf16, ldx, out, out_type, os, m);
    try grouped(e, ch, m, &.{l}, &.{x}, &.{ldx}, &.{out}, &.{os}, &.{dt});
}

/// mm of the chunk's rows.
pub fn mm(e: *const Engine, ch: *const Chunk, l: weights.Linear, x: u64, ldx: usize, out: u64, out_type: tri_markov.OutType, os: usize) !void {
    try mmRows(e, ch, ch.n, l, x, ldx, out, out_type, os);
}

/// attention_k: the partial pa [n, D] fp32 of this rank's heads; the chunk's keys go into the layer's window ring
/// (bf16 [window + RING_EXTRA, head_dim]). A compressed layer also attends to its kv source's latents the indexer picks
/// (a kv source first compresses the chunk into them). `floor`: no window key before this position.
pub fn attention(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, ring: u64, floor: usize, kv_done: bool) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const hl = e.heads();
    // _cs: the layer's RoPE kind (compressed layers rotate by the compressed table) for every rotation it makes
    const rope = if (lay.ratio != 0) e.compressed else e.plain;
    const ring_rows = c.window + ring_extra;
    // ring mode (a chunk of RING_EXTRA rows or fewer): the rows' keys into the ring first, the attention reads the ring
    const ring_mode = n <= ring_extra;
    // attn_in: wq_a and wkv of x, one prompt GEMM each, or one group up to 128 rows
    if (n > 128) {
        try mm(e, ch, lay.wq_a, ch.x, c.hidden, ch.qa, .bf16, lay.wq_a.n);
        try mm(e, ch, lay.wkv, ch.x, c.hidden, ch.y, .bf16, hd);
    } else {
        try grouped(e, ch, n, &.{ lay.wq_a, lay.wkv }, &.{ ch.x, ch.x }, &.{ c.hidden, c.hidden }, &.{ ch.qa, ch.y }, &.{ lay.wq_a.n, hd }, &.{ .bf16, .bf16 });
    }
    pmark(e, .a_in);
    dump(e, li, "x", ch.x, n * c.hidden * 2);
    dump(e, li, "qa", ch.qa, n * lay.wq_a.n * 2);
    dump(e, li, "y", ch.y, n * hd * 2);
    try tri_basic.rmsnorm(e.t, ch.qa, lay.wq_a.n, lay.q_norm, ch.qr, lay.wq_a.n, c.eps, n, lay.wq_a.n);
    dump(e, li, "qr", ch.qr, n * lay.wq_a.n * 2);
    try mm(e, ch, lay.wq_b, ch.qr, lay.wq_a.n, ch.q, .bf16, hl * hd);
    dump(e, li, "q_norope", ch.q, n * hl * hd * 2);
    try tri_norm.ropeHeads(e.t, ch.q, rope.cos, rope.sin, ch.pos, rd, false, n, hl, hd);
    pmark(e, .a_q);
    // the window keys: the ring's last window - 1 positions before start, then this chunk's rows
    const lo = @max(floor, start -| (c.window - 1));
    const wsrc_rows = if (ring_mode) ring_rows else start - lo + n;
    if (ring_mode) {
        // kv_norm_rope into ring rows pos % R (and its rows out), the attention's window keys the ring, WLO zero
        var slots: [ring_extra]i64 = undefined;
        for (0..n) |j| slots[j] = @intCast((start + j) % ring_rows);
        try upload(e, ch.rslot, &slots, n * 8);
        try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.rslot, c.eps, true, rd, ch.wsrc, n, hd);
    } else if (start > lo) {
        // wsrc[:start - lo] = ring[arange(lo, start) % R]
        var ridx: [256]i64 = undefined;
        if (start - lo > ridx.len) return error.WindowTooLong;
        for (0..start - lo) |j| ridx[j] = @intCast((lo + j) % ring_rows);
        try upload(e, ch.ring_idx, &ridx, (start - lo) * 8);
        try fill(e, ch.invalid, 0, 4);
        try e.ops.gatherRows(e.s, ring, ring_rows, ch.ring_idx, ch.wsrc, hd * 2, start - lo, ch.invalid);
    }
    var lo64: i64 = if (ring_mode) 0 else @intCast(lo);
    try upload(e, ch.wlo, &lo64, 8);
    const kv = ch.wsrc + (start - lo) * hd * 2;
    // slots -1: the keys go to wsrc only (ring_mode is off); the ring is written after the attention
    if (!ring_mode) try tri_norm.kvNormRope(e.t, ch.y, lay.kv_norm, rope.cos, rope.sin, ch.pos, ring, ring_rows, ch.neg, c.eps, true, rd, kv, n, hd);
    var comp: tri_attn.Comp = .none;
    var n_idx: usize = 0;
    var padded = false;
    if (lay.ratio != 0) {
        const r: usize = lay.ratio;
        if (lay.comp_wkv != null and !kv_done) try kvSourceUpdate(e, ch, cs, sh, li);
        const src = sh.kv_layer orelse return error.NoKvSource;
        const n_comp_end = (start + n) / r;
        // vis = (pos + 1) // ratio: the compressed entries row i may see (host-made: the same integers)
        var vis_host: [4096]i64 = undefined;
        if (n > vis_host.len) return error.ChunkTooLong;
        for (0..n) |i| vis_host[i] = @intCast((start + i + 1) / r);
        try upload(e, ch.vis, &vis_host, n * 8);
        if (lay.idx_wq_b) |wqb| try indexer(e, ch, cs, sh, li, wqb, src, n_comp_end, rope);
        const kk = sh.kk orelse return error.NoIndexerSelection;
        comp = .{ .fp4 = .{ .codes = cs.comp_codes[src], .scales = cs.comp_scales[src] } };
        n_idx = kk;
        if (kk % 16 != 0 and n > tri.decode_rows) {
            // sparse_attn: a prompt chunk's pick list padded with -1 to whole 16-column tiles (F.pad), read from there
            const kp = (kk + 15) / 16 * 16;
            try fill(e, ch.cidxp, 0xff, n * kp * 8);
            try e.ops.copyRows(e.s, ch.cidx, kk * 8, ch.cidxp, kp * 8, kk * 8, n);
            padded = true;
        }
    }
    pmark(e, .a_idx);
    try tri_attn.sparseAttn(e.t, .{
        .q = ch.q,
        .out = ch.o,
        .rows = n,
        .h = hl,
        .hd = hd,
        .sink = lay.sink,
        .wsrc = if (ring_mode) ring else ch.wsrc,
        .wsrc_rows = wsrc_rows,
        .wlo = ch.wlo,
        .ring = ring_mode,
        .parts = .{ .pm = ch.pm_, .pl = ch.pl_, .po = ch.po_ },
        .comp = comp,
        .idx = if (n_idx > 0) (if (padded) ch.cidxp else ch.cidx) else 0,
        .n_idx = n_idx,
        .pos = ch.pos,
        .scale = scale(hd),
        .window = c.window,
    });
    try tri_norm.ropeHeads(e.t, ch.o, rope.cos, rope.sin, ch.pos, rd, true, n, hl, hd);
    // ring[pos[-keep:] % R] = kv[-keep:] (ring mode wrote them before the attention)
    const keep: usize = if (ring_mode) 0 else @min(n, ring_rows);
    if (keep > 0) {
        var idx: [256]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % ring_rows);
        try upload(e, ch.ring_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, kv + (n - keep) * hd * 2, hd * 2, ch.ring_idx, ring, hd * 2, hd * 2, keep);
    }
    pmark(e, .a_core);
    dump(e, li, "o", ch.o, n * hl * hd * 2);
    // wo_a: each group's column block of o read in place, written into its column block of u; then wo_b to fp32
    const groups = lay.groups;
    const gk = hl * hd / groups;
    const uw = groups * lay.wo_a[0].n;
    var col: usize = 0;
    if (e.mx) |k| {
        // Format.native: o quant-dequantized, then each group's bf16 einsum (Attention.forward)
        try mx.woA(k, e.s, ch.mxs, lay.wo_a[0..groups], ch.o, hl * hd, n, ch.u, uw, null);
        dump(e, li, "o_fake", ch.mxs.fake, n * hl * hd * 2);
        dump(e, li, "wo_a0", lay.wo_a[0].w, @as(usize, lay.wo_a[0].n) * lay.wo_a[0].k * 2);
    } else if (n > 128) {
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            try mm(e, ch, wo, ch.o + g * gk * 2, hl * hd, ch.u + col * 2, .bf16, uw);
            col += wo.n;
        }
    } else {
        // one group: the column blocks of o read in place, written into u's column blocks
        var xs: [4]u64 = undefined;
        var outs: [4]u64 = undefined;
        for (lay.wo_a[0..groups], 0..) |wo, g| {
            xs[g] = ch.o + g * gk * 2;
            outs[g] = ch.u + col * 2;
            col += wo.n;
        }
        const lds = [_]usize{ hl * hd, hl * hd, hl * hd, hl * hd };
        const oss = [_]usize{ uw, uw, uw, uw };
        const dts = [_]exl3_linear.DType{ .bf16, .bf16, .bf16, .bf16 };
        try grouped(e, ch, n, lay.wo_a[0..groups], xs[0..groups], lds[0..groups], outs[0..groups], oss[0..groups], dts[0..groups]);
    }
    pmark(e, .a_woa);
    dump(e, li, "u", ch.u, n * uw * 2);
    if (e.two) |t| {
        try t.woB(e, ch, lay); // 2D: the column partner's groups first, then this pair's columns
        dump(e, li, "u_full", t.u_full, n * (t.uw[0] + t.uw[1]) * 2);
        dump(e, li, "pa", ch.pa, n * lay.wo_b.n * 4);
        return;
    }
    try mm(e, ch, lay.wo_b, ch.u, uw, ch.pa, .fp32, c.hidden);
}

/// kv_source_update of a kv-source layer on the chunk (no earlier pending row: a chunk starting on a group boundary):
/// _compress's new latents (ratio 1: RMSNorm of the bf16 projection; ratio 2: of the pairs' softmax-weighted kv), its
/// positional store's last RAW rows, then the index keys (RMSNorm of idx_wk's projection, RoPE) and the RoPE'd latents
/// into the caches as FP4 (fp4_store).
fn kvSourceUpdate(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const start = ch.start;
    const hd = c.head_dim;
    const rd = c.rope_dim;
    const r: usize = lay.ratio;
    const cwkv = lay.comp_wkv.?;
    if (start % r != 0) return notPorted(@src()); // the group's earlier rows from the positional store
    const full = n / r;
    if (r == 1) {
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.lat2, .bf16, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, n, hd, c.eps);
    } else {
        if (r != 2) return notPorted(@src());
        try mm(e, ch, cwkv, ch.x, c.hidden, ch.kvc, .fp32, hd);
        try mm(e, ch, lay.comp_wgate.?, ch.x, c.hidden, ch.scc, .fp32, hd);
        // rk[pw] = kv[-keep:], rs[pw] = score[-keep:]: pw = positions start + n - keep .. % RAW
        const keep: usize = @min(n, raw_rows);
        var idx: [raw_rows]i64 = undefined;
        for (0..keep) |j| idx[j] = @intCast((start + n - keep + j) % raw_rows);
        try upload(e, ch.raw_idx, &idx, keep * 8);
        try e.ops.scatterRows(e.s, ch.kvc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_kv[li], hd * 4, hd * 4, keep);
        try e.ops.scatterRows(e.s, ch.scc + (n - keep) * hd * 4, hd * 4, ch.raw_idx, cs.raw_score[li], hd * 4, hd * 4, keep);
        if (full == 0) { // a one-row chunk (a prompt one past a chunk end): its row waits in the store, no new group
            sh.kv_layer = li;
            return;
        }
        try e.exact.compress2(e.s, ch.kvc, ch.scc, ch.lat2, full, hd);
        try e.exact.rmsNorm(e.s, ch.lat2, hd, lay.comp_norm, ch.lat, hd, full, hd, c.eps);
    }
    sh.kv_layer = li;
    // groups first // r .. and their positions group * r (the rotations _rot reads)
    var g_host: [4096]i64 = undefined;
    var p_host: [4096]i64 = undefined;
    if (full > g_host.len) return error.ChunkTooLong;
    for (0..full) |j| {
        g_host[j] = @intCast(start / r + j);
        p_host[j] = @intCast((start / r + j) * r);
    }
    try upload(e, ch.groups, &g_host, full * 8);
    try upload(e, ch.gpos, &p_host, full * 8);
    const rot = e.compressed;
    if (lay.idx_wk) |iwk| {
        const id = c.index_head_dim;
        try mmRows(e, ch, full, iwk, ch.lat, hd, ch.lat2, .bf16, id);
        try e.exact.rmsNorm(e.s, ch.lat2, id, lay.idx_k_norm, ch.ik, id, full, id, c.eps);
        try e.exact.rope(e.s, ch.ik, id, id - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
        // switch "comp": the packed cache's rows in one launch (fp4_store: store_rows' bytes)
        try tri_attn.fp4Store(e.t, ch.ik, id, cs.idx_codes[li], cs.idx_scales[li], ch.groups, full, id, 32, false);
    }
    try e.ops.copyRows(e.s, ch.lat, hd * 2, ch.lat2, hd * 2, hd * 2, full);
    try e.exact.rope(e.s, ch.lat2, hd, hd - rd, rot.cos, rot.sin, ch.gpos, full, rd / 2, false);
    try tri_attn.fp4Store(e.t, ch.lat2, hd, cs.comp_codes[li], cs.comp_scales[li], ch.groups, full, hd, 16, true);
}

/// WP6(b) (docs/prefill-plan.md, indexer row-sharding): this layer's owned row range on this node, when
/// TF_DS_IDX_SHARD applies -- a full-cap 2D chunk, strictly before the candidate-source layer (li <
/// c.candidate_source). That bound is not coincidental: it is the indexer's own switch point, after which the
/// per-row "keyed" scoring this shards stops and a cross-layer pool (ch.pool, written once at li ==
/// candidate_source over every row, read by every later layer) takes over -- excluded here by construction, so
/// no boundary resync like WP4's is needed. A short chunk, a resumed chunk under cap rows, a non-2D engine, an
/// unaligned quarter (the kit's index kernels need every block's rows to start on a 16-row boundary) or the
/// flag off (agreeIdxShard) all fall through to today's unsharded path (null).
fn idxShardRange(e: *const Engine, ch: *const Chunk, li: usize) ?QRows {
    const t = e.two orelse return null;
    if (!t.idx_shard) return null;
    if (li >= e.c.candidate_source) return null;
    if (ch.n != ch.cap) return null;
    const world = e.comm.world;
    if (world == 0 or ch.cap % world != 0) return null;
    const q = ch.cap / world;
    if (q % 16 != 0) return null;
    return .{ .start = e.comm.rank * q, .n = q };
}

/// The indexer of a layer with index queries (switch "prompt_keys", layers before the candidate source): the
/// queries (idx_wq_b of q's latent, RoPE, fp4_qd's bytes), the heads' weights (x.float() @ idx_proj.t() through cuBLAS,
/// to bf16, times idx_dim ** -0.5 * idx_heads ** -0.5), each row's int64 keys of the kv source's index keys and the
/// top-k selection into cidx.
/// WP6(b): when idxShardRange applies, the row-block loop below covers only this node's owned quarter (the same
/// pointer-offset-into-the-full-buffer pattern it already uses for block 2, 3, ... of any multi-block call --
/// see the loop's own "a row's scores and picks are its own" comment, and the offline proof,
/// tf-dsv41-wp6b-idx-shard-replay, for a nonzero FIRST block), and ch.cidx is all-gathered back to full width
/// before this returns, so every caller sees the same fully-populated cidx either way.
fn indexer(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, wqb: weights.Linear, src: usize, n_comp_end: usize, rope: Rope) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const ih = c.index_heads;
    const id = c.index_head_dim;
    if (n_comp_end == 0) { // cidx [n, 0]: no compressed entry yet (a one-token prompt), the attention reads none
        sh.kk = 0;
        return;
    }
    if (n_comp_end > ch.max_comp) return error.ChunkTooLong;
    try mm(e, ch, wqb, ch.qr, lay.wq_a.n, ch.iq, .bf16, ih * id);
    try tri_norm.ropeHeads(e.t, ch.iq, rope.cos, rope.sin, ch.pos, c.rope_dim, false, n, ih, id);
    // switch "idx": fp4_qd's bytes in one launch
    try tri_attn.fp4QdP2(e.t, ch.iq, ch.iq4, n * ih * id);
    if (e.mx) |k| {
        // Format.native: weights_proj, a bf16 Linear, times the scale (Indexer.forward), at any row count
        try mx.bf16Gemm(k, e.s, ch.x, c.hidden, lay.idx_proj, c.hidden, n, ih, c.hidden, ch.iw, ih, .bf16, exact.indexScale(id, ih), true, 1, 0, 0, 0, null);
    } else {
        if (n <= tri.decode_rows) {
            // a decode-sized chunk: the row-invariant matmul of the fp16 projection (rowmm2)
            try tri_norm.rowmm2(e.t, ch.x, c.hidden, lay.idx_proj_h, ch.wl, n, c.hidden, ih);
        } else {
            try e.ops.toF32(e.s, ch.x, ch.xf, n * c.hidden);
            try e.blas.xwT(ch.xf, lay.idx_proj, ch.wl, n, c.hidden, ih);
        }
        try e.exact.bf16Scale(e.s, ch.wl, ch.iw, n * ih, exact.indexScale(id, ih));
    }
    const kk = @min(c.index_topk, n_comp_end);
    // rows in blocks so the [rows, n_comp] keys stay bounded at long contexts; a block a multiple of 16 rows, so every
    // block's views (vis + r0 * 8 and the rest) start on a 16-byte boundary, the alignment the kit's index kernels were
    // captured for (a 137,612-token prompt gave 121-row blocks: MissingTritonVariant at the second). A row's scores and
    // picks are its own, so the block size changes no bits.
    const rb = @max(16, @min(n, (@as(usize, 1) << 26) / (4 * @max(n_comp_end, 1)) / 16 * 16));
    if (rb * n_comp_end > key_elems) return error.ChunkTooLong;
    const keyed = li != c.candidate_source and !(c.candidate_source < li) and kk & (kk - 1) == 0;
    const k: tri_index.IndexK = .{ .fp4 = .{ .codes = cs.idx_codes[src], .scales = cs.idx_scales[src] } };
    const torch_topk: tri_index.TopK = .{ .top = ch.ktop, .ctx = @ptrCast(@constCast(e)), .run = torchTopk };
    if (keyed and tri_index.prunes(n_comp_end, kk) and (rb > pruned_rows or ch.pbufs.cand == 0)) return error.ChunkTooLong;
    // the candidate source and the layers after it: fp32 scores, the pool (_candidates at the source, apply_candidates
    // after it), topk_indices and -1 past vis. While every block of the scores fits in the pool (cand_blocks of them)
    // the pool keeps each block with a finite score (the newest pinned: it holds the row's last visible key) and
    // apply_candidates rewrites only scores already -inf: no launch changes a byte. Past it the source's block maxima
    // (the newest block pinned to +inf) pick the pool's cand_blocks blocks (topk_indices: ties to the lower block), the
    // picked blocks with a finite maximum make the chunk's mask, and the later layers' scores outside it become -inf.
    const cb = c.candidate_block;
    const nbk = (n_comp_end + cb - 1) / cb;
    const pooled = !keyed and nbk > c.candidate_blocks;
    if (pooled and (ch.pool == 0 or nbk > ch.pool_ms)) return error.ChunkTooLong;
    const pk = @min(c.candidate_blocks, nbk);
    const idxr = idxShardRange(e, ch, li);
    var r0: usize = if (idxr) |r| r.start else 0;
    const r_end: usize = if (idxr) |r| r.start + r.n else n;
    while (r0 < r_end) : (r0 += rb) {
        // the block's rows: iq[r0:r1], wts[r0:r1] and vis[r0:r1] as views (their offsets the served pointers'), its
        // selection into cidx[r0:r1]
        const m = @min(r_end, r0 + rb) - r0;
        const q = ch.iq4 + r0 * ih * id * 2;
        const wb = ch.iw + r0 * ih * 2;
        const vr = ch.vis + r0 * 8;
        const out = ch.cidx + r0 * kk * 8;
        if (keyed) {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.keys, null, true, ch.tmax, false, m, ih, id);
            try tri_index.topkSelectPruned(e.t, ch.keys, n_comp_end, ch.tmax, kk, vr, out, ch.pbufs, torch_topk, m, n_comp_end);
        } else {
            try tri_index.indexScore(e.t, q, k, wb, vr, n_comp_end, ch.score, null, false, null, false, m, ih, id);
            const mrows = ch.pool + r0 * ch.pool_ms;
            if (pooled and li == c.candidate_source) {
                try e.ops.blockMax(e.s, ch.score, n_comp_end, n_comp_end, cb, vr, ch.bmax, nbk, m);
                try e.ops.topkIndices(e.s, ch.bmax, nbk, m, nbk, pk, ch.every, ch.pidx);
                try e.ops.poolPick(e.s, ch.bmax, nbk, nbk, ch.pidx, pk, mrows, ch.pool_ms, 0, m);
            } else if (pooled and c.candidate_source < li) {
                if (!sh.pool) return error.NoCandidatePool;
                try e.ops.applyPool(e.s, ch.score, n_comp_end, n_comp_end, cb, mrows, ch.pool_ms, m);
            }
            try e.ops.topkIndices(e.s, ch.score, n_comp_end, m, n_comp_end, kk, vr, out);
        }
    }
    if (pooled and li == c.candidate_source) sh.pool = true;
    // WP6(b): this rank's owned-row slice of cidx, all-gathered to every rank -- a plain world-4 all-gather, in
    // place (NCCL's documented rule: send == recv + rank * count), the same pattern qhcGatherRows already uses
    // for ch.x. Bytes, not a typed count: cidx's per-row width (kk int64 keys) is uniform across rows for this
    // call, so a raw byte gather needs no element-type opinion.
    if (idxr) |r| try e.comm.allGather(ch.cidx + r.start * kk * 8, ch.cidx, r.n * kk * 8, .u8, e.s);
    sh.kk = kk;
}

/// topk_select's torch step, keys.topk(k).values: the top-k set of the unique int64 keys (ops.topkI64).
pub fn torchTopk(ctx: ?*anyopaque, _: tri.Tri, keys: u64, ks: usize, rows: usize, n: usize, k: usize, top: u64) anyerror!void {
    const e: *const Engine = @ptrCast(@alignCast(ctx.?));
    try e.ops.topkI64(e.s, keys, ks, rows, n, k, top);
}

/// hd ** -0.5 as Triton passes the Python float: rounded to fp32.
pub fn scale(hd: usize) f32 {
    return @floatCast(std.math.pow(f64, @floatFromInt(hd), -0.5));
}

/// Comm.gather: [world, n, D] fp32 of every rank's partial, in rank order.
/// TF_DS_PAIR_SUM=1 (prompt2d Two.pairSum, agreed at startup): a chunk's gathers as pair sums over the ring's links.
/// Every rank holds the same pair_sum and the same chunk rows, so all take the same path.
/// WP5 (docs/prefill-plan.md, owner-row transport, agreed as owner_gather): `consumer_li` is the layer index
/// whose attnMixes/ffnMixes/tap/engramApply call will actually read `dst` back (the SAME li for the attention
/// gather into ch.ga; li + 1 for the MoE gather into ch.gm, which the next layer reads as its "pending" post --
/// see qhcResyncBoundary's own comment on why that is never the boundary layer itself). qhcRange(consumer_li)
/// is exactly the range that reader will later pull back out with qhcRepackG, so handing it to pairSumOwned
/// keeps the producer and every consumer locked to the identical row range by construction.
pub fn gather(e: *const Engine, ch: *const Chunk, src: u64, dst: u64, consumer_li: usize) !void {
    if (e.two) |t| {
        if (t.owner_gather and t.pair_sum and ch.n >= 64) {
            if (qhcRange(e, ch, consumer_li)) |r| return t.pairSumOwned(e, src, dst, ch.n, t.ow, r);
        }
        if (t.pair_sum and ch.n >= 64) return t.pairSum(e, src, dst, ch.n, t.ow);
        return t.quarters(e, src, dst, ch.n, t.ow, 4); // 2D: the same [2, n, D] from the four quarters
    }
    try e.comm.allGather(src, dst, ch.n * e.c.hidden, .f32, e.s);
}

/// Layer `li`'s FFN mixes: hc_pre_pf with the gathered attention partials posted into the streams first; x the MoE's
/// input rows, pre_f / post / comb the FFN's.
pub fn ffnMixes(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    if (qhcRange(e, ch, li)) |r| {
        const row_h = c.hc * c.hidden * 2;
        const row_p = c.hc * 4;
        const row_c = c.hc * c.hc * 4;
        try qhcRepackG(e, ch, ch.ga, r);
        _ = try tri_hc.hcPrePf(e.t, ch.h + r.start * row_h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a + r.start * row_p, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x + r.start * c.hidden * 2, ch.pre_f + r.start * row_p, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.qhc_part, .{ .gathered = ch.qhc_g, .world = e.world, .h_out = ch.h_alt + r.start * row_h }, r.n, c.hidden);
        const old_h = ch.h;
        ch.h = ch.h_alt;
        ch.h_alt = old_h;
        try qhcGatherRows(e, ch.x, c.hidden, r);
        return;
    }
    const src = try tri_hc.hcPrePf(e.t, ch.h, lay.hc_ffn[0], lay.hc_ffn[1], lay.hc_ffn[2], ch.pre_a, lay.ffn_norm, c.eps, c.hc_eps, c.hc_iters, ch.x, ch.pre_f, ch.post, ch.comb, ch.part, .{ .gathered = ch.ga, .world = e.world, .h_out = ch.h_alt }, ch.n, c.hidden);
    swapStreams(ch, src);
}

/// Model.moe of a prompt chunk: the gate's logits in fp32 (x.float() @ gate_w.float().t(), one cuBLAS GEMM as torch
/// issues it), the routing (the shared expert in every row's last slot) and the routed experts: the partial pm [n, D].
pub fn moe(e: *const Engine, ch: *Chunk, li: usize) !void {
    const lay = e.w.layers[li];
    const c = e.c;
    const n = ch.n;
    const d = c.hidden;
    const sl = e.slots();
    const shared_id = lay.experts.count - 1;
    var kc: usize = 0;
    if (e.mx) |k| {
        // Format.native: Gate.forward's fp32 logits of the bf16 rows and weight (exact products, fp32 sums), any rows
        try mx.bf16Gemm(k, e.s, ch.x, d, lay.gate_w, d, n, c.experts, d, ch.logits, c.experts, .f32, 1, false, 1, 0, 0, 0, null);
        try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
        try routeSpans(e, ch, lay, shared_id, sl, 0);
        pmark(e, .m_rt);
        dump(e, li, "mx", ch.x, n * d * 2);
        dump(e, li, "logits", ch.logits, n * c.experts * 4);
        dump(e, li, "pick", ch.pick, n * sl * 4);
        dump(e, li, "wts", ch.wts, n * sl * 4);
        if (e.two) |t| {
            try t.experts(e, ch, lay);
            const pack = lay.experts.width + t.ir;
            dump(e, li, "xd", ch.xs.xd, n * sl * pack * 2);
            dump(e, li, "xd_full", t.xd_full, n * sl * lay.experts.down_k * 2);
            dump(e, li, "pm", ch.pm, n * lay.experts.down_n * 4);
            return;
        }
        try mx.gateUp(k, e.s, e.d, lay.experts, ch.mxe, ch.x, d, ch.pick, ch.wts, n, c.swiglu_limit, ch.xs.xd, lay.experts.width, false);
        return mx.down(k, e.s, lay.experts, ch.mxe, ch.xs.xd, lay.experts.width, ch.pick, n, ch.pm, d, false, false);
    }
    if (n <= tri.decode_rows) {
        // a decode-sized chunk: the row-invariant gate (rowmm_gate's chunk sums) and its routing
        kc = try tri_norm.rowmmGate(e.t, ch.x, d, lay.gate_w, ch.gl, n, d, c.experts);
        try tri_norm.route(e.t, ch.gl, kc, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    } else {
        try e.ops.toF32(e.s, ch.x, ch.xf, n * d);
        try e.ops.f16ToF32(e.s, lay.gate_w, ch.gate_f, c.experts * d);
        try e.blas.xwT(ch.xf, ch.gate_f, ch.logits, n, d, c.experts);
        try tri_norm.route(e.t, ch.logits, 0, lay.gate_b, c.top_k, c.routed_scaling, shared_id, ch.pick, ch.wts, n, c.experts, sl);
    }
    try routeSpans(e, ch, lay, shared_id, sl, kc);
    if (e.two) |t| return t.experts(e, ch, lay); // 2D: gate / up here, the intermediate's exchange, down here
    // routed(): a chunk of fewer than EXACT_ROWS rows takes the decode window's fused path and scratch
    if (n < exl3_experts.exact_rows) return exl3_experts.decode(e.ex, e.s, lay.experts, ch.xsd, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
    try exl3_experts.prompt(e.ex, e.s, lay.experts, ch.xs, ch.x, d, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
}

/// The VL routing (Gate.forward's bias = where(image_mask, bias_vl, bias)) as the served build runs it (recording v0's
/// launches): every row routed with the text bias, then each span's rows of the gate's output (the logits, or a
/// decode-sized chunk's chunk sums [n, kc, NE]) routed again with the image bias in a launch of their own, their picks and
/// weights put over the text ones. Row by row the routing is independent, so the text rows keep theirs.
fn routeSpans(e: *const Engine, ch: *Chunk, lay: weights.Layer, shared_id: usize, sl: usize, kc: usize) !void {
    if (ch.spans.len == 0) return;
    if (lay.gate_b_vl == 0) return error.NoImageRoutingBias;
    const c = e.c;
    const n = ch.n;
    const src = if (n <= tri.decode_rows and kc != 0) ch.gl else ch.logits;
    const row = @max(kc, 1) * c.experts * 4;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, n) orelse continue;
        const rows = o[1] - o[0];
        try tri_norm.route(e.t, src + o[0] * row, kc, lay.gate_b_vl, c.top_k, c.routed_scaling, shared_id, ch.pick_vl, ch.wts_vl, rows, c.experts, sl);
        const dd = e.d;
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.pick + o[0] * sl * 4, ch.pick_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
        try dd.check(dd.api.cuMemcpyDtoDAsync_v2(ch.wts + o[0] * sl * 4, ch.wts_vl, rows * sl * 4, e.s.handle), "cuMemcpyDtoDAsync");
    }
}

/// Engram on the host: the hasher (the compressed token map, the multipliers), this rank's hash columns [lo, hi) of
/// each Engram layer's, its tables and readers, and host buffers for a chunk's rows (model.py's Engram).
pub const EngramHost = struct {
    hasher: engram.Hasher,
    tables: *const engram_io.Tables,
    pool: *engram_io.Pool,
    lo: usize,
    hi: usize,
    hashes: []i64, // [cap][layers][cols]
    flat: []i64, // [cap * (hi - lo)]
    w: []u8, // FP8 rows
    s: []u8, // their E8M0 scales
    rows: []u16, // bf16 [cap, (hi - lo) * head_dim]
    aio: ?*engram_aio.Aio = null, // a round's reads by Linux AIO on the tables' O_DIRECT descriptors (else the pool)
    pk: ?*engram_rows.Rows = null, // TF_DS_ENGRAM_PACKED: rows staged as packed records and decoded on the GPU
    paio: ?*engram_aio.Aio = null, // a prompt chunk's reads the same way, on a ring of its own (else the pool)
    // the rounds' Engram host time (ns, summed; a profile reads it): hashing, the table reads, the decode, the upload
    io: ?std.Io = null,
    t_hash: u64 = 0,
    t_read: u64 = 0,
    t_decode: u64 = 0,
    t_upload: u64 = 0,
    calls: u64 = 0,
    // a prompt chunk's rows read ahead on a thread (engramReadAhead): Engram layer j's rows of the chunk at
    // [ahead_start, ahead_start + ahead_n), ready when ahead_ready[j]
    ahead_w: [2][]u8 = .{ &.{}, &.{} },
    ahead_s: [2][]u8 = .{ &.{}, &.{} },
    ahead_rows: [2][]u16 = .{ &.{}, &.{} },
    ahead_flat: [2][]i64 = .{ &.{}, &.{} },
    ahead_ready: [2]bool = .{ false, false },
    ahead_start: usize = 0,
    ahead_n: usize = 0,
    ahead_err: ?anyerror = null,
    // a profile's prompt Engram split (ns summed, the stream synchronized at each cut): the rows ready on the host,
    // their upload, the projection, the quarters' exchange, the sum and gate
    p_rows: u64 = 0,
    p_upload: u64 = 0,
    p_mm: u64 = 0,
    p_x: u64 = 0,
    p_gate: u64 = 0,
    p_calls: u64 = 0,
    // the prompt's read-ahead worker (engramWorker; model.zig runs it through a fill): chunk i's rows of Engram layer j
    // in wk_rows[i % 2][j], chunk i ready once wk_done[j] > i, its slot free again once wk_used[j] > i (the rows
    // uploaded), so the worker runs up to a chunk ahead of the layers: a chunk's table reads overlap the chunk before
    wk_rows: [2][2][]u16 = .{ .{ &.{}, &.{} }, .{ &.{}, &.{} } },
    wk_hashes: []i64 = &.{},
    wk_flat: []i64 = &.{},
    wk_w: []u8 = &.{},
    wk_s: []u8 = &.{},
    wk_done: [2]std.atomic.Value(usize) = .{ .init(0), .init(0) },
    // WP0 4.5 (TF_DS_LOG_TIMING): wall-clock ns when wk_done[j] last advanced, read back by engramWorkerRows to log
    // the worker's own finish time next to the wait it measures there -- diagnostic only, no effect on any result.
    wk_done_ns: [2]std.atomic.Value(i128) = .{ .init(0), .init(0) },
    wk_used: [2]std.atomic.Value(usize) = .{ .init(0), .init(0) },
    wk_stop: std.atomic.Value(bool) = .init(false),
    wk_failed: std.atomic.Value(bool) = .init(false),
    wk_err: ?anyerror = null,
    wk_thread: ?std.Thread = null,
    wk_seq: []const i32 = &.{}, // the prompt it reads (its whole ids)
    wk_start: usize = 0, // its first chunk's start
    use_rows: ?[]u16 = null, // engramApply's rows when the worker read them
    use_stage: ?*engram_rows.Stage = null, // its records instead with packed shards (engramWorkerStage)

    pub fn init(gpa: std.mem.Allocator, c: *const Config, hasher: engram.Hasher, tables: *const engram_io.Tables, pool: *engram_io.Pool, rank: usize, world: usize, cap: usize) !EngramHost {
        const cols = hasher.cols();
        const lo = rank * cols / world;
        const hi = (rank + 1) * cols / world;
        const k = hi - lo;
        const t = tables.layers.get(@intCast(c.engram_layers.slice()[0])) orelse return error.MissingEngramTable;
        var wr: [2][2][]u16 = .{ .{ &.{}, &.{} }, .{ &.{}, &.{} } };
        var max_w: usize = 0;
        var max_s: usize = 0;
        for (0..@min(2, c.engram_layers.slice().len)) |j| {
            const tj = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            for (0..2) |slot| wr[slot][j] = try gpa.alloc(u16, cap * k * tj.dim());
            max_w = @max(max_w, tj.row_w);
            max_s = @max(max_s, tj.row_s);
        }
        var aw: [2][]u8 = .{ &.{}, &.{} };
        var as: [2][]u8 = .{ &.{}, &.{} };
        var ar: [2][]u16 = .{ &.{}, &.{} };
        var af: [2][]i64 = .{ &.{}, &.{} };
        for (0..@min(2, c.engram_layers.slice().len)) |j| {
            const tj = tables.layers.get(@intCast(c.engram_layers.slice()[j])) orelse return error.MissingEngramTable;
            aw[j] = try gpa.alloc(u8, cap * k * tj.row_w);
            as[j] = try gpa.alloc(u8, cap * k * tj.row_s);
            ar[j] = try gpa.alloc(u16, cap * k * tj.dim());
            af[j] = try gpa.alloc(i64, cap * k);
        }
        return .{
            .wk_rows = wr,
            .wk_hashes = try gpa.alloc(i64, cap * hasher.layers * cols),
            .wk_flat = try gpa.alloc(i64, cap * k),
            .wk_w = try gpa.alloc(u8, cap * k * max_w),
            .wk_s = try gpa.alloc(u8, cap * k * max_s),
            .ahead_w = aw,
            .ahead_s = as,
            .ahead_rows = ar,
            .ahead_flat = af,
            .hasher = hasher,
            .tables = tables,
            .pool = pool,
            .lo = lo,
            .hi = hi,
            .hashes = try gpa.alloc(i64, cap * hasher.layers * cols),
            .flat = try gpa.alloc(i64, cap * k),
            .w = try gpa.alloc(u8, cap * k * t.row_w),
            .s = try gpa.alloc(u8, cap * k * t.row_s),
            .rows = try gpa.alloc(u16, cap * k * t.dim()),
        };
    }
};

/// A prompt chunk's Engram layer j rows ahead of its layer, on a thread of the caller's while the GPU runs the layers
/// before it: the n-grams hashed (j 0: every layer's at once), the rows read and decoded into ahead_rows[j], exactly
/// as engramApply would at the layer. Errors land in ahead_err.
pub fn engramReadAhead(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) void {
    engramAhead1(eh, c, seq, start, n, j) catch |err| {
        eh.ahead_err = err;
    };
}

/// The next chunk's rows of both Engram layers (fillChunk starts it once this chunk's last Engram layer is applied).
pub fn engramReadAheadAll(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize) void {
    for (0..@min(2, c.engram_layers.slice().len)) |j| {
        engramAhead1(eh, c, seq, start, n, j) catch |err| {
            eh.ahead_err = err;
            return;
        };
    }
}

fn engramAhead1(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize) !void {
    const li = c.engram_layers.slice()[j];
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    if (j == 0) eh.hasher.hashes(seq, start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
    if (eh.pk) |pr| {
        // packed shards: the records staged in this layer's read-ahead buffer (decoded on the GPU when applied)
        try pr.chunkRead(&pr.ahead[j], j, eh.hashes[0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers);
        eh.ahead_ready[j] = true;
        return;
    }
    for (0..n) |r| {
        for (0..k) |q| eh.ahead_flat[j][r * k + q] = eh.hashes[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const m = n * k;
    // by AIO on the prompt's ring when it is open (the main thread reads by it only after joining this one)
    if (eh.paio) |aio| try aioGather(aio, t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]) else try eh.pool.gather(t, eh.ahead_flat[j][0..m], eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s]);
    engram.decodeRows(eh.ahead_w[j][0 .. m * t.row_w], eh.ahead_s[j][0 .. m * t.row_s], eh.ahead_rows[j][0 .. m * t.dim()], m, t.row_w, t.row_s, 8);
    eh.ahead_ready[j] = true;
}

/// The fill's read-ahead worker (a thread of model.zig's fill): from the chunk at `start`, chunk after chunk of
/// `chunk` rows to the end of `seq` (the prompt's whole ids), both Engram layers' rows hashed, read and decoded as
/// engramApply would, into the chunk's slot once the layers freed it. Errors land in wk_err (wk_failed set).
pub fn engramWorker(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, chunk: usize) void {
    const nl = @min(2, c.engram_layers.slice().len);
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    var at = start;
    var ci: usize = 0;
    while (at < seq.len) : (ci += 1) {
        const n = @min(chunk, seq.len - at);
        for (0..nl) |j| {
            while (ci >= eh.wk_used[j].load(.acquire) + 2) {
                if (eh.wk_stop.load(.acquire)) return;
                std.Thread.yield() catch {};
            }
            if (eh.wk_stop.load(.acquire)) return;
            workerRows(eh, c, seq, at, n, j, cols, k, eh.wk_rows[ci % 2][j], ci % 2) catch |err| {
                eh.wk_err = err;
                eh.wk_failed.store(true, .release);
                return;
            };
            eh.wk_done_ns[j].store(nowNs(), .release);
            eh.wk_done[j].store(ci + 1, .release);
        }
        at += n;
    }
}

fn workerRows(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, n: usize, j: usize, cols: usize, k: usize, out: []u16, slot: usize) !void {
    const li = c.engram_layers.slice()[j];
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    if (j == 0) eh.hasher.hashes(seq, start, n, eh.wk_hashes[0 .. n * eh.hasher.layers * cols]);
    // packed shards: the chunk's records staged in the worker's slot, decoded on the GPU when applied
    if (eh.pk) |pr| return pr.chunkRead(&pr.wk[slot][j], j, eh.wk_hashes[0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers);
    for (0..n) |r| {
        for (0..k) |q| eh.wk_flat[r * k + q] = eh.wk_hashes[(r * eh.hasher.layers + j) * cols + eh.lo + q];
    }
    const m = n * k;
    if (eh.paio) |aio| try aioGather(aio, t, eh.wk_flat[0..m], eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s]) else try eh.pool.gather(t, eh.wk_flat[0..m], eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s]);
    engram.decodeRows(eh.wk_w[0 .. m * t.row_w], eh.wk_s[0 .. m * t.row_s], out[0 .. m * t.dim()], m, t.row_w, t.row_s, 8);
}

/// Start the worker for a fill of `seq` from chunk `start` (any worker before it stopped first).
pub fn engramWorkerStart(eh: *EngramHost, c: *const Config, seq: []const i32, start: usize, chunk: usize) !void {
    engramWorkerStop(eh);
    for (&eh.wk_done) |*v| v.store(0, .release);
    for (&eh.wk_used) |*v| v.store(0, .release);
    eh.wk_stop.store(false, .release);
    eh.wk_failed.store(false, .release);
    eh.wk_err = null;
    eh.wk_seq = seq;
    eh.wk_start = start;
    eh.wk_thread = try std.Thread.spawn(.{}, engramWorker, .{ eh, c, seq, start, chunk });
}

/// Stop and join the worker (a fill's end, or its prompt about to change).
pub fn engramWorkerStop(eh: *EngramHost) void {
    const th = eh.wk_thread orelse return;
    eh.wk_stop.store(true, .release);
    th.join();
    eh.wk_thread = null;
    eh.wk_seq = &.{};
}

/// The worker's rows of Engram layer j for the chunk at `start` (waiting for them), for engramApply.
pub fn engramWorkerRows(eh: *EngramHost, j: usize, start: usize, chunk: usize) ![]u16 {
    const ci = (start - eh.wk_start) / chunk;
    // WP0 4.5 (TF_DS_LOG_TIMING=1): the wait this loop spends blocked on the worker, chunk 0 against chunk 1, is the
    // "Engram worker wait" existing counters miss (it precedes engramApply). Diagnostic print only; no other effect.
    const log_timing = std.c.getenv("TF_DS_LOG_TIMING") != null;
    const t_entry: i128 = if (log_timing) nowNs() else 0;
    while (eh.wk_done[j].load(.acquire) <= ci) {
        if (eh.wk_failed.load(.acquire)) return eh.wk_err orelse error.EngramWorkerFailed;
        std.Thread.yield() catch {};
    }
    if (log_timing) {
        const t_now = nowNs();
        const wait_us = @divTrunc(t_now - t_entry, 1000);
        const done_ns = eh.wk_done_ns[j].load(.acquire);
        const done_before_us = if (done_ns != 0) @divTrunc(t_entry - done_ns, 1000) else 0;
        std.debug.print("[tf-timing] engram_worker_wait ci={d} j={d} wait_us={d} entry_was_after_worker_done_by_us={d}\n", .{ ci, j, wait_us, done_before_us });
    }
    return eh.wk_rows[ci % 2][j];
}

/// With packed shards: the worker's records of Engram layer j for the chunk at `start` (waiting for them), for
/// engramApply, as engramWorkerRows gives its rows.
pub fn engramWorkerStage(eh: *EngramHost, j: usize, start: usize, chunk: usize) !*engram_rows.Stage {
    const pr = eh.pk orelse return error.NoPackedRows;
    _ = try engramWorkerRows(eh, j, start, chunk);
    return &pr.wk[((start - eh.wk_start) / chunk) % 2][j];
}

/// Layer li's Engram on the chunk (engram_apply, no image span), as _forward_k runs it before the attention mixes: a
/// pending MoE post into the streams first (hc_post), then the rows of each position's n-grams (hashed on the host from
/// `seq`, the whole sequence's ids so far; read from the tables; decoded to bf16 as Engram._decode does), their
/// projection summed over the ranks in rank order, rounded to bf16, and the gate into the streams.
pub fn engramApply(e: *const Engine, ch: *Chunk, eh: *EngramHost, li: usize, seq: []const i32) !void {
    const lay = e.w.layers[li];
    const ew = lay.engram_wkv orelse return;
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        if (qhcRange(e, ch, li)) |r| {
            // WP4: the same row-sharded flush as attnMixes's pending branch (hcPost alone is _hc_mix_pf's "posted"
            // finisher without the RMSNorm finish; same G-buffer stride gotcha, same repack).
            const row_h = c.hc * c.hidden * 2;
            const row_p = c.hc * 4;
            const row_c = c.hc * c.hc * 4;
            try qhcRepackG(e, ch, ch.gm, r);
            try tri_basic.hcPost(e.t, ch.qhc_g, ch.h + r.start * row_h, ch.post + r.start * row_p, ch.comb + r.start * row_c, ch.h + r.start * row_h, e.world, r.n, c.hidden);
        } else {
            try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        }
        ch.pending = false;
    }
    const li32: u16 = @intCast(li);
    const l = std.mem.indexOfScalar(u16, c.engram_layers.slice(), li32) orelse return error.NotAnEngramLayer;
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    const cols = eh.hasher.cols();
    const k = eh.hi - eh.lo;
    const m = n * k;
    const prof = eh.io != null;
    var tq = if (prof) pstamp(e, eh) else 0;
    if (m * t.dim() != n * ew.k) return error.EngramShape;
    if (eh.pk) |pr| {
        // packed shards: the chunk's records (the fill's worker's, else read ahead when they are this chunk's, else read
        // here), decoded on the GPU
        const st = if (eh.use_stage) |us| us else if (l < 2 and eh.ahead_ready[l] and eh.ahead_start == ch.start and eh.ahead_n == n) blk: {
            eh.ahead_ready[l] = false;
            break :blk &pr.ahead[l];
        } else blk: {
            eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
            try pr.chunkRead(&pr.chunk, l, eh.hashes[0 .. n * eh.hasher.layers * cols], n, eh.hasher.layers);
            break :blk &pr.chunk;
        };
        if (prof) eh.p_rows += lap(e, eh, &tq);
        try pr.chunkApply(st, n, e.comm, e.s, ch.eb);
        if (prof) eh.p_upload += lap(e, eh, &tq);
    } else {
        // the rows read ahead (fillChunk's thread) when they are this chunk's, else read here
        var rows: []u16 = eh.rows;
        if (eh.use_rows) |ur| {
            rows = ur;
        } else if (l < 2 and eh.ahead_ready[l] and eh.ahead_start == ch.start and eh.ahead_n == n) {
            rows = eh.ahead_rows[l];
            eh.ahead_ready[l] = false;
        } else {
            eh.hasher.hashes(seq, ch.start, n, eh.hashes[0 .. n * eh.hasher.layers * cols]);
            for (0..n) |r| {
                for (0..k) |j| eh.flat[r * k + j] = eh.hashes[(r * eh.hasher.layers + l) * cols + eh.lo + j];
            }
            if (eh.paio) |aio| try aioGather(aio, t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]) else try eh.pool.gather(t, eh.flat[0..m], eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s]);
            engram.decodeRows(eh.w[0 .. m * t.row_w], eh.s[0 .. m * t.row_s], eh.rows[0 .. m * t.dim()], m, t.row_w, t.row_s, 8); // the rows on 8 threads
        }
        if (prof) eh.p_rows += lap(e, eh, &tq);
        try upload(e, ch.eb, rows.ptr, m * t.dim() * 2);
        if (prof) eh.p_upload += lap(e, eh, &tq);
    }
    try mm(e, ch, ew, ch.eb, ew.k, ch.ek, .fp32, ew.n);
    if (prof) eh.p_mm += lap(e, eh, &tq);
    // 2D (model.py esum): this node's columns of its rank's projection; the quarters give both ranks' whole ones
    const en = if (e.two) |two| two.engramWidth() else ew.n;
    // 2D chunks past the rings: the pair's sum then the pairs' bf16 halves (Two.engramSum), the same bits
    const psum = if (e.two) |two| two.rings == null or n >= 64 else false;
    if (psum) {
        try e.two.?.engramSum(e, ch.ek, ch.ekg, ch.kv, n);
        if (prof) eh.p_x += lap(e, eh, &tq);
    } else {
        if (e.two) |two| try two.quarters(e, ch.ek, ch.ekg, n, two.ew, 4) else try e.comm.allGather(ch.ek, ch.ekg, n * ew.n, .f32, e.s);
        if (prof) eh.p_x += lap(e, eh, &tq);
        if (e.world != 2) return notPorted(@src()); // Comm.sum of more ranks: acc += g[r]
        try e.ops.add2Bf16(e.s, ch.ekg, ch.ekg + n * en * 4, ch.kv, n * en);
    }
    try tri_basic.engramGate(e.t, ch.h, ch.kv, lay.engram_qk, ch.h_alt, c.eps, n, c.hidden);
    swapStreams(ch, ch.h_alt);
    // image spans take no Engram contribution: their rows keep the streams they came with (engram_mask)
    const rowb = c.hc * c.hidden * 2;
    for (ch.spans) |sp| {
        const o = overlap(sp, ch.start, n) orelse continue;
        try e.ops.copyRows(e.s, ch.h_alt + o[0] * rowb, rowb, ch.h + o[0] * rowb, rowb, rowb, o[1] - o[0]);
    }
    if (prof) {
        eh.p_gate += lap(e, eh, &tq);
        eh.p_calls += 1;
    }
}

/// A profile's clock after the stream drains (prompt Engram split).
fn pstamp(e: *const Engine, eh: *const EngramHost) u64 {
    e.s.synchronize() catch {};
    return @intCast(std.Io.Timestamp.now(eh.io.?, .awake).nanoseconds);
}

fn lap(e: *const Engine, eh: *const EngramHost, t: *u64) u64 {
    const now = pstamp(e, eh);
    defer t.* = now;
    return now - t.*;
}

/// A chunk's rows `ids` of table `t` by AIO (the pool's bytes, at ~0.07 s for 24,576 rows where the pool's preads take
/// ~0.5 s on one GB10 node): batches of a quarter of the ring's slots (two reads a row), all submitted (a submit reaps while the
/// ring is full), then waited for in order.
fn aioGather(aio: *engram_aio.Aio, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
    const per = @max(aio.slots / 4, 1);
    var batch: [64]u64 = undefined;
    var nb: usize = 0;
    var at: usize = 0;
    errdefer for (batch[0..nb]) |id| aio.wait(id) catch {};
    while (at < ids.len) : (at += per) {
        const n = @min(per, ids.len - at);
        if (nb == batch.len) return error.TooManyRows;
        batch[nb] = try aio.start(t, ids[at..][0..n], out_w[at * t.row_w ..][0 .. n * t.row_w], out_s[at * t.row_s ..][0 .. n * t.row_s]);
        nb += 1;
    }
    for (batch[0..nb], 0..) |id, i| {
        aio.wait(id) catch |err| {
            for (batch[i + 1 .. nb]) |rest| aio.wait(rest) catch {};
            nb = 0;
            return err;
        };
    }
    nb = 0;
}

/// The decoder's bounded replay at its first layer (_forward_k's `replay`, CED's prefill): the layer's attention mixes
/// (the pending MoE post fused in) and its kv-source update over every row of the chunk, then the chunk cut to its rows
/// at positions >= replay, the streams and their pre copied down (h[first:], pre[first:]). The decoder layers then run
/// on those rows with no window key before replay and this layer's update done. False: an encoder-only chunk (no
/// row at or past replay), which ends here.
pub fn replayCut(e: *const Engine, ch: *Chunk, cs: *const Caches, sh: *Shared, li: usize, replay: usize, host_pos: []i64) !bool {
    try attnMixes(e, ch, li);
    if (e.w.layers[li].comp_wkv == null) return error.NoKvSource;
    try kvSourceUpdate(e, ch, cs, sh, li);
    const first = @max(ch.start, replay) - ch.start;
    if (first >= ch.n) return false;
    if (first == 0) return true;
    const c = e.c;
    const n = ch.n - first;
    const row_h = c.hc * c.hidden * 2;
    try e.ops.copyRows(e.s, ch.h + first * row_h, row_h, ch.h_alt, row_h, row_h, n);
    std.mem.swap(u64, &ch.h, &ch.h_alt);
    const row_p = c.hc * 4;
    try e.ops.copyRows(e.s, ch.pre + first * row_p, row_p, ch.pre_f, row_p, row_p, n);
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
    ch.start += first;
    ch.n = n;
    // pos[first:]: a view, first * 8 bytes in (its alignment is the one the served kernels were specialized for). An odd
    // `first` leaves it off a 16-byte boundary, and the kit has no variant for some launches with such a view (a short
    // chunk's split attention: _sparse_attn_part with SPLITS > 1 and no row bases); the same positions copied, in the
    // stream's order, to a 16-byte boundary then: the kernels read the same values from an address they were captured for.
    _ = host_pos;
    if ((first * 8) % 16 != 0) {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(ch.pos1, ch.pos + first * 8, n * 8, e.s.handle), "cuMemcpyDtoDAsync");
        ch.pos = ch.pos1;
    } else ch.pos += first * 8;
    return true;
}

/// DSpark tap j (_forward_k at a tap layer): a pending MoE gather posted into the streams first (hc_post, not fused),
/// then h.to(fp32).mean(1).to(bf16) into tap j's rows.
pub fn tap(e: *const Engine, ch: *Chunk, j: usize, li: usize) !void {
    const c = e.c;
    const r = qhcRange(e, ch, li);
    if (ch.pending) {
        if (r) |rr| {
            const row_h = c.hc * c.hidden * 2;
            const row_p = c.hc * 4;
            const row_c = c.hc * c.hc * 4;
            try qhcRepackG(e, ch, ch.gm, rr);
            try tri_basic.hcPost(e.t, ch.qhc_g, ch.h + rr.start * row_h, ch.post + rr.start * row_p, ch.comb + rr.start * row_c, ch.h + rr.start * row_h, e.world, rr.n, c.hidden);
        } else {
            try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, ch.n, c.hidden);
        }
        ch.pending = false;
    }
    // WP4: hcMean4 is a per-row mean (no cross-row reduction), so it is safe to run over the full ch.n even with
    // this node's don't-care garbage in the rows it does not own (that garbage produces garbage only in those same
    // rows' tap output, which the gather below overwrites with every rank's own correct rows next) -- no row-shard
    // needed for the mean itself, only for what already needed it above (hcPost) and for making ch.taps globally
    // correct afterward, since DSpark drafting reads every row of it on every node.
    try e.exact.hcMean4(e.s, ch.h, ch.taps + j * ch.cap * c.hidden * 2, c.hidden, ch.n, c.hidden);
    if (r) |rr| try qhcGatherRows(e, ch.taps + j * ch.cap * c.hidden * 2, c.hidden, rr);
}

/// Tap j's rows (bf16 [n, D]).
pub fn tapRows(e: *const Engine, ch: *const Chunk, j: usize) u64 {
    return ch.taps + j * ch.cap * e.c.hidden * 2;
}

/// The head on the chunk's last row (_forward_k without all_logits): the pending MoE post, collapse_norm of the row with
/// its pre, the head's columns (one row through the head's own group, fp32) and their gather: head_g holds the
/// prompt's logits, the ranks' columns in rank order.
pub fn head(e: *const Engine, ch: *Chunk) !void {
    const c = e.c;
    const n = ch.n;
    if (ch.pending) {
        try tri_basic.hcPost(e.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, e.world, n, c.hidden);
        ch.pending = false;
    }
    const last_h = ch.h + (n - 1) * c.hc * c.hidden * 2;
    const last_pre = ch.pre + (n - 1) * c.hc * 4;
    try tri_basic.collapseNorm(e.t, last_h, last_pre, e.w.norm, ch.head_x, c.eps, 1, c.hidden);
    const hl = e.w.head;
    try mmRows(e, ch, 1, hl, ch.head_x, c.hidden, ch.head_l, .fp32, hl.n);
    dump(e, 0, "head_x", ch.head_x, c.hidden * 2);
    dump(e, 0, "head_l", ch.head_l, hl.n * 4);
    if (e.two) |t| return t.quarters(e, ch.head_l, ch.head_g, 1, t.hw, 4); // 2D: the vocabulary quarters, TP2's layout
    try e.comm.allGather(ch.head_l, ch.head_g, hl.n, .f32, e.s);
}

/// A layer's end: the MoE's gather is pending (the next layer's attention mixes post it) and the FFN's pre_out is the
/// next layer's pre (_forward_k's `pre, pre_f = pre_f, pre`).
pub fn endLayer(ch: *Chunk) void {
    ch.pending = true;
    std.mem.swap(u64, &ch.pre, &ch.pre_f);
}

test "hd ** -0.5 as the served build passes it" {
    try std.testing.expectEqual(@as(u32, 0x3d3504f3), @as(u32, @bitCast(scale(512))));
}

/// error.NotPortedYet with its site on stderr (a request that reaches a path the port lacks names it in the log).
fn notPorted(src: std.builtin.SourceLocation) error{NotPortedYet} {
    std.debug.print("{{\"not_ported\": \"{s}:{d}\"}}\n", .{ src.file, src.line });
    return error.NotPortedYet;
}
