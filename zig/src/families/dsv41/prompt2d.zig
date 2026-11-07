//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. Prompt chunks exchange over NCCL across the four (the RDMA rings drop prompt-size bursts on the
//! switch: slots P6, P7b), each as one all-gather of every node's part then copies into place: the bytes Comm2D's
//! NCCL paths (gather4, _p2p) put there. With RDMA rings (Two.rings, ring2d.zig) the exchanges the lane sends over
//! them take them, written straight into place: quarters of fewer than 64 rows (decode windows; TF_DS_2D_PROMPT_NCCL
//! keeps prompt chunks on NCCL) and every cat_rank its column pair's ring takes; the same bytes either way.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const mx = @import("mx.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const rdma_mod = @import("rdma.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Ports = @import("link.zig").Ports;
const Comm = @import("comm.zig").Comm;

const nccl = cuda.nccl;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    return switch (world) {
        1, 2 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// How long a follower retries its connect to rank 0: TF_DS_CONNECT_S seconds, 300 by default. Rank 0's ports listen
/// from its start (Fan.listenAt, before its weights load), so this covers only rank 0's process starting late.
pub fn connectSeconds() u32 {
    const v = std.c.getenv("TF_DS_CONNECT_S") orelse return 300;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 300;
}

/// The links of a run and its NCCL world: rank 0 listens on PORT + g - 1 for follower g, all the followers' ports at
/// once, and takes them in order (a follower that is ready early waits in its port's backlog), so two ranks link
/// exactly as before (rank 1 on PORT). NCCL's unique id goes from rank 0 down every link. (The port's link to N
/// followers replaces this.)
pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },
    ports: ?Ports = null, // rank 0's, from listenAt until join has taken every follower

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        return openAt(io, ip, null, port, rank, world);
    }

    /// open, rank 0 listening for follower g on `listen[g - 1]` when given (each follower's own link address of this
    /// node: on a ring without a switch a follower reaches the head at the address of the cable it comes in on), else
    /// on `ip` for all of them.
    pub fn openAt(io: std.Io, ip: [4]u8, listen: ?[3][4]u8, port: u16, rank: u32, world: u32) !Fan {
        var f = try listenAt(ip, listen, port, rank, world);
        errdefer f.close();
        try f.join(io, ip, port, rank, world, connectSeconds());
        return f;
    }

    /// openAt's first half: rank 0's ports for all its followers, listening from here on (Model.open calls it before
    /// the weights load, so a follower that loads faster than rank 0 is not left retrying); nothing on a follower.
    pub fn listenAt(ip: [4]u8, listen: ?[3][4]u8, port: u16, rank: u32, world: u32) !Fan {
        if (world > 4) return error.UnsupportedWorld;
        var f: Fan = .{};
        if (rank != 0 or world < 2) return f;
        var ips: [3][4]u8 = undefined;
        for (ips[0 .. world - 1], 0..) |*a, i| a.* = if (listen) |l| l[i] else ip;
        f.ports = try Ports.open(ips[0 .. world - 1], port);
        return f;
    }

    /// openAt's second half: rank 0 takes its followers in order; a follower connects to its port on rank 0, retrying
    /// for `seconds` while rank 0 starts.
    pub fn join(f: *Fan, io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32, seconds: u32) !void {
        if (world < 2) return;
        if (rank == 0) {
            for (1..world) |g| f.links[g - 1] = try f.ports.?.accept(g);
            f.ports = null;
            return;
        }
        const at = port + @as(u16, @intCast(rank - 1));
        f.links[0] = Link.connect(io, ip, at, seconds) catch |err| {
            if (err == error.ConnectFailed) std.log.err("rank {d}: rank 0 did not answer at {d}.{d}.{d}.{d}:{d} in {d} s (TF_DS_CONNECT_S)", .{ rank, ip[0], ip[1], ip[2], ip[3], at, seconds });
            return err;
        };
    }

    pub fn close(f: *Fan) void {
        if (f.ports) |*p| p.close();
        f.ports = null;
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// Comm.init over every link: rank 0 makes the id and sends it to each follower. Needs the CUDA context current.
    pub fn comm(f: *const Fan, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            if (rank == 0) {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                for (f.links) |l| if (l) |x| try x.send(std.mem.asBytes(&id));
            } else {
                var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
                const got = try f.links[0].?.recv(&buf);
                if (got.len != buf.len) return error.BadUniqueId;
                id = std.mem.bytesToValue(nccl.UniqueId, got);
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }
};

/// TF_DS_2D_ORDER: the 2D node (role) each NCCL rank serves, "g0,g1,g2,g3" (rank 0 must serve node 0: it serves the
/// API); unset, rank g serves node g. On four GB10s cabled as a ring 0-1-2-3-0 without a switch, "0,1,3,2" makes every
/// node's row partner (g ^ 1) and column partner (g ^ 2) its cabled neighbours, and NCCL's ring all-gathers run over
/// the cables in rank order.
pub fn order() ![4]u32 {
    const v = std.c.getenv("TF_DS_2D_ORDER") orelse return .{ 0, 1, 2, 3 };
    return parseOrder(std.mem.span(v));
}

pub fn parseOrder(text: []const u8) ![4]u32 {
    var out: [4]u32 = undefined;
    var seen: [4]bool = @splat(false);
    var it = std.mem.splitScalar(u8, text, ',');
    for (&out) |*o| {
        const x = std.mem.trim(u8, it.next() orelse return error.BadOrder, " ");
        o.* = std.fmt.parseInt(u32, x, 10) catch return error.BadOrder;
        if (o.* > 3 or seen[o.*]) return error.BadOrder;
        seen[o.*] = true;
    }
    if (it.next() != null or out[0] != 0) return error.BadOrder;
    return out;
}

/// The NCCL rank of each node g under `ord` (the inverse permutation).
pub fn ncclOf(ord: [4]u32) [4]u32 {
    var inv: [4]u32 = undefined;
    for (ord, 0..) |g, i| inv[g] = @intCast(i);
    return inv;
}

test "TF_DS_2D_ORDER: a ring's order and its inverse" {
    const o = try parseOrder("0,1,3,2");
    try std.testing.expectEqual([4]u32{ 0, 1, 3, 2 }, o);
    try std.testing.expectEqual([4]u32{ 0, 1, 3, 2 }, ncclOf(o));
    try std.testing.expectEqual([4]u32{ 0, 2, 3, 1 }, ncclOf(try parseOrder("0,3,1,2")));
    try std.testing.expectError(error.BadOrder, parseOrder("1,0,2,3"));
    try std.testing.expectError(error.BadOrder, parseOrder("0,1,1,2"));
    try std.testing.expectError(error.BadOrder, parseOrder("0,1,2"));
}

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    parts: u64, // [4, rows, widest] where the all-gathers land
    pad: u64, // this node's part at the wider pair's width (when the pairs' widths differ)
    part_bytes: usize, // the largest part a node sends
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const ring2d.Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them
    // TF_DS_2D_GU=parity: the rest block's width (0: "first"), the rest launches' scratch for a decode window and for a
    // prompt chunk, and the column partner's pack (its main rows, then its rest rows) as the exchange delivers it
    ir: usize = 0,
    rest_rows: usize = 0, // the prompt chunks' rows the rest scratch holds (init's cap)
    rest_dec: ?exl3_experts2d.RestScratch = null,
    rest_chunk: ?exl3_experts2d.RestScratch = null,
    peer: u64 = 0,
    // the NCCL rank of each 2D node g (TF_DS_2D_ORDER, model.zig): the all-gathers' slots and the partners' ranks. The
    // identity on a switch; on a ring of four cabled in NCCL rank order, roles 0, 1, 3, 2 around it put every row and
    // column partner on a cable of its own
    nccl: [4]u32 = .{ 0, 1, 2, 3 },
    // TF_DS_PAIR_SUM=1, agreed by every rank at startup (agreePairSum): a prompt chunk's gathers as pair sums
    pair_sum: bool = false,
    // WP4 (docs/prefill-plan.md): quarter-HC, agreed at startup (agreeQuarterHc) like pair_sum.
    qhc: bool = false,
    // WP5 (docs/prefill-plan.md, owner-row transport): agreed at startup (agreeOwnerGather), needs qhc and equal
    // pair widths. Shrinks pairSum's column exchange to each rank's own owned quarter for a row-sharded layer's
    // gather, instead of every row.
    owner_gather: bool = false,
    // WP6(b) (docs/prefill-plan.md, indexer row-sharding): agreed at startup (agreeIdxShard), independent of qhc
    // and owner_gather -- the indexer's own scoring kernels, not the HC-mix/gather path those two touch.
    idx_shard: bool = false,

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.pair_sum = false; // agreePairSum sets it on a model start; other callers keep the four-quarter gather
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        const biggest = @max(@max(@max(cap * @max(t.ow[0], t.ow[1]) * 4, cap * @max(t.ew[0], t.ew[1]) * 4), @max(cap * @max(t.uw[0], t.uw[1]) * 2, cap * slots * @max(t.gu[0], t.gu[1]) * 2)), @max(t.hw[0], t.hw[1]) * 4);
        t.part_bytes = biggest;
        t.parts = try a.take(4 * biggest);
        t.pad = try a.take(biggest);
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        // parity: the rest block after both pairs' main blocks in the assembled intermediate
        const ex0 = l0.experts;
        t.ir = if (ex0.rest) |r| r.width else 0;
        t.rest_rows = cap;
        t.rest_dec = null;
        t.rest_chunk = null;
        t.peer = 0;
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1] + t.ir) * 2);
        if (t.ir != 0) {
            if (t.gu[0] != t.gu[1]) return error.UnequalParityParts;
            // (Format.native computes the rest in the main launch: no rest scratch)
            if (ex0.native == null) {
                t.rest_dec = try restScratch(a, ex0, exl3_experts.exact_rows, slots, false);
                t.rest_chunk = try restScratch(a, ex0, cap, slots, true);
            }
            t.peer = try a.take(@max(cap, round_rows) * slots * (t.gu[t.p] + t.ir) * 2);
        }
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        t.nccl = .{ 0, 1, 2, 3 };
        return t;
    }

    fn restScratch(a: *prompt.Arena, ex: weights.Experts, rows: usize, slots: usize, chunk: bool) !exl3_experts2d.RestScratch {
        const sz = try exl3_experts2d.RestScratch.sizes(rows, slots, ex, chunk);
        var rs: exl3_experts2d.RestScratch = undefined;
        inline for (.{ "ids", "count", "counts", "members", "cnt_gu", "z", "work_gu", "work_d", "rpick" }, 0..) |f, j| @field(rs, f) = try a.take(sz[j]);
        return rs;
    }

    /// The rest scratch's first use: members -1, the counters and partials zero (as the main scratch starts).
    pub fn clearRest(t: *const Two, e: *const prompt.Engine) !void {
        for ([_]?exl3_experts2d.RestScratch{ t.rest_dec, t.rest_chunk }, [_]bool{ false, true }) |o, chunk| {
            const rs = o orelse continue;
            const ex = e.w.layers[0].experts;
            const sz = try exl3_experts2d.RestScratch.sizes(if (chunk) t.rest_rows else exl3_experts.exact_rows, e.slots(), ex, chunk);
            try prompt.fill32(e, rs.members, 0xffffffff, sz[3] / 4);
            for ([_]u64{ rs.ids, rs.count, rs.counts, rs.cnt_gu, rs.z }, [_]usize{ sz[0], sz[1], sz[2], sz[4], sz[5] }) |ptr, n| try prompt.fill(e, ptr, 0, n);
        }
    }

    /// The column partner's pack (a parity split's main rows, then its rest rows: `bytes` each way) into t.peer: over
    /// the column ring when the caller opened the rings and it fits, else NCCL send / recv with the partner.
    pub fn packExchange(t: *const Two, e: *const prompt.Engine, mine: u64, rows: usize, width: usize) !void {
        if (t.rings) |rs| {
            var dsts: [2]rdma_mod.Ring.Dst = .{ .{}, .{} };
            dsts[1 - t.p] = .{ .ptr = t.peer, .row4 = @intCast(width * 2 / 16), .stride4 = width * 2 / 16 };
            if (rs.col.gatherInto(e.s, mine, @intCast(rows), @intCast(width * 2 / 16), &dsts)) |_| return else |err| switch (err) {
                error.TooLarge => {},
                else => return err,
            }
        }
        const g: u32 = @intCast(t.r + 2 * t.p);
        try t.exchange(e, mine, rows * width, t.peer, rows * width, .f16, 2, t.nccl[g ^ 2]);
    }

    /// moe_2d's parity experts after the main gate / up and the rest's: the packs exchanged with the column partner,
    /// the half's intermediate assembled into t.xd_full (pair 0's main, pair 1's main, each slot's rest from its
    /// owner), for down.
    pub fn assembleParity(t: *const Two, e: *const prompt.Engine, ex: weights.Experts, xd: u64, pick: u64, p: usize) !void {
        const rest = ex.rest orelse return error.NoRest;
        const ig = t.gu[t.p];
        try t.packExchange(e, xd, p, ig + t.ir);
        const part0 = if (t.p == 0) xd else t.peer;
        const part1 = if (t.p == 1) xd else t.peer;
        try e.ops.assembleRest(e.s, part0, part1, pick, rest.owner, t.xd_full, p, ig, t.ir);
    }

    /// `count` elements (`esize` bytes each) to NCCL rank `peer` and `rcount` from it: over the two-rank link to that ring
    /// neighbour when the communicator has links (a ring without a switch: a send / recv on the world communicator may
    /// take a port with no cable to the peer), as the link's all-gather (both parts at the larger count, through
    /// `parts`, the partner's copied out); else NCCL send / recv.
    pub fn exchange(t: *const Two, e: *const prompt.Engine, send: u64, count: usize, recv: u64, rcount: usize, dtype: cuda.nccl.DataType, esize: usize, peer: u32) !void {
        const l = e.comm.linkTo(peer) orelse return e.comm.exchange(send, count, recv, rcount, dtype, peer, e.s);
        const m = @max(count, rcount);
        if (2 * m * esize > 4 * t.part_bytes or m * esize > t.part_bytes) return error.PartTooBig;
        var src = send;
        if (count < m) {
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(t.pad, send, count * esize, e.s.handle), "cuMemcpyDtoDAsync");
            src = t.pad;
        }
        try e.comm.linkGather(l.comm, src, t.parts, m, dtype, e.s);
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(recv, t.parts + (1 - l.me) * m * esize, rcount * esize, e.s.handle), "cuMemcpyDtoDAsync");
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        try e.comm.allGather(send, t.parts, n * wmax * esize, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    /// quarters (Comm2D.quarters): `src` [n, w[p]] (this node's columns of its TP2 rank's partial) -> `dst`
    /// [2, n, w0 + w1]: rank r's partial, pair 0's columns then pair 1's, for every r.
    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (n < 64 and try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return;
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..4) |g| {
            const rr = g % 2;
            const pg = g / 2;
            try e.ops.copyRows(e.s, t.parts + t.nccl[g] * n * wmax * esize, wmax * esize, dst + rr * n * row + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// TF_DS_PAIR_SUM, resolved once for the whole world: every rank sends its setting, its node order (TF_DS_2D_ORDER,
    /// as `nccl`), its gather widths (`ow`), whether both its partners (the TP2 partner, node ^ 1, and the other
    /// pair's same TP2 rank, node ^ 2) are ring neighbours with a link, and whether its RDMA rings are enabled. Every
    /// rank then checks every other rank's node order and widths against its own: a rank alone on a different
    /// permutation (WP1 hardening; Codex's counterexample: three ranks on `0,1,3,2` and one on `0,2,3,1` would each
    /// pass a same-setting-only check, since every chosen partner is still a ring neighbour on paper, yet the ranks
    /// would address different partners under the same role numbers) now fails this exchange outright, with or
    /// without TF_DS_PAIR_SUM: these hazards (`openLinks` and the RDMA rings also depend on each rank's own
    /// environment) predate pair-sum and affect the plain four-quarter gather too. Ranks that disagree on the
    /// *setting* still refuse to start, as before (a rank alone on the other path would leave the rest in collectives
    /// it never joins); with the setting on everywhere, pair_sum engages only when every rank has both links, so all
    /// ranks take the same path for every chunk. Before any prompt; the world all-gather is its barrier.
    pub fn agreePairSum(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, rdma_on: bool) !void {
        const want = if (std.c.getenv("TF_DS_PAIR_SUM")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const me: u32 = @intCast(t.r + 2 * t.p);
        const links = comm.linkTo(t.nccl[me ^ 1]) != null and comm.linkTo(t.nccl[me ^ 2]) != null;
        const w = comm.world;
        const Cfg = extern struct { nccl: [4]u32, ow: [2]u32, flags: u32, pad: u32 = 0 };
        const mine: Cfg = .{
            .nccl = t.nccl,
            .ow = .{ @intCast(t.ow[0]), @intCast(t.ow[1]) },
            .flags = @as(u32, @intFromBool(want)) | (@as(u32, @intFromBool(links)) << 1) | (@as(u32, @intFromBool(rdma_on)) << 2),
        };
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * @sizeOf(Cfg));
        defer dev.free();
        try dev.upload(0, std.mem.asBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + @sizeOf(Cfg), @sizeOf(Cfg) / 4, .u32, stream);
        try stream.synchronize();
        var all: [8]Cfg = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(@sizeOf(Cfg), std.mem.sliceAsBytes(all[0..w]));
        var on: usize = 0;
        var linked: usize = 0;
        for (all[0..w], 0..) |x, i| {
            if (!std.mem.eql(u32, &x.nccl, &mine.nccl)) {
                std.log.err("TF_DS_PAIR_SUM: rank {d}'s node order {any} differs from this rank's {any} (TF_DS_2D_ORDER mismatch): refusing to start", .{ i, x.nccl, mine.nccl });
                return error.TopologyMismatch;
            }
            if (x.ow[0] != mine.ow[0] or x.ow[1] != mine.ow[1]) {
                std.log.err("TF_DS_PAIR_SUM: rank {d}'s gather widths {any} differ from this rank's {any}: refusing to start", .{ i, x.ow, mine.ow });
                return error.TopologyMismatch;
            }
            on += x.flags & 1;
            linked += (x.flags >> 1) & 1;
        }
        if (on != 0 and on != w) {
            std.log.err("TF_DS_PAIR_SUM: set on {d} of {d} ranks (this one: {s}); start every rank with the same value", .{ on, w, if (want) "1" else "unset" });
            return error.PairSumMismatch;
        }
        t.pair_sum = on == w and linked == w;
        if (on == w and !t.pair_sum) std.log.info("TF_DS_PAIR_SUM=1: {d} of {d} ranks have both partners as ring neighbours; every rank keeps the four-quarter gather", .{ linked, w });
        if (t.pair_sum) std.log.info("TF_DS_PAIR_SUM=1: prompt gathers of 64+ rows as pair sums on all {d} ranks", .{w});
    }

    /// TF_DS_PAIR_SUM=1 (pair_sum, agreed at startup): a prompt chunk's quarters as the readers use them. Every
    /// column's two TP2 partials are summed with NCCL over the link to this rank's TP2 partner (both ranks get the same
    /// result), then the sums swapped with the other pair over that link; `dst` [2, n, w0 + w1] holds the sums in its
    /// first plane and zeros in its second, and a reader adds G[0] + G[1]. Two link transfers of one part each instead
    /// of the four-rank all-gather's three. Called only when pair_sum is set, so both links exist on every rank.
    pub fn pairSum(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize) !void {
        const me: u32 = @intCast(t.r + 2 * t.p);
        const tp_link = e.comm.linkTo(t.nccl[me ^ 1]) orelse return error.PairSumNoLink;
        const pair_link = e.comm.linkTo(t.nccl[me ^ 2]) orelse return error.PairSumNoLink;
        const wmax = @max(w[0], w[1]);
        if (n * wmax * 4 > t.part_bytes) return error.PartTooBig;
        const mine = n * w[t.p];
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(t.pad, src, mine * 4, e.s.handle), "cuMemcpyDtoDAsync");
        try e.comm.linkSum(tp_link.comm, t.pad, mine, e.s);
        try e.comm.linkGather(pair_link.comm, t.pad, t.parts, n * wmax, .f32, e.s);
        const row_bytes = (w[0] + w[1]) * 4;
        for (0..2) |pg| {
            const slot: usize = if (pg == t.p) pair_link.me else 1 - pair_link.me;
            try e.ops.copyRows(e.s, t.parts + slot * n * wmax * 4, w[pg] * 4, dst + pg * w[0] * 4, row_bytes, w[pg] * 4, n);
        }
        // WP1 (Hector's reviewer, item 2): plane 1 is the identity element for the reader's add (dst[0] + dst[1]).
        // Positive zero is the identity for every case except x = -0, where round-to-nearest gives x + (+0) = +0 but
        // x + (-0) = x = -0: TF_DS_PAIR_SUM_NEGZERO=1 fills plane 1 with negative zero instead, so the add is the
        // identity for every non-NaN x, signed zeros included (it does not address NaN or flush-to-zero; the
        // tensor-level comparison, not this fill, is what decides exactness -- see WP1's exit gate).
        if (std.c.getenv("TF_DS_PAIR_SUM_NEGZERO") != null)
            try prompt.fill32(e, dst + n * row_bytes, 0x80000000, n * (w[0] + w[1]))
        else
            try prompt.fill(e, dst + n * row_bytes, 0, n * row_bytes);
    }

    /// WP5 (docs/prefill-plan.md, owner-row transport): pairSum, but the column link's exchange moves only the
    /// `owned` rows this node actually keeps (a row-sharded quarter-HC layer reads nothing else -- qhcRepackG
    /// copies exactly this same range back out of dst). linkSum (the TP partner's all-reduce) stays over every
    /// row unchanged: every row's sum is needed by SOME rank's owned range, not just this node's own, and it is
    /// already the cheaper of pairSum's two steps. `owned.start`/`owned.n` are qhcRange's own row range (q = cap
    /// / world, uniform across ranks -- the NCCL rank IS the owner index), so the column partner's owned range is
    /// computable here with no extra round trip: `peer_nccl * owned.n`.
    ///
    /// The exchange is NOT symmetric in content: this node sends its own pg's slice at the PARTNER's owned rows
    /// (what the partner needs back, not what this node needs), and receives the partner's pg slice at THIS
    /// node's owned rows, in the one ncclAllGather pairSum's own link already runs (same comm, same `t.parts`
    /// scratch, never t.pad -- t.pad holds this node's own tp-summed data at every row and is read from, not
    /// written to, by this step). This node's own pg at its own owned rows never needs the network at all: it is
    /// already sitting in t.pad from linkSum, exactly where qhcRepackG's reasoning says a rank's own correctly-
    /// computed rows need no collective to read back.
    pub fn pairSumOwned(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, owned: prompt.QRows) !void {
        const me: u32 = @intCast(t.r + 2 * t.p);
        const tp_link = e.comm.linkTo(t.nccl[me ^ 1]) orelse return error.PairSumNoLink;
        const peer_role: u32 = me ^ 2;
        const pair_link = e.comm.linkTo(t.nccl[peer_role]) orelse return error.PairSumNoLink;
        const wmax = @max(w[0], w[1]);
        if (n * wmax * 4 > t.part_bytes or owned.n * wmax * 4 > t.part_bytes) return error.PartTooBig;
        const mine = n * w[t.p];
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(t.pad, src, mine * 4, e.s.handle), "cuMemcpyDtoDAsync");
        try e.comm.linkSum(tp_link.comm, t.pad, mine, e.s);
        const peer_nccl = t.nccl[peer_role];
        const peer_start = peer_nccl * owned.n;
        try e.comm.linkGather(pair_link.comm, t.pad + peer_start * w[t.p] * 4, t.parts, owned.n * wmax, .f32, e.s);
        const row_bytes = (w[0] + w[1]) * 4;
        try e.ops.copyRows(e.s, t.pad + owned.start * w[t.p] * 4, w[t.p] * 4, dst + owned.start * row_bytes + t.p * w[0] * 4, row_bytes, w[t.p] * 4, owned.n);
        const op = 1 - t.p;
        const slot: usize = 1 - pair_link.me;
        try e.ops.copyRows(e.s, t.parts + slot * owned.n * wmax * 4, w[op] * 4, dst + owned.start * row_bytes + op * w[0] * 4, row_bytes, w[op] * 4, owned.n);
        if (std.c.getenv("TF_DS_PAIR_SUM_NEGZERO") != null)
            try prompt.fill32(e, dst + n * row_bytes + owned.start * row_bytes, 0x80000000, owned.n * (w[0] + w[1]))
        else
            try prompt.fill(e, dst + n * row_bytes + owned.start * row_bytes, 0, owned.n * row_bytes);
    }

    /// TF_DS_QUARTER_HC, resolved once for the whole world (WP4, docs/prefill-plan.md): every rank sends its setting;
    /// any disagreement refuses to start (the same reasoning as agreePairSum's setting check -- a rank alone on the
    /// unsharded path would leave the rest in a world-4 all-gather it never joins). No link-adjacency check is
    /// needed here, unlike agreePairSum: quarter-HC's new gathers are plain world-4 all-gathers over the base
    /// communicator (every rank always has that), not point-to-point two-rank links.
    pub fn agreeQuarterHc(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, d2: usize, cap: usize) !void {
        const want = if (std.c.getenv("TF_DS_QUARTER_HC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const w = comm.world;
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
        defer dev.free();
        const mine: u32 = @intFromBool(want);
        try dev.upload(0, std.mem.asBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 4, 1, .u32, stream);
        try stream.synchronize();
        var all: [8]u32 = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
        var on: usize = 0;
        for (all[0..w]) |x| on += x;
        if (on != 0 and on != w) {
            std.log.err("TF_DS_QUARTER_HC: set on {d} of {d} ranks (this one: {s}); start every rank with the same value", .{ on, w, if (want) "1" else "unset" });
            return error.QuarterHcMismatch;
        }
        t.qhc = on == w;
        if (t.qhc) std.log.info("TF_DS_QUARTER_HC=1: row-sharded hc-mix (encoder-half full chunks) on all {d} ranks", .{w});
        if (t.qhc) {
            // WP1's own precedent (initLinks' link warm-up; patch 0010's "an all-reduce is its own first collective,
            // distinct from the all-gather above" fix): NCCL connects a communicator's channels at a given
            // message size/protocol on first use, not just at communicator creation -- the tiny config exchange
            // just above does not warm the much larger per-layer x-gather this flag adds. Warm one real-sized
            // world-4 all-gather here (q * D bf16 elements, q = cap / world: exactly qhcGatherRows' own shape),
            // so the first row-sharded chunk's gather does not pay that connection cost live.
            if (w != 0 and cap % w == 0) {
                const q = cap / w;
                var wsend = try cuda.DeviceBuffer.alloc(d, q * d2 * 2);
                defer wsend.free();
                var wrecv = try cuda.DeviceBuffer.alloc(d, w * q * d2 * 2);
                defer wrecv.free();
                try comm.allGather(wsend.ptr, wrecv.ptr, q * d2, .bf16, stream);
                try stream.synchronize();
            }
        }
    }

    /// WP5 (docs/prefill-plan.md, owner-row transport): agreed at startup like pair_sum and qhc. Needs qhc (no
    /// row-sharded layer reads the saving otherwise) and equal pair widths (pairSumOwned reads t.pad as a tight
    /// w[t.p]-wide buffer at a row offset, pairSum's own layout, which only lines up with wmax when the two
    /// pairs' widths are equal -- refuse here rather than read misaligned bytes live, the exact class of mistake
    /// WP4's own buffer-stride bugs were).
    pub fn agreeOwnerGather(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream) !void {
        const want = if (std.c.getenv("TF_DS_OWNER_GATHER")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const w = comm.world;
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
        defer dev.free();
        const mine: u32 = @intFromBool(want);
        try dev.upload(0, std.mem.asBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 4, 1, .u32, stream);
        try stream.synchronize();
        var all: [8]u32 = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
        var on: usize = 0;
        for (all[0..w]) |x| on += x;
        if (on != 0 and on != w) {
            std.log.err("TF_DS_OWNER_GATHER: set on {d} of {d} ranks (this one: {s}); start every rank with the same value", .{ on, w, if (want) "1" else "unset" });
            return error.OwnerGatherMismatch;
        }
        t.owner_gather = on == w and t.qhc and t.ow[0] == t.ow[1];
        if (on == w and !t.owner_gather) std.log.info("TF_DS_OWNER_GATHER=1: needs TF_DS_QUARTER_HC=1 and equal pair widths (this run: qhc={}, ow={any}); keeping pairSum's full-row gather", .{ t.qhc, t.ow });
        if (t.owner_gather) std.log.info("TF_DS_OWNER_GATHER=1: a row-sharded layer's gather moves only each rank's owned quarter over the column link, not every row", .{});
    }

    /// WP6(b) (docs/prefill-plan.md, indexer row-sharding): agreed at startup like pair_sum, qhc and
    /// owner_gather, but independent of them -- it only needs a 2D engine with every rank's cap evenly
    /// divisible into 16-row-aligned quarters (idxShardRange's own check; refused per-rank there, not here, so
    /// a mismatch across ranks still shows as every rank agreeing on the flag but only some actually sharding).
    pub fn agreeIdxShard(t: *Two, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream) !void {
        const want = if (std.c.getenv("TF_DS_IDX_SHARD")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        const w = comm.world;
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + w) * 4);
        defer dev.free();
        const mine: u32 = @intFromBool(want);
        try dev.upload(0, std.mem.asBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 4, 1, .u32, stream);
        try stream.synchronize();
        var all: [8]u32 = undefined;
        if (w > all.len) return error.WorldTooLarge;
        try dev.download(4, std.mem.sliceAsBytes(all[0..w]));
        var on: usize = 0;
        for (all[0..w]) |x| on += x;
        if (on != 0 and on != w) {
            std.log.err("TF_DS_IDX_SHARD: set on {d} of {d} ranks (this one: {s}); start every rank with the same value", .{ on, w, if (want) "1" else "unset" });
            return error.IdxShardMismatch;
        }
        t.idx_shard = on == w;
        if (t.idx_shard) std.log.info("TF_DS_IDX_SHARD=1: the indexer's scoring and top-k, for layers before the candidate source on a full chunk, run only each rank's owned quarter, all-gathering the picks", .{});
    }

    /// cat_rank (Comm2D.cat_rank): `src` [n, w[p]] -> `dst` [n, w0 + w1]: this node's TP2 rank's whole width, pair 0's
    /// part then pair 1's (from its column partner, the same rank of the other pair).
    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return;
        // a prompt chunk's (64 rows or more; decode windows as before): the column partner's part alone, over the
        // two-rank link to it when the communicator has one (a quarter of the four-rank all-gather's bytes; the same
        // bytes land in dst)
        const partner = if (n >= 64) e.comm.linkTo(t.nccl[@as(u32, @intCast(t.r + 2 * t.p)) ^ 2]) else null;
        if (partner) |l| {
            const wm = @max(w[0], w[1]);
            const bytes = n * wm * esize;
            if (2 * bytes > 4 * t.part_bytes or bytes > t.part_bytes) return error.PartTooBig;
            var send = src;
            if (w[t.p] != wm) {
                try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wm * esize, w[t.p] * esize, n);
                send = t.pad;
            }
            try e.comm.linkGather(l.comm, send, t.parts, bytes, .u8, e.s);
            const row = (w[0] + w[1]) * esize;
            for (0..2) |pg| {
                const slot: usize = if (pg == t.p) l.me else 1 - l.me;
                try e.ops.copyRows(e.s, t.parts + slot * bytes, wm * esize, dst + pg * w[0] * esize, row, w[pg] * esize, n);
            }
            return;
        }
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..2) |pg| {
            const g = t.r + 2 * pg;
            try e.ops.copyRows(e.s, t.parts + t.nccl[g] * n * wmax * esize, wmax * esize, dst + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        const wmax = try t.allParts(e, src, n, .{ width, width }, esize);
        // the pair's two nodes in TP2 rank order (their slots need not be adjacent under TF_DS_2D_ORDER)
        for (0..2) |rr| try e.ops.copyRows(e.s, t.parts + t.nccl[2 * t.p + rr] * n * wmax * esize, wmax * esize, dst + rr * n * width * esize, width * esize, width * esize, n);
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.catRank(e, ch.u, t.u_full, ch.n, t.uw, 2);
        try prompt.mm(e, ch, lay.wo_b, t.u_full, t.uw[0] + t.uw[1], ch.pa, .fp32, lay.wo_b.n);
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (e.mx != null) return t.expertsNative(e, ch, lay.experts, ch.x, c.hidden, ch.pick, ch.wts, ch.xs.xd, ch.pm, n, false);
        if (n < exl3_experts.exact_rows) return round2d.experts(t, e, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
        if (t.ir != 0) {
            // parity: the rest columns of the experts this pair computes them for, the packs exchanged and assembled
            try exl3_experts2d.gateUpRest(e.ex, e.ops, e.s, lay.experts, ch.xs, t.rest_chunk.?, ch.pick, n, c.swiglu_limit);
            try t.assembleParity(e, lay.experts, ch.xs.xd, ch.pick, n * ch.xs.slots);
            return exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
        }
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
    }

    /// Format.native's experts on a 2D node, chunk or round alike: this node's pack of every expert's gate / up (its main
    /// blocks, its rest block for the experts whose rest is this pair's) into xd [rows * slots, pack] bf16, the rank's
    /// whole intermediate assembled from the two pairs' packs (the same bytes moved as the EXL3 path's fp16 rows), down
    /// for this node's output columns: out [rows, dw[p]] fp32. split_k: a decode round's (mx.gateUp).
    pub fn expertsNative(t: *const Two, e: *const prompt.Engine, ch: *const prompt.Chunk, ex: weights.Experts, x: u64, ldx: usize, pick: u64, wts: u64, xd: u64, out: u64, rows: usize, split_k: bool) !void {
        const k = e.mx.?;
        try mx.gateUp(k, e.s, e.d, ex, ch.mxe, x, ldx, pick, wts, rows, e.c.swiglu_limit, xd, ex.width, split_k);
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_gu) catch {};
        if (t.ir != 0) try t.assembleParity(e, ex, xd, pick, rows * ch.mxe.slots) else try t.catRank(e, xd, t.xd_full, rows * ch.mxe.slots, t.gu, 2);
        if (!split_k) if (prompt.ptimer) |pt| pt.mark(e.s, .m_x) catch {};
        try mx.down(k, e.s, ex, ch.mxe, t.xd_full, ex.down_k, pick, rows, out, t.dw[t.p], false, split_k);
    }

    /// Engram's projection summed over the TP2 ranks and cast, kv [n, ew0 + ew1] bf16 = bf16(g0 + g1) (model.py esum
    /// then .to(bf16)), from this node's fp32 columns `src` [n, ew[p]] with less traffic than the quarters: the row
    /// partner's columns over NCCL send / recv, one fp32 add (two operands: the same bits in either order) cast to
    /// bf16, then the column partner's bf16 sums the same way. `tmp` holds n * (ew[p] * 6 + ew[1 - p] * 2) bytes.
    pub fn engramSum(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, n: usize) !void {
        const w = t.ew;
        const mine = n * w[t.p];
        const other = tmp;
        const half = tmp + mine * 4;
        const their = half + mine * 2;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try t.exchange(e, src, mine, other, mine, .f32, 4, t.nccl[g ^ 1]);
        try e.ops.add2Bf16(e.s, src, other, half, mine);
        try t.exchange(e, half, mine, their, n * w[1 - t.p], .bf16, 2, t.nccl[g ^ 2]);
        const row = (w[0] + w[1]) * 2;
        for (0..2) |pg| {
            const from = if (pg == t.p) half else their;
            try e.ops.copyRows(e.s, from, w[pg] * 2, kv + pg * w[0] * 2, row, w[pg] * 2, n);
        }
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 3));
}
