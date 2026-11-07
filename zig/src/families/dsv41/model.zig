//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
const prompt2d = @import("prompt2d.zig");
const round = @import("round.zig");
const sampling = @import("sampling.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_rows = @import("engram_rows.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const mx = @import("mx.zig");
const ops_mod = @import("ops.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");
const tri_index = @import("tri_index.zig");

/// The served prompt chunk (TF_DS_PREFILL_CHUNK): a prompt runs in chunks that start at its multiples.
pub const chunk_rows = 2048;

/// TF_DS_ENGRAM_WORKER=0: a fill's Engram rows read by a thread a chunk (the chunk before's end to this one's first
/// Engram layer) instead of the fill's worker.
var wk_on: bool = true;
pub fn readWorkerEnv() void {
    if (std.c.getenv("TF_DS_ENGRAM_WORKER")) |v| wk_on = !std.mem.eql(u8, std.mem.span(v), "0");
}
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    listen: ?[3][4]u8 = null, // rank 0 at world 4: the address to listen on for each follower (TF_TP_LISTEN), else master
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    context: usize = 0, // a stream's extent at most (its prompt, budget and a round's rows; 0: the pool)
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

/// TF_DS_2D_GU=parity (the served switch): the 2D split's experts with half the gate / up blocks a pair and the ninth by
/// expert parity (plan.Split.parity); the rank cache must be one written for it.
pub fn parityGU() bool {
    const v = std.c.getenv("TF_DS_2D_GU") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "parity");
}

/// The parity split's decode gate / up in one launch (experts_par.cu, the same bits); TF_DS_PAR_ONE=0: the served
/// build's launches (main, then the rest's grouping and gate / up, beside it on a stream of its own with --side).
pub fn parOne() bool {
    const v = std.c.getenv("TF_DS_PAR_ONE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    ctx_cap: usize, // a stream's extent at most (Options.context, else the pool)
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    par: ?exl3_experts2d.ParKernels, // the parity split's one-launch gate / up (parOne)
    mxk: ?mx.Kernels = null, // Format.native: DeepSeek's own FP8 / MXFP4 checkpoint's kernels (mx.zig)
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    erows: ?*engram_rows.Rows = null, // TF_DS_ENGRAM_PACKED: Engram's rows from packed shards (engram_rows.zig)
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    // Track B (TF_DS_SAMPLE_CAND=1): a round's candidates on the device (vals f32, ids i64, flags i32; max_rows rows of
    // sampling.max_candidates) and on the host
    cand: u64 = 0,
    cand_vals: [round.max_rows * sampling.max_candidates]f32 = undefined,
    cand_ids: [round.max_rows * sampling.max_candidates]i64 = undefined,
    cand_flags: [round.max_rows]i32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        readWorkerEnv();
        if (o.world != 2 and o.world != 4) return error.NotPortedYet; // TP2, or the four-node 2D split (prompt2d.zig)
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        // WP6 (docs/prefill-plan.md): TF_DS_IDX_RB=2 batches the indexer's prompt-chunk scoring kernel to the
        // kit's own captured RB=2 variant (dequantizing a 64-key tile once for 2 rows instead of 1); the kit
        // never captured RB above 2. Off (RB=1, today's behavior) unless set; validate on your own ring first.
        if (std.c.getenv("TF_DS_IDX_RB")) |v| tri_index.idx_rb = std.fmt.parseInt(usize, std.mem.span(v), 10) catch tri_index.idx_rb;
        if (std.c.getenv("TF_DS_L2_SKIP")) |v| round.l2_skip = std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
        for ([_][:0]const u8{ "TF_DS_L2_ATTN_MB", "TF_DS_L2_MOE_MB", "TF_DS_L2_NEXT_MB", "TF_DS_L2_STRETCH_END_MB" }, 0..) |name, i| {
            if (std.c.getenv(name)) |v| round.l2_mb[i] = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_mb[i];
        }
        if (std.c.getenv("TF_DS_L2_DN_MB")) |v| round.l2_dn_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dn_mb;
        if (std.c.getenv("TF_DS_L2_DEC_MB")) |v| round.l2_dec_mb = std.fmt.parseInt(u64, std.mem.span(v), 10) catch round.l2_dec_mb;
        if (std.c.getenv("TF_EXL3_L2_DELAY_NS")) |v| ops_mod.l2_delay_ns = std.fmt.parseInt(c_longlong, std.mem.span(v), 10) catch ops_mod.l2_delay_ns;
        if (std.c.getenv("TF_EXL3_L2_RATE_GBPS")) |v| ops_mod.l2_rate_gbps = std.fmt.parseFloat(f64, std.mem.span(v)) catch ops_mod.l2_rate_gbps;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool;
        m.ctx_cap = if (o.context != 0) @min(o.context, o.pool) else o.pool;
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.erows = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.two = null;
        m.rdma_rings = null;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer m.stream.deinit();
        // rank 0's ports for its followers listen before its weights load: a follower that loads first connects and
        // waits in its port's backlog instead of retrying until rank 0 and the followers before it are in
        m.fan = try prompt2d.Fan.listenAt(o.master, o.listen, o.port, o.rank, o.world);
        errdefer m.fan.close();

        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer m.cache_dir.close(io);
        // world 4: the 2D node this NCCL rank serves (TF_DS_2D_ORDER; its weight file is that node's)
        const ord = if (o.world == 4) try prompt2d.order() else [4]u32{ 0, 1, 2, 3 };
        const role: u32 = if (o.world == 4) ord[o.rank] else o.rank;
        const cache_name = try rank_cache.find(a, io, m.cache_dir, role, o.world);
        m.ix = try rank_cache.open(gpa, io, m.cache_dir, cache_name);
        errdefer m.ix.deinit();
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer m.cache_file.close(io);
        var sp = try prompt2d.split(role, o.world);
        if (std.c.getenv("TF_DS_DUMP")) |dd| {
            prompt.dump_dir = std.mem.span(dd);
            prompt.dump_node = role;
            // WP1's exit gate wants several layers' tensors in one dump (x, pa, pm, h at layers 0, 8, 19): accept a
            // comma list, not just one layer, so one instrumented restart covers all of them.
            if (std.c.getenv("TF_DS_DUMP_LAYER")) |dl| {
                var it = std.mem.splitScalar(u8, std.mem.span(dl), ',');
                var i: usize = 0;
                while (it.next()) |tok| : (i += 1) {
                    if (i >= prompt.dump_layers.len) break;
                    prompt.dump_layers[i] = std.fmt.parseInt(i64, tok, 10) catch -1;
                }
            }
        }
        if (std.c.getenv("TF_DS_DRAFT_PROF")) |v| draft.dprof = v[0] == '1';
        if (std.c.getenv("TF_DS_DRAFT_LINK")) |v| draft.pair_link = v[0] != '0';
        if (std.c.getenv("TF_DS_CONF_CAL")) |v| {
            var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
            draft.conf_cal[0] = std.fmt.parseFloat(f64, it.next() orelse "1") catch return error.BadConfCal;
            draft.conf_cal[1] = std.fmt.parseFloat(f64, it.next() orelse "0") catch return error.BadConfCal;
        }
        if (std.c.getenv("TF_DS_DRAFT_MS")) |v| draft.cost_ms = std.fmt.parseFloat(f64, std.mem.span(v)) catch return error.BadDraftMs;
        if (std.c.getenv("TF_DS_DRAFT_LOG")) |v| {
            draft.dlog = std.mem.span(v);
            draft.ddump_node = role;
        }
        if (std.c.getenv("TF_DS_DRAFT_DUMP")) |dd| {
            draft.ddump = std.mem.span(dd);
            draft.ddump_node = role;
            if (std.c.getenv("TF_DS_DRAFT_DUMP_PASS")) |dp| draft.ddump_pass = std.fmt.parseInt(usize, std.mem.span(dp), 10) catch 2;
        }
        sp.parity = sp.pair != null and parityGU(); // TF_DS_2D_GU=parity: the balanced experts split
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' links (rank 0 takes its followers in order; a follower retries its connect for TF_DS_CONNECT_S)
        // and NCCL
        try m.fan.join(io, o.master, o.port, o.rank, o.world, prompt2d.connectSeconds());
        m.comm = try m.fan.comm(o.rank, o.world);
        errdefer m.comm.deinit();
        // world 4 on a ring without a switch (TF_DS_LINK_GRAPHS: this node's NCCL graph files of its links to the previous
        // and the next rank): the partners' exchanges over two-rank links, each pinned to its cable
        if (o.world == 4) if (std.c.getenv("TF_DS_LINK_GRAPHS")) |lg| try openLinks(gpa, d, &m.comm, m.stream, std.mem.span(lg));
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            m.ring = try openRing(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream);
        }
        errdefer if (m.ring) |r| r.destroy(gpa);
        errdefer if (m.rdma_mod) |*x| x.unload();

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer m.exk.unload();
        if (!cuda.kernels.available) return error.NoKernelImages;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer m.ops.unload();
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer m.ex.unload();
        m.par = if (sp.parity and parOne() and c.format == .exl3) try exl3_experts2d.ParKernels.load(d, cuda.kernels.dsv41_experts_par) else null;
        errdefer if (m.par) |*x| x.unload();
        m.mxk = if (c.format == .native) try mx.Kernels.load(d, cuda.kernels.dsv41_native) else null;
        errdefer if (m.mxk) |*x| x.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), every position a stream reaches
        const half = c.rope_dim / 2;
        const rope_rows = m.ctx_cap + round.max_rows + 1;
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table of fewer rows than a stream's context
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer m.arena.deinit();
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            m.two.?.nccl = prompt2d.ncclOf(ord);
            try m.two.?.agreePairSum(d, &m.comm, m.stream, o.rdma_devices != null);
            try m.two.?.agreeQuarterHc(d, &m.comm, m.stream, c.hidden, chunk_rows);
            try m.two.?.agreeOwnerGather(d, &m.comm, m.stream);
            try m.two.?.agreeIdxShard(d, &m.comm, m.stream);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, role, ord, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid(), .pdl = rdmaPdl() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        logPdlControls(m.comm.rank);
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        if (m.par) |*x| m.eng.par = x;
        if (m.mxk) |*x| m.eng.mx = x;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, chunk_rows, m.ctx_cap); // a stream's keys at most
        m.caches = try prompt.Caches.init(&m.eng, &m.arena, o.pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.ctx_cap); // a round's deepest bucket: a stream's context
        m.amax = try m.arena.take(round.max_rows * 4);
        m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        m.eng.agree_flag = try m.arena.take(64); // G0.3: the graph captures' agreement byte
        if (o.graphs) m.graphs = round.Graphs.init(gpa);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (m.two) |t| if (t.ir != 0 and o.side and m.par == null) try m.rd.usePar(&m.eng); // parity in two launches: the rest beside the main gate / up
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, chunk_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
            // TF_DS_ENGRAM_PACKED (a folder of tf-dsv41-engram-pack's shards): a row is one read, kept in a row cache
            // (TF_DS_ENGRAM_CACHE_MIB, default 2048, 0 none) and decoded on the GPU; TF_DS_ENGRAM_OWNED=1: each node
            // reads only its range of the rows and a byte max over the nodes fills the rest; TF_DS_ENGRAM_INSTREAM=1:
            // a round's landing as a host step on the stream (with TF_DS_ENGRAM_AIO)
            if (std.c.getenv("TF_DS_ENGRAM_PACKED")) |pd| {
                // packed shards hold FP8 rows and decode them on the GPU; NVIDIA's MXFP4 tables are read as stored
                var eit = m.tables.?.layers.valueIterator();
                while (eit.next()) |t| if (t.dim() != t.row_w) return error.PackedEngramIsFp8;
                const owned = engram_rows.envOn("TF_DS_ENGRAM_OWNED");
                const eh = &m.eh.?;
                const cols = eh.hasher.cols();
                const k = eh.hi - eh.lo;
                const sl = if (owned) cols else k;
                const rec = c.engram_head_dim + c.engram_head_dim / 32;
                const mib: usize = if (std.c.getenv("TF_DS_ENGRAM_CACHE_MIB")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
                m.erows = try engram_rows.Rows.open(gpa, d, &m.tables.?, m.epool.?, m.aio, m.paio, .{
                    .dir = std.mem.span(pd),
                    .owned = owned,
                    .node = role,
                    .nodes = o.world,
                    .lo = eh.lo,
                    .k = k,
                    .cols = cols,
                    .dim = c.engram_head_dim,
                    .layer_ids = c.engram_layers.slice(),
                    .round_rows = round.max_rows,
                    .chunk_rows = chunk_rows,
                    .cache_bytes = mib << 20,
                    .instream = engram_rows.envOn("TF_DS_ENGRAM_INSTREAM"),
                    .round_dev = try m.arena.take(round.max_rows * sl * rec),
                    .chunk_dev = try m.arena.take(chunk_rows * sl * rec),
                });
                eh.pk = m.erows;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision and c.format == .exl3) { // (Format.native: the checkpoint's own gate.bias_vl, loaded with the weights)
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, chunk_rows);
        m.ids64 = try a.alloc(i64, chunk_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        return m;
    }

    pub fn close(m: *Model) void {
        m.joinAhead();
        m.stream.synchronize() catch {};
        if (m.img_buf) |*b| b.free();
        m.spans.deinit(m.gpa);
        if (m.graphs) |*g| g.deinit();
        if (m.pgraphs) |*g| g.deinit();
        if (m.dr) |*dr| dr.deinit();
        if (m.timer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropPar();
        if (m.erows) |r| r.close();
        if (m.aio) |x| x.deinit();
        if (m.paio) |x| x.deinit();
        if (m.epool) |p| p.deinit(m.gpa);
        if (m.tables) |*t| t.close();
        if (m.rdma_rings) |*r| r.close(m.gpa);
        m.blas.close();
        m.arena.deinit();
        m.rope_buf.free();
        if (m.par) |*x| x.unload();
        if (m.mxk) |*x| x.unload();
        m.ex.unload();
        m.ops.unload();
        m.exk.unload();
        m.lg.unload();
        m.pf.unload();
        m.set.deinit();
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (m.rdma_mod) |*x| x.unload();
        m.comm.deinit();
        m.fan.close();
        m.w.deinit();
        m.cache_file.close(m.io);
        m.ix.deinit();
        m.cache_dir.close(m.io);
        m.stream.deinit();
        m.host.deinit();
        m.gpa.destroy(m);
    }

    /// The two-rank ring links (Comm.initLinks): every rank's id for its next link gathered over the world communicator,
    /// then both links made with their graph files ("prev,next", from TF_DS_LINK_GRAPHS).
    fn openLinks(gpa: std.mem.Allocator, d: *const cuda.Driver, comm: *comm_mod.Comm, stream: cuda.Stream, spec: []const u8) !void {
        var it = std.mem.splitScalar(u8, spec, ',');
        const prev = it.next() orelse return error.BadLinkGraphs;
        const next = it.next() orelse return error.BadLinkGraphs;
        if (it.next() != null) return error.BadLinkGraphs;
        const pz = try gpa.dupeSentinel(u8, prev, 0);
        defer gpa.free(pz);
        const nz = try gpa.dupeSentinel(u8, next, 0);
        defer gpa.free(nz);
        const nccl = cuda.nccl;
        const per = @sizeOf(nccl.UniqueId);
        var mine: nccl.UniqueId = undefined;
        try comm.lib.check(comm.lib.api.ncclGetUniqueId(&mine), "ncclGetUniqueId");
        var dev = try cuda.DeviceBuffer.alloc(d, (1 + comm.world) * per);
        defer dev.free();
        try dev.upload(0, std.mem.asBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + per, per, .u8, stream);
        try stream.synchronize();
        const ids = try gpa.alloc(nccl.UniqueId, comm.world);
        defer gpa.free(ids);
        try dev.download(per, std.mem.sliceAsBytes(ids));
        try comm.initLinks(ids, .{ pz, nz });
        // NCCL connects a communicator at its first collective (~0.1 s on the ring's links): warm both links here, in
        // initLinks' order (each link's two ranks reach it together), so no prompt's first exchange over a link (the
        // first Engram layer's row-partner sum) pays it. WP1 hardening (Codex's open item 4): an all-reduce is its own
        // first collective on a link (distinct from the all-gather above), so TF_DS_PAIR_SUM's first pairSum call
        // would otherwise pay that connection cost on the live prompt; warm one fp32 all-reduce here too.
        const order: [2]usize = if (comm.rank % 2 == 0) .{ 1, 0 } else .{ 0, 1 };
        for (order) |which| if (comm.links[which]) |l| {
            try comm.linkGather(l, dev.ptr, dev.ptr + per, per, .u8, stream);
            try stream.synchronize();
            try comm.linkSum(l, dev.ptr, 1, stream);
            try stream.synchronize();
        };
    }

    /// The decode plan's G0.2 controls (TF_DS_DEC_PDL, TF_DS_RDMA_PDL, TF_DS_TRITON_PDL_ATTR, TF_DS_PDL), one line on each
    /// rank when any is off, so a run can show that every rank has the same ones.
    fn logPdlControls(rank: usize) void {
        const off = struct {
            fn f(name: [*:0]const u8) bool {
                return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
            }
        }.f;
        const native = off("TF_DS_DEC_PDL");
        const ring = !rdmaPdl();
        const tri_attr = off("TF_DS_TRITON_PDL_ATTR");
        const all = off("TF_DS_PDL");
        if (!(native or ring or tri_attr or all)) return;
        const w = struct {
            fn f(o: bool) []const u8 {
                return if (o) "off" else "on";
            }
        }.f;
        std.log.info("decode PDL controls on rank {d}: native decode kernels {s}, RDMA kernels {s}, Triton consumer launch attribute {s}, every launch attribute {s}", .{ rank, w(native or all), w(ring or all), w(tri_attr or all), w(all) });
    }

    /// TF_DS_RDMA_PDL=0 (or the Python build's TF_RDMA_PDL=0): the RDMA rings' kernels as ordinary launches with their
    /// pdl argument 0 (no griddepcontrol in them): the decode plan's G0.2 control (b). Unset or any other value: as served.
    pub fn rdmaPdl() bool {
        for ([_][*:0]const u8{ "TF_DS_RDMA_PDL", "TF_RDMA_PDL" }) |name| if (std.c.getenv(name)) |v| return !std.mem.eql(u8, std.mem.span(v), "0");
        return true;
    }

    /// TF_DS_RDMA_GID: the RDMA rings' GID index on every device; unset, each device's first RoCE v2 IPv4 GID.
    pub fn rdmaGid() ?u8 {
        const v = std.c.getenv("TF_DS_RDMA_GID") orelse return null;
        return std.fmt.parseInt(u8, std.mem.span(v), 10) catch null;
    }

    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        if (comm.world != 2) return error.NotTwoRanks;
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, 2, .{ .max_bytes = ring_bytes, .gid_index = rdmaGid() });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        const mine = [2]rdma.Info{ r.info(0), r.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 3 * 2 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 2 * per, .u8, stream);
        try stream.synchronize();
        var all: [2][2]rdma.Info = undefined;
        try dev.download(2 * per, std.mem.sliceAsBytes(&all));
        try r.connect(&.{ all[0][rank], all[1][rank] });
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + 2 * per, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    pub fn joinAhead(m: *Model) void {
        if (m.ahead) |th| th.join();
        m.ahead = null;
        if (m.eh) |*eh| prompt.engramWorkerStop(eh);
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        const c = m.eng.c;
        const cs = &m.caches;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
            const n = at / r;
            try m.copySpan(cs.comp_codes[i], from / r, to / r, n, c.head_dim / 2);
            try m.copySpan(cs.comp_scales[i], from / r, to / r, n, c.head_dim / 16);
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
    }

    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.stream.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        if (n == 0 or n > chunk_rows or seq.len != start + n) return error.BadChunk;
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        var shared: prompt.Shared = .{};
        const pt = m.ptimer;
        prompt.ptimer = pt;
        defer prompt.ptimer = null;
        if (pt) |t| try t.mark(m.stream, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        const taps = m.dr != null;
        var floor: usize = 0;
        var kv_done: ?usize = null;
        // Engram rows read ahead while the GPU runs the layers before them. A fill of the prompt (seq a prefix of
        // fill_seq): the fill's worker thread (prompt.engramWorker), up to a chunk ahead of the layers, so a chunk's
        // table reads overlap the chunk before it. Else (a chunk alone): a thread from the chunk's start, the first
        // layer's rows then the next's once the first is applied.
        const use_wk = m.eh != null and wk_on and m.fill_seq.len >= start + n and m.fill_seq.ptr == seq.ptr;
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        defer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (use_wk) {
                if (ahead) |th| th.join();
                ahead = null;
                const same = eh.wk_thread != null and eh.wk_seq.ptr == m.fill_seq.ptr and eh.wk_seq.len == m.fill_seq.len and
                    start >= eh.wk_start and (start - eh.wk_start) % chunk_rows == 0 and
                    (start - eh.wk_start) / chunk_rows <= eh.wk_used[0].load(.acquire);
                if (!same) try prompt.engramWorkerStart(eh, c, m.fill_seq, start, chunk_rows);
            } else {
                prompt.engramWorkerStop(eh);
                if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                    ahead.?.join();
                    ahead = null;
                }
                if (ahead == null) {
                    eh.ahead_start = start;
                    eh.ahead_n = n;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
                }
            }
        }
        for (0..c.layers) |li| {
            if (li == c.layers / 2) {
                try prompt.qhcResyncBoundary(e, ch, li);
                if (!try prompt.replayCut(e, ch, &m.view, &shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    return false;
                }
                floor = replay;
                kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                if (use_wk and j < 2) {
                    // the worker's rows, or its records with packed shards (engram_rows.zig)
                    if (eh.pk != null) {
                        eh.use_stage = try prompt.engramWorkerStage(eh, j, start, chunk_rows);
                    } else eh.use_rows = try prompt.engramWorkerRows(eh, j, start, chunk_rows);
                    defer {
                        eh.use_rows = null;
                        eh.use_stage = null;
                    }
                    try prompt.engramApply(e, ch, eh, li, seq);
                    // the rows are uploaded (a pageable copy is staged before it returns): the slot is the worker's
                    eh.wk_used[j].store((start - eh.wk_start) / chunk_rows + 1, .release);
                } else {
                    if (ahead) |th| {
                        th.join();
                        ahead = null;
                    }
                    if (eh.ahead_err) |err| return err;
                    try prompt.engramApply(e, ch, eh, li, seq);
                    const ne = @min(2, c.engram_layers.slice().len);
                    if (j + 1 < ne) {
                        if (!eh.ahead_ready[j + 1])
                            ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                    }
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j, li);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &shared, li, m.ring_view[li], floor, kv_done != null and kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga, li);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm, li + 1);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            // WP1's exit gate: the gathers' own output (what TF_DS_PAIR_SUM changes), both sublayers, at every
            // requested layer of the first chunk of at least 64 rows (prompt.dump_layers; dump_done stops it after
            // that chunk, below). The >= 64 guard matters: TF_DS_PAIR_SUM only engages at 64+ rows (prompt.gather),
            // so a smoke test or validate.py's short prompts would otherwise spend the one-shot dump on a chunk
            // where pair-sum never ran at all, comparing the untouched quarters path with itself.
            if (prompt.dump_dir != null and !prompt.dump_done and ch.n >= 64) {
                prompt.dump(e, li, "ga", ch.ga, 2 * ch.n * c.hidden * 4);
                prompt.dump(e, li, "gm", ch.gm, 2 * ch.n * c.hidden * 4);
            }
            prompt.endLayer(ch);
        }
        try prompt.head(e, ch);
        if (ch.n >= 64) prompt.dump_done = prompt.dump_dir != null;
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        return true;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.pool_cap, null, if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.erows) |r| try r.roundCheck(); // a stream host step's failed read fails the round
        if (m.prof) m.t_absorb = m.now() - t1;
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        // TF_DS_CLOCK_RATIO: the compress ratio whose layers mode 3 times (2 unless set)
        const r: u8 = if (std.c.getenv("TF_DS_CLOCK_RATIO")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch 2 else 2;
        m.rd.clk_ratio = if (marks == 3) r else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        const d = m.ctx.d;
        try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    /// Track B: the round's first `count` candidates a row (sampling.topCandidates' set and order) on the host, from the
    /// device's logits: vals and ids [R, count], flags [R] (non-zero: that row must be read whole, roundLogits).
    pub fn roundCandidates(m: *Model, R: usize, count: usize) !struct { vals: []const f32, ids: []const i64, flags: []const i32 } {
        const d = m.ctx.d;
        const mc = sampling.max_candidates;
        const vals = m.cand;
        const ids = m.cand + round.max_rows * mc * 4;
        const flags = ids + round.max_rows * mc * 8;
        try m.ops.topCandRows(m.stream, m.rd.logits, m.vocab, m.vocab, R, count, vals, ids, flags);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_vals, vals, R * count * 4), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_ids, ids, R * count * 8), "cuMemcpyDtoH");
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_flags, flags, R * 4), "cuMemcpyDtoH");
        return .{ .vals = m.cand_vals[0 .. R * count], .ids = m.cand_ids[0 .. R * count], .flags = m.cand_flags[0..R] };
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0 .. R * m.vocab];
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null);
    }
};
