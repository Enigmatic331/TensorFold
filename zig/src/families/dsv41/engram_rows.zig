//! Engram's rows from packed shards (TF_DS_ENGRAM_PACKED, engram_pack.zig) instead of the tables' separate weight and
//! scale rows: each slot's record staged on the host (a cache hit copied, a miss read in one go, a row another node
//! owns zeroed), uploaded as it is, combined over the nodes when each owns a range of the rows (TF_DS_ENGRAM_OWNED: a
//! byte max, one node holding a slot's bytes and the rest zeros) and decoded to bf16 on the GPU (engram_gpu.zig).
//! A layer's rows keep the host path's bits; only where they come from changes.
const std = @import("std");
const cuda = @import("cuda");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const engram_pack = @import("engram_pack.zig");
const engram_cache = @import("engram_cache.zig");
const engram_gpu = @import("engram_gpu.zig");
const comm = @import("comm.zig");

/// Whether environment switch `name` is set to anything but 0.
pub fn envOn(name: [:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// A staging buffer: rows x slots records, the misses' row ids and where each lands, their AIO batch.
pub const Stage = struct {
    j: usize, // the Engram layer slot it serves (its cache)
    bytes: []u8,
    ids: []i64,
    dsts: [][*]u8,
    misses: usize = 0,
    batch: u64 = 0, // the misses' AIO batch (0: none)
    host: ?cuda.HostBuffer = null, // pinned bytes (a round's), else the allocator's
    rows: ?*Rows = null, // what a stream's host step reads
    failed: std.atomic.Value(bool) = .init(false),

    fn init(gpa: std.mem.Allocator, d: ?*const cuda.Driver, j: usize, records: usize, rec: usize) !Stage {
        var st: Stage = .{ .j = j, .bytes = &.{}, .ids = &.{}, .dsts = &.{} };
        errdefer st.deinit(gpa);
        if (d) |dr| {
            st.host = try cuda.HostBuffer.alloc(dr, records * rec);
            st.bytes = st.host.?.bytes[0 .. records * rec];
        } else st.bytes = try gpa.alloc(u8, records * rec);
        st.ids = try gpa.alloc(i64, records);
        st.dsts = try gpa.alloc([*]u8, records);
        return st;
    }

    fn deinit(st: *Stage, gpa: std.mem.Allocator) void {
        if (st.host) |*h| h.free() else if (st.bytes.len > 0) gpa.free(st.bytes);
        if (st.ids.len > 0) gpa.free(st.ids);
        if (st.dsts.len > 0) gpa.free(st.dsts);
        st.* = undefined;
    }
};

pub const Options = struct {
    dir: []const u8, // the packed shards
    owned: bool, // node `node` of `nodes` reads its own range of each table's rows (else the whole tables, node 0 of 1)
    node: u32,
    nodes: u32,
    lo: usize, // this node's hash columns [lo, lo + k) of a row's `cols`
    k: usize,
    cols: usize,
    dim: usize, // a row's values (engram_head_dim)
    layer_ids: []const u16, // the Engram layers in order
    round_rows: usize, // a round's rows at most
    chunk_rows: usize, // a prompt chunk's
    cache_bytes: usize, // the row cache over the layers (0: none)
    instream: bool, // a round's landing as a host step on the stream (needs the round's AIO ring)
    round_dev: u64, // device records of a round (round_rows * slots * record bytes)
    chunk_dev: u64, // of a chunk (chunk_rows * slots * record bytes)
};

pub const Rows = struct {
    gpa: std.mem.Allocator,
    owned: bool,
    node: u32,
    nodes: u32,
    lo: usize,
    k: usize,
    cols: usize,
    dim: usize,
    rec: usize,
    layers: usize,
    shards: engram_pack.Shards,
    pk: [2]engram_io.Packed,
    caches: [2]?engram_cache.Cache,
    kernels: engram_gpu.Kernels,
    round: [2]Stage, // a round's records by Engram layer slot (pinned)
    chunk: Stage, // a prompt chunk's, read on the pass's thread
    ahead: [2]Stage, // a prompt chunk's read ahead, by Engram layer slot
    wk: [2][2]Stage, // a fill's worker's records, by chunk parity and Engram layer slot (prompt.engramWorker)
    round_dev: u64,
    chunk_dev: u64,
    pool: *engram_io.Pool,
    aio: ?*engram_aio.Aio, // a round's reads (TF_DS_ENGRAM_AIO), else the pool
    paio: ?*engram_aio.Aio, // a chunk's
    instream: bool,

    /// The shards of node o.node of o.nodes (owned) or the whole ones, checked against `tables`; the kernels; the
    /// staging buffers and the caches. Needs the CUDA context current.
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, tables: *const engram_io.Tables, pool: *engram_io.Pool, aio: ?*engram_aio.Aio, paio: ?*engram_aio.Aio, o: Options) !*Rows {
        if (o.layer_ids.len == 0 or o.layer_ids.len > 2 or o.dim == 0 or o.dim % 32 != 0 or o.lo + o.k > o.cols) return error.EngramShape;
        if (o.instream and aio == null) return error.EngramInStreamNeedsAio;
        if (!cuda.kernels.available) return error.NoKernelImages;
        const rec = o.dim + o.dim / 32;
        const nodes: u32 = if (o.owned) o.nodes else 1;
        const node: u32 = if (o.owned) o.node else 0;
        if (node >= nodes) return error.EngramShape;
        const sl = if (o.owned) o.cols else o.k;
        var shards = try engram_pack.Shards.open(gpa, o.dir, tables, node, nodes);
        errdefer shards.close();
        var pks: [2]engram_io.Packed = undefined;
        for (o.layer_ids, 0..) |li, j| {
            pks[j] = shards.layers.get(li) orelse return error.MissingEngramTable;
            if (pks[j].rec != rec) return error.EngramShape;
        }
        var kernels = try engram_gpu.Kernels.load(d, cuda.kernels.dsv41_engram);
        errdefer kernels.unload();
        var round0 = try Stage.init(gpa, d, 0, o.round_rows * sl, rec);
        errdefer round0.deinit(gpa);
        var round1 = try Stage.init(gpa, d, 1, o.round_rows * sl, rec);
        errdefer round1.deinit(gpa);
        var chunk = try Stage.init(gpa, null, 0, o.chunk_rows * sl, rec);
        errdefer chunk.deinit(gpa);
        var ahead0 = try Stage.init(gpa, null, 0, o.chunk_rows * sl, rec);
        errdefer ahead0.deinit(gpa);
        var ahead1 = try Stage.init(gpa, null, 1, o.chunk_rows * sl, rec);
        errdefer ahead1.deinit(gpa);
        var wk: [2][2]Stage = undefined;
        var made: usize = 0;
        errdefer for (0..made) |i| wk[i / 2][i % 2].deinit(gpa);
        for (0..4) |i| {
            wk[i / 2][i % 2] = try Stage.init(gpa, null, i % 2, o.chunk_rows * sl, rec);
            made += 1;
        }
        var caches: [2]?engram_cache.Cache = .{ null, null };
        errdefer for (&caches) |*c| {
            if (c.*) |*x| x.deinit(gpa);
        };
        if (o.cache_bytes > 0) {
            for (0..o.layer_ids.len) |j| caches[j] = try engram_cache.Cache.init(gpa, o.cache_bytes / o.layer_ids.len, rec, 16);
        }
        const r = try gpa.create(Rows);
        r.* = .{
            .gpa = gpa,
            .owned = o.owned,
            .node = node,
            .nodes = nodes,
            .lo = o.lo,
            .k = o.k,
            .cols = o.cols,
            .dim = o.dim,
            .rec = rec,
            .layers = o.layer_ids.len,
            .shards = shards,
            .pk = pks,
            .caches = caches,
            .kernels = kernels,
            .round = .{ round0, round1 },
            .chunk = chunk,
            .ahead = .{ ahead0, ahead1 },
            .wk = wk,
            .round_dev = o.round_dev,
            .chunk_dev = o.chunk_dev,
            .pool = pool,
            .aio = aio,
            .paio = paio,
            .instream = o.instream,
        };
        for (&r.round) |*st| st.rows = r;
        return r;
    }

    pub fn close(r: *Rows) void {
        const gpa = r.gpa;
        for (&r.round) |*st| st.deinit(gpa);
        r.chunk.deinit(gpa);
        for (&r.ahead) |*st| st.deinit(gpa);
        for (&r.wk) |*row| {
            for (row) |*st| st.deinit(gpa);
        }
        for (&r.caches) |*c| {
            if (c.*) |*x| x.deinit(gpa);
        }
        r.kernels.unload();
        r.shards.close();
        gpa.destroy(r);
    }

    /// A row's slots this node stages: every hash column (owned), else its own k.
    pub fn slots(r: *const Rows) usize {
        return if (r.owned) r.cols else r.k;
    }

    /// `st` for `n` rows of Engram layer slot `j` from `hashes` ([n][layers][cols] row ids): each slot a cache hit's
    /// record, zeros (a row another node owns), or a miss listed with where it lands.
    fn plan(r: *Rows, st: *Stage, j: usize, hashes: []const i64, n: usize, layers: usize) !void {
        const sl = r.slots();
        if (n * sl * r.rec > st.bytes.len or hashes.len < n * layers * r.cols or j >= r.layers) return error.EngramShape;
        const first = if (r.owned) 0 else r.lo;
        const pk = r.pk[j];
        const cache: ?*engram_cache.Cache = if (r.caches[j]) |*c| c else null;
        var m: usize = 0;
        for (0..n) |row| {
            for (0..sl) |q| {
                const x = hashes[(row * layers + j) * r.cols + first + q];
                if (x < 0) return error.BadEngramRow;
                const id: u64 = @intCast(x);
                const dst = st.bytes[(row * sl + q) * r.rec ..][0..r.rec];
                if (r.owned and (id < pk.lo or id >= pk.hi)) {
                    @memset(dst, 0);
                    continue;
                }
                if (cache) |c| {
                    if (c.get(id, dst)) continue;
                }
                st.ids[m] = x;
                st.dsts[m] = dst.ptr;
                m += 1;
            }
        }
        st.j = j;
        st.misses = m;
        st.batch = 0;
    }

    /// The staged misses' records kept in their layer's cache.
    fn keep(r: *Rows, st: *const Stage) void {
        const c = if (r.caches[st.j]) |*x| x else return;
        for (st.ids[0..st.misses], st.dsts[0..st.misses]) |x, dst| c.put(@intCast(x), dst[0..r.rec]);
    }

    /// The staged misses read now (an AIO ring when given, else the reader pool) and kept.
    fn readMisses(r: *Rows, st: *Stage, aio: ?*engram_aio.Aio) !void {
        const m = st.misses;
        if (m == 0) return;
        const pk = r.pk[st.j];
        if (aio) |x| try aioRecords(x, pk, st.ids[0..m], st.dsts[0..m]) else try r.pool.gatherRecords(pk, st.ids[0..m], st.dsts[0..m]);
        r.keep(st);
    }

    /// A round's rows of Engram layer slot `j` staged from `hashes` ([rows][layers][cols]), the misses submitted to the
    /// round's AIO ring when there is one (else they are read when the layer lands).
    pub fn roundAhead(r: *Rows, j: usize, hashes: []const i64, rows: usize, layers: usize) !void {
        const st = &r.round[j];
        st.failed.store(false, .monotonic);
        try r.plan(st, j, hashes, rows, layers);
        if (st.misses > 0) {
            if (r.aio) |x| {
                if (st.misses <= x.slots) st.batch = try x.startRecords(r.pk[j], st.ids[0..st.misses], st.dsts[0..st.misses]);
            }
        }
    }

    /// Engram layer slot `j`'s rows of the round into `out` (bf16 [rows][k][dim]) in `s`'s order: the misses landed
    /// (read now when none were submitted) and kept, the records uploaded, combined over the nodes (owned) and
    /// decoded. In stream mode the landing is a host step on `s`, so the host goes on: roundCheck after the round.
    pub fn roundLand(r: *Rows, j: usize, rows: usize, c: *const comm.Comm, s: cuda.Stream, out: u64) !void {
        const st = &r.round[j];
        const sl = r.slots();
        const bytes = rows * sl * r.rec;
        if (st.batch != 0 and r.instream) {
            try s.hostFunc(stepLand, st);
        } else if (st.batch != 0) {
            const id = st.batch;
            st.batch = 0;
            try r.aio.?.wait(id);
            r.keep(st);
        } else try r.readMisses(st, r.aio);
        try upload(s, r.round_dev, st.bytes[0..bytes]);
        if (r.owned and r.nodes > 1) try c.maxBytes(r.round_dev, bytes, s);
        try r.kernels.decodeRows(s, r.round_dev, out, rows, sl, if (r.owned) r.lo else 0, r.k, r.dim);
    }

    /// A round's layer slot `j` that will not land (its round failed, G0.3): its submitted batch waited for and dropped
    /// (nothing kept), so the slot's staging and batch id are free for the next round. Errors are ignored.
    pub fn roundDrain(r: *Rows, j: usize) void {
        const st = &r.round[j];
        if (st.batch == 0) return;
        const id = st.batch;
        st.batch = 0;
        if (r.aio) |x| x.wait(id) catch {};
    }

    /// After a round's synchronize: error.ReadFailed when a stream host step's read failed.
    pub fn roundCheck(r: *Rows) !void {
        var bad = false;
        for (&r.round) |*st| bad = st.failed.swap(false, .acq_rel) or bad;
        if (bad) return error.ReadFailed;
    }

    /// A prompt chunk's rows of Engram layer slot `j` staged in `st` (the chunk's own, or a read-ahead's) and read.
    pub fn chunkRead(r: *Rows, st: *Stage, j: usize, hashes: []const i64, n: usize, layers: usize) !void {
        try r.plan(st, j, hashes, n, layers);
        try r.readMisses(st, r.paio);
    }

    /// A staged chunk's records uploaded, combined over the nodes (owned) and decoded into `out` (bf16 [n][k][dim]).
    pub fn chunkApply(r: *Rows, st: *const Stage, n: usize, c: *const comm.Comm, s: cuda.Stream, out: u64) !void {
        const sl = r.slots();
        const bytes = n * sl * r.rec;
        try upload(s, r.chunk_dev, st.bytes[0..bytes]);
        if (r.owned and r.nodes > 1) try c.maxBytes(r.chunk_dev, bytes, s);
        try r.kernels.decodeRows(s, r.chunk_dev, out, n, sl, if (r.owned) r.lo else 0, r.k, r.dim);
    }
};

/// A stream's host step for a staged round layer: its AIO batch waited for and its misses kept (no CUDA call).
fn stepLand(data: ?*anyopaque) callconv(.c) void {
    const st: *Stage = @ptrCast(@alignCast(data.?));
    const r = st.rows.?;
    const id = st.batch;
    st.batch = 0;
    r.aio.?.wait(id) catch {
        st.failed.store(true, .release);
        return;
    };
    r.keep(st);
}

/// Bytes to the device in `s`'s order (pinned bytes copy when the stream gets there; pageable ones are staged now).
fn upload(s: cuda.Stream, dst: u64, src: []const u8) !void {
    if (src.len == 0) return;
    try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, s.handle), "cuMemcpyHtoDAsync");
}

/// Records of `ids` from `pk` into `dsts` through `aio`: batches of half its ring, all submitted, waited for in order.
fn aioRecords(aio: *engram_aio.Aio, pk: engram_io.Packed, ids: []const i64, dsts: []const [*]u8) !void {
    const per = @max(aio.slots / 2, 1);
    var batch: [64]u64 = undefined;
    var nb: usize = 0;
    var at: usize = 0;
    errdefer for (batch[0..nb]) |id| aio.wait(id) catch {};
    while (at < ids.len) : (at += per) {
        const n = @min(per, ids.len - at);
        if (nb == batch.len) return error.TooManyRows;
        batch[nb] = try aio.startRecords(pk, ids[at..][0..n], dsts[at..][0..n]);
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

test "an owned plan stages hits, zeros another node's rows and lists the misses" {
    const a = std.testing.allocator;
    var cache = try engram_cache.Cache.init(a, 64 * 4, 4, 4);
    defer cache.deinit(a);
    cache.put(12, &.{ 1, 2, 3, 4 });
    var st = try Stage.init(a, null, 0, 2 * 3, 4);
    defer st.deinit(a);
    var r: Rows = undefined;
    r.owned = true;
    r.lo = 1;
    r.k = 2;
    r.cols = 3;
    r.rec = 4;
    r.layers = 1;
    r.pk = .{ .{ .fd = -1, .base = 0, .rec = 4, .rows = 40, .lo = 10, .hi = 20 }, undefined };
    r.caches = .{ cache, null };
    // two rows, one layer, three columns: 12 cached, 5 and 25 another node's, 11 / 13 / 19 misses
    const hashes = [_]i64{ 12, 5, 11, 13, 25, 19 };
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqual(@as(usize, 3), st.misses);
    try std.testing.expectEqualSlices(i64, &.{ 11, 13, 19 }, st.ids[0..3]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, st.bytes[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, st.bytes[4..8]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, st.bytes[16..20]);
    try std.testing.expectEqual(@intFromPtr(st.bytes[8..].ptr), @intFromPtr(st.dsts[0]));
    try std.testing.expectEqual(@intFromPtr(st.bytes[20..].ptr), @intFromPtr(st.dsts[2]));
    // the misses kept: the next plan of the same rows reads nothing
    @memset(st.bytes[8..12], 7);
    @memset(st.bytes[12..16], 8);
    @memset(st.bytes[20..24], 9);
    r.keep(&st);
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqual(@as(usize, 0), st.misses);
    try std.testing.expectEqualSlices(u8, &.{ 9, 9, 9, 9 }, st.bytes[20..24]);
    // the whole tables (not owned): this node's columns only, every row a miss or a hit
    r.owned = false;
    r.caches = .{ null, null };
    r.pk[0].lo = 0;
    r.pk[0].hi = 40;
    try r.plan(&st, 0, &hashes, 2, 1);
    try std.testing.expectEqualSlices(i64, &.{ 5, 11, 25, 19 }, st.ids[0..st.misses]);
}
