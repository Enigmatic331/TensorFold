//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    // Two-rank links to the ring neighbours (0: rank - 1, 1: rank + 1, mod world), each a communicator of its own made
    // with its own NCCL graph file, so that its channels run over the cable to that neighbour (initLinks); null: none.
    links: [2]?nccl.Comm = .{ null, null },

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        for (c.links) |l| if (l) |x| {
            _ = c.lib.api.ncclCommDestroy(x);
        };
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// The two-rank links (links): `ids[r]` is the id rank r made for its link to rank r + 1 (every rank's, from an
    /// all-gather); `graphs` the graph files of this rank's link to its previous and to its next neighbour (NCCL reads
    /// NCCL_GRAPH_FILE as each communicator is made). Even ranks join their next link first, odd ones their previous:
    /// pairs of neighbours meet, and no rank waits on a ring of blocked ones.
    pub fn initLinks(c: *Comm, ids: []const nccl.UniqueId, graphs: [2]?[:0]const u8) !void {
        const w = c.world;
        const r = c.rank;
        if (w % 2 != 0 or ids.len != w) return error.LinksNeedEvenRing;
        const order: [2]usize = if (r % 2 == 0) .{ 1, 0 } else .{ 0, 1 };
        const saved = std.c.getenv("NCCL_GRAPH_FILE");
        defer {
            if (saved) |v| _ = setenv("NCCL_GRAPH_FILE", v, 1) else _ = unsetenv("NCCL_GRAPH_FILE");
        }
        for (order) |which| {
            // the previous link (r - 1, r): its id from rank r - 1, this rank its rank 1; the next (r, r + 1): rank 0
            const id = if (which == 0) ids[(r + w - 1) % w] else ids[r];
            if (graphs[which]) |g| _ = setenv("NCCL_GRAPH_FILE", g.ptr, 1) else _ = unsetenv("NCCL_GRAPH_FILE");
            var comm: nccl.Comm = null;
            try c.lib.check(c.lib.api.ncclCommInitRank(&comm, 2, id, if (which == 0) 1 else 0), "ncclCommInitRank (ring link)");
            c.links[which] = comm;
        }
    }

    /// The link to world rank `peer` (a ring neighbour) and this rank's place in it, else null.
    pub fn linkTo(c: *const Comm, peer: u32) ?struct { comm: nccl.Comm, me: u32 } {
        const w = c.world;
        if (c.links[1]) |l| if (peer == (c.rank + 1) % w) return .{ .comm = l, .me = 0 };
        if (c.links[0]) |l| if (peer == (c.rank + w - 1) % w) return .{ .comm = l, .me = 1 };
        return null;
    }

    /// `count` elements of `dtype` from both ranks of a link into `recv`, link rank 0's then 1's.
    pub fn linkGather(c: Comm, link: nccl.Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, link, stream.handle), "ncclAllGather (ring link)");
    }

    /// The two-rank link's sum of `count` fp32 elements, in place (NCCL leaves both ranks the same result).
    pub fn linkSum(c: Comm, link: nccl.Comm, buf: u64, count: usize, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllReduce(buf, buf, count, .f32, .sum, link, stream.handle), "ncclAllReduce (ring link, sum)");
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// `count` bytes every rank holds in `buf`, each made the largest of the ranks' (NCCL max over u8): exact for Engram's
    /// owned records, where one rank holds a slot's bytes and the others zeros, whatever the order.
    pub fn maxBytes(c: Comm, buf: u64, count: usize, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllReduce(buf, buf, count, .u8, .max, c.comm, stream.handle), "ncclAllReduce (max)");
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }
};
