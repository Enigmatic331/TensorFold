//! The family's one-hop RDMA all-gather of small fp32 slices (TP decode on GB10), host half: the Python build's
//! cuda/rdma_gather.cu v7 in Zig. The staging kernel copies this rank's slice into a slot of pinned host memory and
//! rings a doorbell; the proxy thread splits the slot over the node's RDMA devices and writes the parts, with an
//! immediate, into every peer's receive slot for this rank; a peer's proxy sees each part's receive completion and,
//! once all of a sequence's parts are in, publishes it in that peer's flag; the collecting kernel waits for the flags
//! and copies the slots out in rank order (or straight into the caller's views). The kernel pair is rdma_gather.cu's
//! and runs in graphs like any other kernel; a failed write or `abort` sets an abort word every wait also reads.
//!
//! Routes (a ring of nodes without a switch): every listed device is opened, each with its own RoCE v2 IPv4 GID (the
//! first in its table unless Settings.gid_index says), and toward each peer the devices on a subnet the peer's share
//! (TF_DS_RDMA_PREFIX bits, default 24; a port and its PCIe twin on one subnet both, in list order on either side)
//! carry the stripes. A peer no device reaches is relayed: the ranks routed to both it and this one (on four nodes in a
//! ring, the two neighbours) each write half of this rank's slot on from their own receive slot into its, as the slot's
//! owner would (the immediate names the owner), once all of it is in; a relaying rank posts its own sequence q only
//! after every relay of sequences up to q - (slots - 1) has completed, so no owner overwrites a slot the NIC is still
//! reading (the owner writes q + 1 into that slot only after collecting this rank's q). The
//! GPU half and the bytes are the same either way.
const std = @import("std");
const cuda = @import("cuda");
const fabric = @import("fabric");
const seqs = @import("rdma_seq.zig");
const IdleGate = @import("idle_gate.zig").Gate;

const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const DevicePtr = cuda.abi.DevicePtr;

pub const max_ranks = seqs.max_ranks;
pub const max_devices = seqs.max_devices;
const recv_depth = 256; // receive requests kept posted on every QP (a write with immediate consumes one)
const send_depth = 512;
const cq_depth = 4096;
const stage_threads = 512;
const collect_threads = 256;
const mtu_4096 = 5;
const port_num = 1;
const relay_wr: u64 = 1 << 63; // a relayed write's wr_id: its sequence with this bit
const wc_recv_rdma_with_imm = 129;
const host_register_portable = 0x01;
const host_register_devicemap = 0x02;

pub const Error = error{ VerbsFailed, PostFailed, BadInfo, TooLarge, Failed, NoHostRegister, NoGid } || cuda.Error || fabric.verbs.Error || std.posix.MMapError || std.Thread.SpawnError;

/// The Python build's switches, with their defaults (TF_RDMA_SLOTS, TF_RDMA_STAGE_BLOCKS, ..., TF_RDMA_PDL as served).
pub const Settings = struct {
    slots: u32 = 4,
    max_bytes: usize,
    gid_index: ?u8 = null, // null: each device's first RoCE v2 IPv4 GID
    prefix: ?u8 = null, // the subnets' prefix length that pairs devices with a peer's (null: TF_DS_RDMA_PREFIX, else 24)
    stage_blocks: u32 = 16,
    collect_blocks: u32 = 48,
    collect_grain: u32 = 4,
    pdl: bool = true,
    own_early: bool = true,
    unroll: bool = true,
    fence: bool = true,
    probing: bool = false,
    cpu: ?u16 = null, // the proxy thread's CPU (TF_DS_RDMA_CPUS), else wherever the scheduler puts it
};

/// What the other side's queue pairs toward this rank need (a device each), and where this rank's receive ring is.
pub const Info = extern struct {
    devices: u32,
    qpn: [max_devices]u32,
    psn: [max_devices]u32,
    rkey: [max_devices]u32,
    gid: [max_devices][16]u8,
    recv: u64,
};

/// The kernel pair, from the family's rdma_gather fatbin.
pub const Kernels = struct {
    stage: cuda.Function,
    collect: cuda.Function,
    collect_into: cuda.Function,

    pub fn load(m: cuda.Module) cuda.Error!Kernels {
        return .{ .stage = try m.function("dsv41_rdma_stage"), .collect = try m.function("dsv41_rdma_collect"), .collect_into = try m.function("dsv41_rdma_collect_into") };
    }
};

/// cuMemHostRegister / cuMemHostUnregister, which the runtime's driver table does not hold.
const HostReg = struct {
    lib: std.DynLib,
    register: *const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result,
    unregister: *const fn (?*anyopaque) callconv(.c) cuda.abi.Result,

    fn open() Error!HostReg {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.NoHostRegister;
        errdefer lib.close();
        return .{
            .lib = lib,
            .register = lib.lookup(*const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result, "cuMemHostRegister_v2") orelse return error.NoHostRegister,
            .unregister = lib.lookup(*const fn (?*anyopaque) callconv(.c) cuda.abi.Result, "cuMemHostUnregister") orelse return error.NoHostRegister,
        };
    }
};

extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

/// Zeroed pages registered with CUDA (mapped, portable). On GB10 the GPU reaches them over the coherent path of its own
/// memory, where cuMemHostAlloc memory goes through an uncached one; mlock keeps them resident (best effort).
const Pinned = struct {
    mem: []align(std.heap.page_size_min) u8,
    dev: DevicePtr,

    fn alloc(d: *const cuda.Driver, hr: *const HostReg, len: usize) Error!Pinned {
        const n = std.mem.alignForward(usize, len, 1 << 16);
        const mem = try std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        errdefer std.posix.munmap(mem);
        @memset(mem, 0);
        _ = mlock(mem.ptr, mem.len);
        try d.check(hr.register(mem.ptr, mem.len, host_register_portable | host_register_devicemap), "cuMemHostRegister");
        var dev: DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, mem.ptr, 0), "cuMemHostGetDevicePointer");
        return .{ .mem = mem, .dev = dev };
    }

    fn free(p: *Pinned, hr: *const HostReg) void {
        _ = hr.unregister(p.mem.ptr);
        std.posix.munmap(p.mem);
    }
};

/// One RDMA device: its context, protection domain, one completion queue for every QP, the rings' registrations and an
/// RC queue pair per peer (only those on a route leave INIT).
const Nic = struct {
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq: *vabi.Cq,
    mr_send: *vabi.Mr,
    mr_recv: *vabi.Mr,
    gid: [16]u8,
    gid_index: u8,
    mtu: c_int = mtu_4096, // the port's active MTU (ibv_query_port), capped at 4096: the QPs' path MTU
    used: bool = false, // some route runs on it (the proxy polls its queue)
    qps: [max_ranks]?*vabi.Qp = @splat(null),
    psns: [max_ranks]u32 = @splat(0),
};

/// A peer: the lanes toward it (this rank's device and the peer's device each lane pairs, the peer's key and queue
/// pair there) and its receive ring; or, with no lane, the ranks that relay to it and from it.
const Peer = struct {
    lanes: u32 = 0,
    nic: [max_devices]u8 = @splat(0), // this rank's device of each lane
    qpn: [max_devices]u32 = @splat(0),
    psn: [max_devices]u32 = @splat(0),
    rkey: [max_devices]u32 = @splat(0),
    gid: [max_devices][16]u8 = @splat(@splat(0)),
    recv: u64 = 0,
    relayers: [max_ranks]u8 = @splat(0), // relayed: the relaying ranks, in rank order (relayer k carries half k)
    n_relayers: u32 = 0,
};

/// One relay this rank runs each sequence: half `half` of `of` halves of rank `src`'s slot on to rank `dst`, on lane
/// `lane` toward it.
const Job = struct { src: u8, dst: u8, half: u8, of: u8, lane: u8 };

/// The IPv4 address in a RoCE v2 IPv4 GID (::ffff:a.b.c.d), else null.
fn gidIp(g: [16]u8) ?u32 {
    for (g[0..10]) |b| if (b != 0) return null;
    if (g[10] != 0xff or g[11] != 0xff) return null;
    return std.mem.readInt(u32, g[12..16], .big);
}

/// Whether sysfs calls device `name`'s GID `index` RoCE v2 (/sys/class/infiniband/NAME/ports/1/gid_attrs/types/INDEX).
fn sysfsV2(name: []const u8, index: u32) bool {
    var path_buf: [256]u8 = undefined;
    const path = std.mem.printSentinel(&path_buf, "/sys/class/infiniband/{s}/ports/{d}/gid_attrs/types/{d}", .{ name, port_num, index }, 0) catch return false;
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, path, .{ .ACCMODE = .RDONLY }, 0) catch return false;
    defer _ = std.posix.system.close(fd);
    var buf: [64]u8 = undefined;
    const n = std.posix.read(fd, &buf) catch return false;
    return std.mem.indexOf(u8, buf[0..n], "v2") != null;
}

fn sameNet(a: u32, b: u32, prefix: u8) bool {
    if (prefix == 0) return true;
    if (prefix >= 32) return a == b;
    const sh: u5 = @intCast(32 - @as(u32, prefix));
    return (a >> sh) == (b >> sh);
}

/// The lanes between two ranks' devices: in each subnet, the k-th device of one with the k-th of the other (in list
/// order), so both ends pair the same devices. `out[k]` = (index in `mine`, index in `theirs`); null: no IPv4 GIDs, so
/// device d pairs with device d (the switch's layout).
fn pairLanes(mine: []const [16]u8, theirs: []const [16]u8, prefix: u8, out: *[max_devices][2]u8) ?u32 {
    for (mine) |g| if (gidIp(g) == null) return null;
    for (theirs) |g| if (gidIp(g) == null) return null;
    var n: u32 = 0;
    var taken: [max_devices]bool = @splat(false);
    for (mine, 0..) |g, i| {
        const ip = gidIp(g).?;
        // this device's place among this rank's devices on its subnet
        var k: usize = 0;
        for (mine[0..i]) |h| if (sameNet(gidIp(h).?, ip, prefix)) {
            k += 1;
        };
        var seen: usize = 0;
        for (theirs, 0..) |t, j| {
            if (!sameNet(gidIp(t).?, ip, prefix)) continue;
            if (seen == k and !taken[j]) {
                taken[j] = true;
                out[n] = .{ @intCast(i), @intCast(j) };
                n += 1;
                break;
            }
            seen += 1;
        }
    }
    return n;
}

fn linked(a: Info, b: Info, prefix: u8) bool {
    var out: [max_devices][2]u8 = undefined;
    const n = pairLanes(a.gid[0..a.devices], b.gid[0..b.devices], prefix, &out) orelse return a.devices == b.devices;
    return n > 0;
}

pub const Ring = struct {
    d: *const cuda.Driver,
    verbs: Verbs,
    v: *const Verbs = undefined,
    k: Kernels,
    s: Settings,
    rank: u32,
    world: u32,
    hr: HostReg,
    send: Pinned,
    recv: Pinned,
    meta: Pinned,
    scalars: cuda.DeviceBuffer, // seq (u64), staged (u32), probe (8 x u64)
    nics: [max_devices]Nic = undefined,
    nd: u32 = 0,
    peers: [max_ranks]Peer = @splat(.{}),
    jobs: [2 * max_ranks]Job = undefined, // the relays this rank runs (on a ring of four: two)
    n_jobs: u32 = 0,
    proxy: ?std.Thread = null,
    idle_gate: IdleGate = .{},
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    why: [128]u8 = @splat(0),

    // the metadata block: flags [slots][world], doorbell, sizes [slots], sent [slots], abort word
    fn flags(r: *const Ring) [*]u64 {
        return @ptrCast(@alignCast(r.meta.mem.ptr));
    }
    fn doorbell(r: *const Ring) *u64 {
        return &r.flags()[r.s.slots * r.world];
    }
    fn sizes(r: *const Ring) [*]u64 {
        return r.flags() + r.s.slots * r.world + 1;
    }
    fn sent(r: *const Ring) [*]u64 {
        return r.sizes() + r.s.slots;
    }
    fn abortWord(r: *const Ring) *u64 {
        return &r.sent()[r.s.slots];
    }
    fn metaDev(r: *const Ring, p: anytype) DevicePtr {
        return r.meta.dev + (@intFromPtr(p) - @intFromPtr(r.meta.mem.ptr));
    }

    /// Rings, metadata and queue pairs (in INIT, receives posted) on `devices`; `connect` with every rank's `info` next.
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, devices: []const []const u8, rank: u32, world: u32, s: Settings) Error!*Ring {
        if (world < 2 or world > max_ranks or rank >= world or devices.len == 0 or devices.len > max_devices) return error.BadInfo;
        if (s.slots < 2 or s.slots > seqs.max_slots or s.max_bytes == 0 or s.max_bytes % 64 != 0) return error.BadInfo;
        const r = try gpa.create(Ring);
        errdefer gpa.destroy(r);
        var verbs = try Verbs.open(null);
        errdefer verbs.close();
        var hr = try HostReg.open();
        errdefer hr.lib.close();
        var send = try Pinned.alloc(d, &hr, s.slots * s.max_bytes);
        errdefer send.free(&hr);
        var recv = try Pinned.alloc(d, &hr, s.slots * world * s.max_bytes);
        errdefer recv.free(&hr);
        var meta = try Pinned.alloc(d, &hr, (s.slots * world + 1 + 2 * s.slots + 1) * 8);
        errdefer meta.free(&hr);
        var scalars = try cuda.DeviceBuffer.alloc(d, 80);
        errdefer scalars.free();
        try scalars.fill8(0, null);
        r.* = .{ .d = d, .verbs = verbs, .k = k, .s = s, .rank = rank, .world = world, .hr = hr, .send = send, .recv = recv, .meta = meta, .scalars = scalars };
        r.v = &r.verbs;
        errdefer for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        for (devices) |name| {
            r.nics[r.nd] = try r.openNic(name);
            r.nd += 1;
        }
        return r;
    }

    fn openNic(r: *Ring, name: []const u8) Error!Nic {
        const api = r.v.api;
        const ctx = try r.v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const cq = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        const access = vabi.access_local_write | vabi.access_remote_write;
        const mr_send = api.ibv_reg_mr(pd, r.send.mem.ptr, r.send.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_send);
        const mr_recv = api.ibv_reg_mr(pd, r.recv.mem.ptr, r.recv.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_recv);
        const gi = try r.pickGid(ctx, name);
        const gid = try r.v.gid(ctx, port_num, gi);
        // Path MTU from the port's active MTU, capped at 4096 (enum 5), instead of a fixed 4096: on a fabric whose
        // Ethernet MTU is 1500 the RoCE active MTU is 1024 and a 4096 path MTU loses every larger packet (the idea of
        // bertholomus/TensorFold#12, local endpoint only; #12 also exchanges and takes the smaller of both ends).
        const pa = try r.v.port(ctx, port_num);
        const mtu: c_int = if (pa.active_mtu >= 1 and pa.active_mtu < mtu_4096) pa.active_mtu else mtu_4096;
        var nic: Nic = .{ .ctx = ctx, .pd = pd, .cq = cq, .mr_send = mr_send, .mr_recv = mr_recv, .gid = gid.raw, .gid_index = gi, .mtu = mtu };
        errdefer for (nic.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            var init: vabi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = recv_depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = vabi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            nic.qps[p] = qp;
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = port_num, .qp_access_flags = vabi.access_remote_write };
            const m = vabi.mask;
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            for (0..recv_depth) |_| try postRecv(ctx, qp);
            nic.psns[p] = (qp.qp_num *% 2654435761) & 0xFF_FFFF;
        }
        return nic;
    }

    /// Settings.gid_index, else the device's first RoCE v2 GID that holds an IPv4 address (its type from
    /// ibv_query_gid_ex, else from sysfs; neither: error.NoGid, and Settings.gid_index / TF_DS_RDMA_GID must say).
    fn pickGid(r: *Ring, ctx: *vabi.Context, name: []const u8) Error!u8 {
        if (r.s.gid_index) |g| return g;
        const p = try r.v.port(ctx, port_num);
        var i: u32 = 0;
        while (i < @as(u32, @intCast(@max(0, @min(p.gid_tbl_len, 256))))) : (i += 1) {
            const g = r.v.gid(ctx, port_num, i) catch continue;
            if (gidIp(g.raw) == null) continue;
            const v2 = if (r.v.gidType(ctx, port_num, i)) |t| t == 2 else sysfsV2(name, i);
            if (v2) return @intCast(i);
        }
        return error.NoGid;
    }

    fn closeNic(r: *Ring, n: *Nic) void {
        const api = r.v.api;
        for (n.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        _ = api.ibv_dereg_mr(n.mr_recv);
        _ = api.ibv_dereg_mr(n.mr_send);
        _ = api.ibv_destroy_cq(n.cq);
        _ = api.ibv_dealloc_pd(n.pd);
        _ = api.ibv_close_device(n.ctx);
    }

    pub fn destroy(r: *Ring, gpa: std.mem.Allocator) void {
        r.stop();
        for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        r.scalars.free();
        r.meta.free(&r.hr);
        r.recv.free(&r.hr);
        r.send.free(&r.hr);
        r.hr.lib.close();
        r.verbs.close();
        gpa.destroy(r);
    }

    /// What peer `p`'s queue pairs toward this rank need.
    pub fn info(r: *const Ring, p: u32) Info {
        var out: Info = std.mem.zeroes(Info);
        out.devices = r.nd;
        out.recv = @intFromPtr(r.recv.mem.ptr);
        for (r.nics[0..r.nd], 0..) |n, d| {
            out.qpn[d] = if (n.qps[p]) |qp| qp.qp_num else 0;
            out.psn[d] = n.psns[p];
            out.rkey[d] = n.mr_recv.rkey;
            out.gid[d] = n.gid;
        }
        return out;
    }

    /// `remote[p]`: what rank p's `info(rank)` gave. Routes each peer (pairLanes; a peer with no lane relayed through the
    /// ranks routed to both), finds this rank's relays, and moves every routed QP to RTR and RTS (the Python build's
    /// attributes).
    pub fn connect(r: *Ring, remote: []const Info) Error!void {
        if (remote.len != r.world) return error.BadInfo;
        const api = r.v.api;
        const m = vabi.mask;
        const prefix: u8 = r.s.prefix orelse blk: {
            const e = std.c.getenv("TF_DS_RDMA_PREFIX") orelse break :blk 24;
            break :blk std.fmt.parseInt(u8, std.mem.span(e), 10) catch 24;
        };
        // every rank's devices (this rank's own from its info toward any peer): who reaches whom
        var all: [max_ranks]Info = undefined;
        for (0..r.world) |p| all[p] = if (p == r.rank) r.info(@intCast((r.rank + 1) % r.world)) else remote[p];
        for (remote, 0..) |ri, p| if (p != r.rank and (ri.devices == 0 or ri.devices > max_devices)) return error.BadInfo;
        var mine_g: [max_devices][16]u8 = undefined;
        for (r.nics[0..r.nd], 0..) |n, d| mine_g[d] = n.gid;
        for (remote, 0..) |ri, p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            peer.* = .{ .recv = ri.recv };
            var pairs: [max_devices][2]u8 = undefined;
            const n = pairLanes(mine_g[0..r.nd], ri.gid[0..ri.devices], prefix, &pairs) orelse blk: {
                // no IPv4 GIDs: device d with device d, every device
                if (ri.devices != r.nd) return error.BadInfo;
                for (0..r.nd) |d| pairs[d] = .{ @intCast(d), @intCast(d) };
                break :blk r.nd;
            };
            peer.lanes = n;
            for (pairs[0..n], 0..) |pr, k| {
                peer.nic[k] = pr[0];
                peer.qpn[k] = ri.qpn[pr[1]];
                peer.psn[k] = ri.psn[pr[1]];
                peer.rkey[k] = ri.rkey[pr[1]];
                peer.gid[k] = ri.gid[pr[1]];
            }
            if (n == 0) {
                // relayed: every rank linked to both, in rank order
                for (0..r.world) |q| {
                    if (q == p or q == r.rank) continue;
                    if (linked(all[q], all[p], prefix) and linked(all[q], all[r.rank], prefix)) {
                        peer.relayers[peer.n_relayers] = @intCast(q);
                        peer.n_relayers += 1;
                    }
                }
                if (peer.n_relayers == 0 or peer.n_relayers > max_devices) return error.BadInfo;
            }
        }
        // this rank's relays: from a peer to a peer it routes to both of, when those two have no lane between them
        r.n_jobs = 0;
        for (0..r.world) |src| for (0..r.world) |dst| {
            if (src == dst or src == r.rank or dst == r.rank) continue;
            if (r.peers[src].lanes == 0 or r.peers[dst].lanes == 0 or linked(all[src], all[dst], prefix)) continue;
            var k: u8 = 0;
            var of: u8 = 0;
            for (0..r.world) |q| {
                if (q == src or q == dst) continue;
                const both = if (q == r.rank) true else linked(all[q], all[src], prefix) and linked(all[q], all[dst], prefix);
                if (!both) continue;
                if (q == r.rank) k = of;
                of += 1;
            }
            if (r.n_jobs == r.jobs.len) return error.BadInfo;
            r.jobs[r.n_jobs] = .{ .src = @intCast(src), .dst = @intCast(dst), .half = k, .of = of, .lane = @intCast(k % r.peers[dst].lanes) };
            r.n_jobs += 1;
        };
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            for (0..peer.lanes) |k| {
                const n = &r.nics[peer.nic[k]];
                n.used = true;
                const qp = n.qps[p].?;
                var a: vabi.QpAttr = .{ .qp_state = vabi.qps_rtr, .path_mtu = n.mtu, .dest_qp_num = peer.qpn[k], .rq_psn = peer.psn[k], .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
                a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid[k] }, .sgid_index = n.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = port_num };
                if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
                a = .{ .qp_state = vabi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = n.psns[p], .max_rd_atomic = 1 };
                if (api.ibv_modify_qp(qp, &a, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
            }
        }
    }

    /// The routes connect found, for the startup line: per peer its lanes (0: relayed) and this rank's relay count.
    pub fn routes(r: *const Ring) struct { lanes: [max_ranks]u32, relays: u32 } {
        var out: [max_ranks]u32 = @splat(0);
        for (0..r.world) |p| out[p] = r.peers[p].lanes;
        return .{ .lanes = out, .relays = r.n_jobs };
    }

    pub fn start(r: *Ring) Error!void {
        if (r.running.swap(true, .acq_rel)) return;
        r.proxy = try std.Thread.spawn(.{}, proxyLoop, .{r});
    }

    pub fn stop(r: *Ring) void {
        if (r.running.swap(false, .acq_rel)) {
            r.idle_gate.wake(); // a parked proxy must wake before join
            if (r.proxy) |t| t.join();
        }
        r.proxy = null;
    }

    /// Only the CPU control protocol calls these at globally quiescent boundaries.
    pub fn park(r: *Ring) void { r.idle_gate.park(); }
    pub fn wake(r: *Ring) void { r.idle_gate.wake(); }

    /// Every wait on this rank gives up from now on (a watchdog lost a peer); `failure` says why.
    pub fn abort(r: *Ring, why: []const u8) void {
        r.fail(why);
    }

    pub fn failure(r: *const Ring) ?[]const u8 {
        if (!r.failed.load(.acquire)) return null;
        return std.mem.sliceTo(&r.why, 0);
    }

    fn fail(r: *Ring, why: []const u8) void {
        if (!r.failed.swap(true, .acq_rel)) {
            const n = @min(why.len, r.why.len - 1);
            @memcpy(r.why[0..n], why[0..n]);
        }
        @atomicStore(u64, r.abortWord(), 1, .release);
    }

    fn proxyLoop(r: *Ring) void {
        if (r.s.cpu) |c| {
            var set: std.os.linux.cpu_set_t = @splat(0);
            const bits = @bitSizeOf(usize);
            set[c / bits] |= @as(usize, 1) << @intCast(c % bits);
            std.os.linux.sched_setaffinity(0, &set) catch {};
        }
        var need: u32 = 0;
        var arrivals: seqs.Arrivals = .{};
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            need += r.peers[p].lanes;
            arrivals.lanes[p] = if (r.peers[p].lanes > 0) r.peers[p].lanes else r.peers[p].n_relayers;
        }
        var sends: seqs.Sends = .{ .need = need, .slots = r.s.slots };
        var relays = seqs.Relays.init(r.n_jobs, r.s.slots);
        // the bytes of each peer's sequence in (by slot), for the halves a relay sends on
        var got_bytes: [max_ranks][seqs.max_slots]u64 = @splat(@splat(0));
        var posted: u64 = 0;
        var wc: [64]vabi.Wc = undefined;
        while (r.running.load(.monotonic)) {
            r.idle_gate.wait() catch {
                r.fail("RDMA idle futex wait failed");
                return;
            };
            if (!r.running.load(.acquire)) break;
            const ring = @atomicLoad(u64, r.doorbell(), .acquire);
            // a relaying rank's own sequence q goes out once its relays of every sequence up to q - (slots - 1) are
            // done: an owner sends q + 1 (into the slot of q + 1 - slots) only after collecting this rank's q
            while (posted < ring and (r.n_jobs == 0 or relays.upto + r.s.slots >= posted + 2)) {
                const q = posted + 1;
                const slot: usize = @intCast(q % r.s.slots);
                const n = @atomicLoad(u64, &r.sizes()[slot], .acquire);
                for (0..r.world) |p| {
                    if (p == r.rank) continue;
                    const peer = &r.peers[p];
                    for (0..peer.lanes) |k| {
                        const nic = &r.nics[peer.nic[k]];
                        const pt = seqs.part(n, peer.lanes, @intCast(k));
                        const off: usize = @intCast(pt.off);
                        const len: usize = @intCast(pt.len);
                        const local = r.send.mem[slot * r.s.max_bytes + off ..][0..len];
                        const at: u64 = @intCast((slot * r.world + r.rank) * r.s.max_bytes + off);
                        postWrite(nic.ctx, nic.qps[p].?, local, nic.mr_send.lkey, peer.recv + at, peer.rkey[k], q, seqs.immediate(q, null)) catch {
                            r.fail("ibv_post_send failed");
                            return;
                        };
                    }
                }
                posted = q;
            }
            for (r.nics[0..r.nd], 0..) |nic, d| {
                if (!nic.used) continue;
                const k = nic.ctx.ops.poll_cq(nic.cq, wc.len, &wc);
                if (k < 0) {
                    r.fail("ibv_poll_cq failed");
                    return;
                }
                for (wc[0..@intCast(k)]) |w| {
                    if (w.status != vabi.wc_success) {
                        r.fail(r.v.statusText(w.status));
                        return;
                    }
                    if (w.opcode == wc_recv_rdma_with_imm) {
                        const from = r.peerOf(nic, w.qp_num) orelse {
                            r.fail("receive completion on an unknown QP");
                            return;
                        };
                        const im = seqs.decode(w.imm_data);
                        // a relayed part: its owner's, on the lane of the relaying rank `from`
                        const p: usize = if (im.relayed_from) |s| s else from;
                        if (p >= r.world or p == r.rank) {
                            r.fail("receive completion for a bad rank");
                            return;
                        }
                        const lane: usize = if (im.relayed_from != null) r.relayerLane(p, from) orelse {
                            r.fail("relayed part from a rank that does not relay it");
                            return;
                        } else r.laneOf(from, d) orelse {
                            r.fail("receive completion on a device off the peer's route");
                            return;
                        };
                        const got = arrivals.arrive(p, lane, im.low);
                        if (im.relayed_from == null) got_bytes[p][@intCast(got.seq % r.s.slots)] += w.byte_len;
                        var q = got.from;
                        while (q <= got.to) : (q += 1) {
                            if (im.relayed_from == null and r.n_jobs > 0) {
                                const bytes = &got_bytes[p][@intCast(q % r.s.slots)];
                                r.relay(p, q, bytes.*) catch {
                                    r.fail("ibv_post_send (relay) failed");
                                    return;
                                };
                                bytes.* = 0;
                            } else if (im.relayed_from == null) got_bytes[p][@intCast(q % r.s.slots)] = 0;
                            @atomicStore(u64, r.flagAt(q, p), q, .release);
                        }
                        postRecv(nic.ctx, nic.qps[from].?) catch {
                            r.fail("ibv_post_recv failed");
                            return;
                        };
                    } else if (w.wr_id & relay_wr != 0) {
                        relays.complete(w.wr_id & ~relay_wr);
                    } else if (sends.complete(w.wr_id)) |q| {
                        @atomicStore(u64, &r.sent()[@as(usize, @intCast(q % r.s.slots))], q, .release);
                    }
                }
            }
        }
    }

    /// Rank `src`'s sequence q (`n` bytes, all in this rank's receive slot) on to the ranks this one relays it to.
    fn relay(r: *Ring, src: usize, q: u64, n: u64) Error!void {
        const slot: usize = @intCast(q % r.s.slots);
        for (r.jobs[0..r.n_jobs]) |j| {
            if (j.src != src) continue;
            const peer = &r.peers[j.dst];
            const nic = &r.nics[peer.nic[j.lane]];
            const pt = seqs.part(n, j.of, j.half);
            const at: usize = (slot * r.world + src) * r.s.max_bytes + @as(usize, @intCast(pt.off));
            const local = r.recv.mem[at..][0..@intCast(pt.len)];
            try postWrite(nic.ctx, nic.qps[j.dst].?, local, nic.mr_recv.lkey, peer.recv + at, peer.rkey[j.lane], q | relay_wr, seqs.immediate(q, @intCast(src)));
        }
    }

    /// The lane of peer p that runs on this rank's device d.
    fn laneOf(r: *const Ring, p: usize, d: usize) ?usize {
        const peer = &r.peers[p];
        for (0..peer.lanes) |k| if (peer.nic[k] == d) return k;
        return null;
    }

    /// The lane of relayed peer p that relaying rank `via` brings.
    fn relayerLane(r: *const Ring, p: usize, via: usize) ?usize {
        const peer = &r.peers[p];
        for (0..peer.n_relayers) |k| if (peer.relayers[k] == via) return k;
        return null;
    }

    fn flagAt(r: *const Ring, q: u64, p: usize) *u64 {
        const slot: usize = @intCast(q % r.s.slots);
        return &r.flags()[slot * r.world + p];
    }

    fn peerOf(r: *const Ring, nic: Nic, qpn: u32) ?usize {
        for (0..r.world) |p| if (nic.qps[p]) |qp| if (qp.qp_num == qpn) return p;
        return null;
    }

    fn scalar(r: *const Ring, offset: usize) DevicePtr {
        return r.scalars.ptr + offset;
    }

    /// recv [world x n4 float4] <- every rank's send [n4 float4] (fp32, 16-byte aligned), on `stream`.
    pub fn gather(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, recv: DevicePtr) Error!void {
        if (r.failure() != null) return error.Failed;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        const own: DevicePtr = if (r.s.own_early) recv + @as(u64, r.rank) * n4 * 16 else 0;
        try r.stage(stream, send, n4, own, n4, n4);
        var a: cuda.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.recv.dev);
        a.add(r.metaDev(r.flags()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(r.s.own_early)));
        a.add(@as(c_int, @intFromBool(r.s.unroll)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const per = collect_threads * r.s.collect_grain;
        const blocks = std.math.clamp((@as(u64, n4) * r.world + per - 1) / per, 1, r.s.collect_blocks);
        try cuda.launch.launch(r.k.collect, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// A rank's slice in `dsts` (its rows `row4` float4 wide, `stride4` apart; ptr 0: not copied out).
    pub const Dst = struct { ptr: DevicePtr = 0, row4: u32 = 0, stride4: u64 = 0 };

    /// Every rank's [rows, w_p] fp32 slice straight into dsts[p]; `send` is this rank's slice, contiguous. Every rank
    /// names the same widths, so each peer's ring slot holds rows x w_p floats.
    pub fn gatherInto(r: *Ring, stream: cuda.Stream, send: DevicePtr, rows: u32, own_row4: u32, dsts: []const Dst) Error!void {
        if (r.failure() != null) return error.Failed;
        if (dsts.len != r.world) return error.BadInfo;
        const n4 = rows * own_row4;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        const D = extern struct { ptr: [8]u64, stride4: [8]i64, row4: [8]c_int };
        var dd: D = std.mem.zeroes(D);
        var copy4: u64 = 0;
        for (dsts, 0..) |t, p| {
            if (t.ptr == 0) continue;
            if (@as(usize, rows) * t.row4 * 16 > r.s.max_bytes) return error.TooLarge;
            dd.ptr[p] = t.ptr;
            dd.row4[p] = @intCast(t.row4);
            dd.stride4[p] = @intCast(t.stride4);
            if (p != r.rank) copy4 += @as(u64, rows) * t.row4;
        }
        const mine = dsts[r.rank];
        try r.stage(stream, send, n4, mine.ptr, if (mine.ptr != 0) mine.row4 else n4, if (mine.ptr != 0) mine.stride4 else n4);
        var a: cuda.Args = .{};
        a.add(dd);
        a.add(@as(c_int, @intCast(rows)));
        a.add(r.recv.dev);
        a.add(r.metaDev(r.flags()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const per = collect_threads * r.s.collect_grain;
        const blocks = std.math.clamp((copy4 + per - 1) / per, 1, r.s.collect_blocks);
        try cuda.launch.launch(r.k.collect_into, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    fn probe(r: *const Ring) DevicePtr {
        return if (r.s.probing) r.scalar(16) else 0;
    }

    fn stage(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, own: DevicePtr, own_row4: u32, own_stride4: u64) Error!void {
        var a: cuda.Args = .{};
        a.add(send);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.send.dev);
        a.add(r.metaDev(r.sizes()));
        a.add(r.metaDev(r.doorbell()));
        a.add(r.metaDev(r.sent()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(r.scalar(8));
        a.add(@as(c_int, @intFromBool(r.s.fence)));
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride4)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const blocks = std.math.clamp((@as(u64, n4) + 1023) / 1024, 1, r.s.stage_blocks);
        try cuda.launch.launch(r.k.stage, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = stage_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }

    /// (stage, doorbell to every flag, copy-out) mean ns over the probed gathers and their count; resets the sums.
    pub fn probes(r: *Ring) Error![4]f64 {
        var host: [8]u64 = undefined;
        try r.scalars.download(16, std.mem.sliceAsBytes(&host));
        try r.d.check(r.d.api.cuMemsetD8_v2(r.scalar(16), 0, 64), "cuMemsetD8");
        const n: f64 = if (host[3] > 0) @floatFromInt(host[3]) else 1;
        return .{ @as(f64, @floatFromInt(host[0])) / n, @as(f64, @floatFromInt(host[1])) / n, @as(f64, @floatFromInt(host[2])) / n, @floatFromInt(host[3]) };
    }
};

fn postRecv(ctx: *vabi.Context, qp: *vabi.Qp) Error!void {
    var wr: vabi.RecvWr = .{};
    var bad: ?*vabi.RecvWr = null;
    if (ctx.ops.post_recv(qp, &wr, &bad) != 0) return error.PostFailed;
}

fn postWrite(ctx: *vabi.Context, qp: *vabi.Qp, local: []const u8, lkey: u32, remote: u64, rkey: u32, wr_id: u64, imm: u32) Error!void {
    var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = lkey };
    var wr: vabi.SendWr = .{ .wr_id = wr_id, .sg_list = @ptrCast(&sge), .num_sge = @intFromBool(local.len > 0), .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = imm, .remote_addr = remote, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
}

test "every declaration compiles (the GPU and verbs paths run only on the nodes)" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Ring);
    std.testing.refAllDecls(Kernels);
}

test "the info a peer reads is a plain extern struct (it crosses the link as bytes)" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Info));
    try std.testing.expectEqual(@as(usize, 120), @offsetOf(Info, "recv"));
}

test "lanes pair each subnet's devices in list order; a peer across the ring has none" {
    const g = struct {
        fn of(a: u8, b: u8, c: u8, d: u8) [16]u8 {
            return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, a, b, c, d };
        }
    }.of;
    // spark1: f0 toward rank 3 (10.0.41.x), f1 toward rank 1 (10.0.22.x), and their PCIe twins (.10x)
    const s1 = [_][16]u8{ g(10, 0, 41, 1), g(10, 0, 22, 1), g(10, 0, 41, 101), g(10, 0, 22, 101) };
    const s2 = [_][16]u8{ g(10, 0, 33, 2), g(10, 0, 22, 2), g(10, 0, 33, 102), g(10, 0, 22, 102) };
    const s3 = [_][16]u8{ g(10, 0, 33, 3), g(10, 0, 34, 3), g(10, 0, 33, 103), g(10, 0, 34, 103) };
    var out: [max_devices][2]u8 = undefined;
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s1, &s2, 24, &out));
    try std.testing.expectEqual([2]u8{ 1, 1 }, out[0]);
    try std.testing.expectEqual([2]u8{ 3, 3 }, out[1]);
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s2, &s1, 24, &out));
    try std.testing.expectEqual([2]u8{ 1, 1 }, out[0]);
    try std.testing.expectEqual(@as(?u32, 0), pairLanes(&s1, &s3, 24, &out));
    // /26 tells the twins apart: .1 with .2, .101 with .102
    try std.testing.expectEqual(@as(?u32, 2), pairLanes(&s1, &s2, 26, &out));
    try std.testing.expect(sameNet(0x0a001601, 0x0a001602, 26));
    try std.testing.expect(!sameNet(0x0a001601, 0x0a001666, 26));
    // a link-local GID: no routes by subnet
    const ll = [_][16]u8{.{ 0xfe, 0x80, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8 }};
    try std.testing.expectEqual(@as(?u32, null), pairLanes(&ll, &s2, 24, &out));
}
