//! Engram's packed shards: one file a layer and node with rows [lo, hi) of the layer's table, each row's FP8 values then
//! its E8M0 scales (one record), after a 4 KiB header, so a row is one read. Node g of n holds rows
//! [rows * g / n, rows * (g + 1) / n); n = 1 is the whole table. The records are the tables' own bytes.
const std = @import("std");
const dio = @import("core").direct_io;
const engram_io = @import("engram_io.zig");

pub const magic = "TFENGRM1".*;
pub const version: u32 = 1;
pub const header_bytes = 4096;

/// A shard's layer, its table's rows and row widths, and its own rows [lo, hi).
pub const Header = struct { layer: u32, rows: u64, lo: u64, hi: u64, row_w: u32, row_s: u32 };

pub fn encode(h: Header, out: *[header_bytes]u8) void {
    @memset(out, 0);
    @memcpy(out[0..8], &magic);
    std.mem.writeInt(u32, out[8..12], version, .little);
    std.mem.writeInt(u32, out[12..16], h.layer, .little);
    std.mem.writeInt(u64, out[16..24], h.rows, .little);
    std.mem.writeInt(u64, out[24..32], h.lo, .little);
    std.mem.writeInt(u64, out[32..40], h.hi, .little);
    std.mem.writeInt(u32, out[40..44], h.row_w, .little);
    std.mem.writeInt(u32, out[44..48], h.row_s, .little);
}

pub fn decode(b: *const [header_bytes]u8) !Header {
    if (!std.mem.eql(u8, b[0..8], &magic)) return error.NotAPackedShard;
    if (std.mem.readInt(u32, b[8..12], .little) != version) return error.PackedShardVersion;
    const h: Header = .{
        .layer = std.mem.readInt(u32, b[12..16], .little),
        .rows = std.mem.readInt(u64, b[16..24], .little),
        .lo = std.mem.readInt(u64, b[24..32], .little),
        .hi = std.mem.readInt(u64, b[32..40], .little),
        .row_w = std.mem.readInt(u32, b[40..44], .little),
        .row_s = std.mem.readInt(u32, b[44..48], .little),
    };
    if (h.lo > h.hi or h.hi > h.rows) return error.BadPackedShard;
    return h;
}

/// Node `node` of `nodes`'s rows of a table of `rows`.
pub fn range(rows: u64, node: u32, nodes: u32) [2]u64 {
    return .{ rows * node / nodes, rows * (node + 1) / nodes };
}

/// Layer `layer`'s shard file for node `node` of `nodes`.
pub fn name(buf: []u8, layer: u32, node: u32, nodes: u32) ![]u8 {
    return std.fmt.bufPrint(buf, "engram-l{d}-n{d}of{d}.tfe", .{ layer, node, nodes });
}

/// A node's shards, one a layer, open for its readers.
pub const Shards = struct {
    gpa: std.mem.Allocator,
    fds: std.ArrayList(std.c.fd_t) = .empty,
    layers: std.AutoHashMapUnmanaged(u32, engram_io.Packed) = .empty,

    /// Node `node` of `nodes`'s shard of every layer of `tables` from `dir`.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, tables: *const engram_io.Tables, node: u32, nodes: u32) !Shards {
        var s: Shards = .{ .gpa = gpa };
        errdefer s.close();
        var it = tables.layers.iterator();
        while (it.next()) |kv| {
            const t = kv.value_ptr.*;
            const pk = try openOne(gpa, &s.fds, dir, kv.key_ptr.*, t.rows, t.row_w, t.row_s, node, nodes);
            try s.layers.put(gpa, kv.key_ptr.*, pk);
        }
        return s;
    }

    pub fn close(s: *Shards) void {
        for (s.fds.items) |fd| _ = std.c.close(fd);
        s.fds.deinit(s.gpa);
        s.layers.deinit(s.gpa);
        s.* = undefined;
    }
};

/// Layer `layer`'s shard of node `node` of `nodes` in `dir`, checked against its table (rows, row widths) and the
/// node's range, and whole (its last record readable); its descriptors go into `fds`.
pub fn openOne(gpa: std.mem.Allocator, fds: *std.ArrayList(std.c.fd_t), dir: []const u8, layer: u32, rows: u64, row_w: usize, row_s: usize, node: u32, nodes: u32) !engram_io.Packed {
    var nb: [64]u8 = undefined;
    const path = try std.fs.path.joinZ(gpa, &.{ dir, try name(&nb, layer, node, nodes) });
    defer gpa.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    fds.append(gpa, fd) catch |err| {
        _ = std.c.close(fd);
        return err;
    };
    var hb: [header_bytes]u8 = undefined;
    try engram_io.preadAll(fd, &hb, 0);
    const h = try decode(&hb);
    const want = range(rows, node, nodes);
    if (h.layer != layer or h.rows != rows or h.lo != want[0] or h.hi != want[1] or h.row_w != row_w or h.row_s != row_s) {
        std.log.err("dsv41: {s} holds layer {d} rows [{d}, {d}) of {d}; node {d} of {d} reads [{d}, {d}) of {d}", .{ path, h.layer, h.lo, h.hi, h.rows, node, nodes, want[0], want[1], rows });
        return error.PackedShardMismatch;
    }
    const rec = row_w + row_s;
    if (h.hi > h.lo) {
        var last: [1]u8 = undefined;
        try engram_io.preadAll(fd, &last, header_bytes + (h.hi - h.lo) * rec - 1);
    }
    var pk: engram_io.Packed = .{ .fd = fd, .base = header_bytes, .rec = rec, .rows = rows, .lo = h.lo, .hi = h.hi };
    var df = dio.File.open(path) catch null;
    if (df) |*f| {
        if (f.direct) {
            fds.append(gpa, f.fd) catch |err| {
                f.close();
                return err;
            };
            pk.fd_d = f.fd;
        } else f.close();
    }
    return pk;
}

/// Rows [h.lo, h.hi) of table `t` into `file` as a shard with header `h`: runs of rows, each run's weights and its
/// scales one sequential read each, interleaved into records.
pub fn write(gpa: std.mem.Allocator, io: std.Io, t: engram_io.Table, h: Header, file: std.Io.File) !void {
    if (h.row_w != t.row_w or h.row_s != t.row_s or h.rows != t.rows or h.lo > h.hi or h.hi > t.rows) return error.BadPackedShard;
    var hb: [header_bytes]u8 = undefined;
    encode(h, &hb);
    try file.writePositionalAll(io, &hb, 0);
    const run: usize = 1 << 18;
    const rec = t.row_w + t.row_s;
    const w = try gpa.alloc(u8, run * t.row_w);
    defer gpa.free(w);
    const s = try gpa.alloc(u8, run * t.row_s);
    defer gpa.free(s);
    const out = try gpa.alloc(u8, run * rec);
    defer gpa.free(out);
    var at: u64 = h.lo;
    var report: u64 = h.lo + (16 << 20);
    while (at < h.hi) {
        const n: usize = @intCast(@min(@as(u64, run), h.hi - at));
        try engram_io.preadAll(t.fd_w, w[0 .. n * t.row_w], t.base_w + at * t.row_w);
        try engram_io.preadAll(t.fd_s, s[0 .. n * t.row_s], t.base_s + at * t.row_s);
        for (0..n) |i| {
            @memcpy(out[i * rec ..][0..t.row_w], w[i * t.row_w ..][0..t.row_w]);
            @memcpy(out[i * rec + t.row_w ..][0..t.row_s], s[i * t.row_s ..][0..t.row_s]);
        }
        try file.writePositionalAll(io, out[0 .. n * rec], header_bytes + (at - h.lo) * rec);
        at += n;
        if (at >= report) {
            std.debug.print("engram layer {d}: {d} of {d} rows\n", .{ h.layer, at - h.lo, h.hi - h.lo });
            report += 16 << 20;
        }
    }
}

test "a packed shard's records are the table's rows, a read each" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a [300, 100] weight table and its [300, 3] scales for layer 1
    const rows = 300;
    var json_buf: [256]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"U8\",\"shape\":[{d},100],\"data_offsets\":[0,{d}]}},\"layers.1.engram.embed.scale\":{{\"dtype\":\"U8\",\"shape\":[{d},3],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 100, rows, rows * 100, rows * 103 });
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try bytes.appendSlice(a, &len);
    try bytes.appendSlice(a, json);
    for (0..rows * 103) |i| try bytes.append(a, @intCast((i * 7 + 3) % 251));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = bytes.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    const t = tables.layers.get(1).?;
    // node 1 of 3: rows [100, 200)
    const r = range(rows, 1, 3);
    try std.testing.expectEqual([2]u64{ 100, 200 }, r);
    var nb: [64]u8 = undefined;
    var file = try tmp.dir.createFile(io, try name(&nb, 1, 1, 3), .{});
    try write(a, io, t, .{ .layer = 1, .rows = rows, .lo = r[0], .hi = r[1], .row_w = 100, .row_s = 3 }, file);
    file.close(io);
    var shards = try Shards.open(a, dir, &tables, 1, 3);
    defer shards.close();
    const pk = shards.layers.get(1).?;
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    const ids = [_]i64{ 100, 199, 150, 101, 150 };
    var want_w: [ids.len * 100]u8 = undefined;
    var want_s: [ids.len * 3]u8 = undefined;
    try pool.gather(t, &ids, &want_w, &want_s);
    var got: [ids.len * 103]u8 = undefined;
    var dsts: [ids.len][*]u8 = undefined;
    for (&dsts, 0..) |*x, i| x.* = got[i * 103 ..].ptr;
    try pool.gatherRecords(pk, &ids, &dsts);
    for (0..ids.len) |i| {
        try std.testing.expectEqualSlices(u8, want_w[i * 100 ..][0..100], got[i * 103 ..][0..100]);
        try std.testing.expectEqualSlices(u8, want_s[i * 3 ..][0..3], got[i * 103 + 100 ..][0..3]);
    }
    // a row outside the node's range is refused, and another node's open refuses this file
    try std.testing.expectError(error.ReadFailed, pool.gatherRecords(pk, &.{99}, dsts[0..1]));
    try std.testing.expectError(error.FileNotFound, Shards.open(a, dir, &tables, 0, 3));
}
