//! tf-dsv41-engram-pack TABLES_DIR OUT_DIR NODE NODES: each Engram layer's packed shard for node NODE of NODES
//! (engram_pack.zig: rows [rows * NODE / NODES, rows * (NODE + 1) / NODES), each row's FP8 values and scales side by
//! side) from the checkpoint's tables in TABLES_DIR, into OUT_DIR; NODES 1 packs the whole tables. A shard in OUT_DIR
//! that checks out stays; the others are written as dot files and renamed when whole. No GPU.
const std = @import("std");
const dsv41 = @import("dsv41");

const usage = "usage: tf-dsv41-engram-pack TABLES_DIR OUT_DIR NODE NODES\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const node = std.fmt.parseInt(u32, args[3], 10) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const nodes = std.fmt.parseInt(u32, args[4], 10) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    if (nodes == 0 or node >= nodes) {
        std.debug.print("NODE is below NODES\n{s}", .{usage});
        return 2;
    }
    var tables = try dsv41.engram_io.Tables.open(gpa, io, args[1]);
    defer tables.close();
    if (tables.layers.count() == 0) {
        std.debug.print("{s} holds no Engram tables\n", .{args[1]});
        return 1;
    }
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[2], .{});
    defer dir.close(io);
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    var it = tables.layers.iterator();
    while (it.next()) |kv| {
        const layer = kv.key_ptr.*;
        const t = kv.value_ptr.*;
        const r = dsv41.engram_pack.range(t.rows, node, nodes);
        const bytes = dsv41.engram_pack.header_bytes + (r[1] - r[0]) * (t.row_w + t.row_s);
        var nb: [64]u8 = undefined;
        const final = try dsv41.engram_pack.name(&nb, layer, node, nodes);
        // a whole shard from an earlier run stays
        var kept: dsv41.engram_pack.Shards = .{ .gpa = gpa };
        defer kept.close();
        if (dsv41.engram_pack.openOne(gpa, &kept.fds, args[2], layer, t.rows, t.row_w, t.row_s, node, nodes)) |_| {
            try out.interface.print("{{\"file\": \"{s}\", \"layer\": {d}, \"lo\": {d}, \"hi\": {d}, \"bytes\": {d}, \"kept\": true}}\n", .{ final, layer, r[0], r[1], bytes });
            continue;
        } else |_| {}
        const partial = try std.fmt.allocPrint(a, ".{s}.partial", .{final});
        var file = try dir.createFile(io, partial, .{});
        var file_open = true;
        defer if (file_open) file.close(io);
        errdefer dir.deleteFile(io, partial) catch {};
        try dsv41.engram_pack.write(gpa, io, t, .{ .layer = layer, .rows = t.rows, .lo = r[0], .hi = r[1], .row_w = @intCast(t.row_w), .row_s = @intCast(t.row_s) }, file);
        try file.sync(io);
        file.close(io);
        file_open = false;
        try dir.rename(partial, dir, final, io);
        try out.interface.print("{{\"file\": \"{s}\", \"layer\": {d}, \"lo\": {d}, \"hi\": {d}, \"bytes\": {d}, \"kept\": false}}\n", .{ final, layer, r[0], r[1], bytes });
        try out.interface.flush();
    }
    try out.interface.flush();
    return 0;
}
