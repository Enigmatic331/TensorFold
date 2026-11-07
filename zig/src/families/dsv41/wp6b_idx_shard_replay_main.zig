//! WP6(b) offline differential: `_index_score_rows` (tri_index.indexScore) called once as a 2048-row launch (the
//! reference, row offset 0 into [2048, ...] buffers -- today's unsharded path) and once as four 512-row launches
//! at row offsets 0/512/1024/1536 into the SAME full buffers (the row-sharded path: each node only computes its
//! own owned quarter). This is exactly what the real engine's indexer() will do once row-sharded -- not a fresh,
//! tightly-repacked buffer per quarter (unlike WP4's qhc_g/qhc_part): the indexer's own block-looping already
//! pointer-offsets into these same full-size q/wt/vis/out/tmax buffers at nonzero row starts for every call with
//! more than one row-block (prompt.zig's existing comment: "a row's scores and picks are its own, so the block
//! size changes no bits"), so a nonzero r0 is exercised by any long-context prompt in production today. What
//! row-sharding adds that is genuinely new is calling r0 = q, 2q, 3q as the FIRST launch of a call, not block 2,
//! 3, 4 of a single call that started at r0 = 0 -- this test is the direct check that distinction changes nothing.
//! Two "keys" modes, both idx_rb settings (RB=1 and the shipped default RB=2), and a key count (2048) big enough
//! to span multiple 64-key tiles, so a tile-boundary accounting bug would have somewhere to show up.
//!   tf-dsv41-wp6b-idx-shard-replay AOT_DIR [seed]
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const tri_index = dsv41.tri_index;

const IH: usize = 32;
const ID: usize = 128;
const N: usize = 2048; // compressed keys visible: 32 tiles of 64
const ROWS: usize = 2048; // the real chunk capacity
const QROWS: usize = 512; // ROWS / world (world = 4)
const NQ: usize = ROWS / QROWS;

fn fillBf16(buf: cuda.DeviceBuffer, rng: *std.Random.DefaultPrng, a: std.mem.Allocator) !void {
    const n = buf.len / 2;
    const host = try a.alloc(u16, n);
    defer a.free(host);
    for (host) |*v| {
        const f: f32 = rng.random().floatNorm(f32) * 0.5;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    try buf.upload(0, std.mem.sliceAsBytes(host));
}

fn fillK(codes: cuda.DeviceBuffer, scales: cuda.DeviceBuffer, rng: *std.Random.DefaultPrng, a: std.mem.Allocator) !void {
    const cb = try a.alloc(u8, codes.len);
    defer a.free(cb);
    for (cb) |*v| v.* = rng.random().int(u8);
    try codes.upload(0, cb);
    const sb = try a.alloc(u8, scales.len);
    defer a.free(sb);
    for (sb) |*v| v.* = 23 + rng.random().uintLessThan(u8, 127 - 23 + 1);
    try scales.upload(0, sb);
}

// strictly increasing with row index, same spirit as idx_replay_main.zig's: no two rows share a visible count,
// and every row sees comfortably more than one 64-key tile.
fn fillVis(buf: cuda.DeviceBuffer, rows: usize, a: std.mem.Allocator) !void {
    const host = try a.alloc(i64, rows);
    defer a.free(host);
    for (host, 0..) |*v, r| v.* = @intCast(@min(N, 64 + r));
    try buf.upload(0, std.mem.sliceAsBytes(host));
}

fn zero(buf: cuda.DeviceBuffer, a: std.mem.Allocator) !void {
    const z = try a.alloc(u8, buf.len);
    @memset(z, 0);
    try buf.upload(0, z);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-wp6b-idx-shard-replay AOT_DIR [seed]\n", .{});
        return 2;
    }
    const seed: u64 = if (args.len > 2) try std.fmt.parseInt(u64, args[2], 10) else 0xc0ffee;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, &driver, ctx.device, args[1]);
    const t: dsv41.tri.Tri = .{ .set = &set, .stream = stream, .pdl = true };
    const d = &driver;
    const nt = (N + 63) / 64;

    var ok = true;
    for ([_]usize{ 1, 2 }) |rb| {
        for ([_]bool{ true, false }) |keyed| {
            var rng = std.Random.DefaultPrng.init(seed);
            var q = try cuda.DeviceBuffer.alloc(d, ROWS * IH * ID * 2);
            defer q.free();
            var kcodes = try cuda.DeviceBuffer.alloc(d, N * (ID / 2));
            defer kcodes.free();
            var kscales = try cuda.DeviceBuffer.alloc(d, N * (ID / 32));
            defer kscales.free();
            var wt = try cuda.DeviceBuffer.alloc(d, ROWS * IH * 2);
            defer wt.free();
            var vis = try cuda.DeviceBuffer.alloc(d, ROWS * 8);
            defer vis.free();
            const out_elem: usize = if (keyed) 8 else 4;
            var out_full = try cuda.DeviceBuffer.alloc(d, ROWS * N * out_elem);
            defer out_full.free();
            var out_shard = try cuda.DeviceBuffer.alloc(d, ROWS * N * out_elem);
            defer out_shard.free();
            var tmax_full: ?cuda.DeviceBuffer = if (keyed) try cuda.DeviceBuffer.alloc(d, ROWS * nt * 8) else null;
            defer if (tmax_full) |*b| b.free();
            var tmax_shard: ?cuda.DeviceBuffer = if (keyed) try cuda.DeviceBuffer.alloc(d, ROWS * nt * 8) else null;
            defer if (tmax_shard) |*b| b.free();

            try fillBf16(q, &rng, a);
            try fillK(kcodes, kscales, &rng, a);
            try fillBf16(wt, &rng, a);
            try fillVis(vis, ROWS, a);
            const k: tri_index.IndexK = .{ .fp4 = .{ .codes = kcodes.ptr, .scales = kscales.ptr } };

            tri_index.idx_rb = rb;

            // Reference: one call, the full 2048 rows, row offset 0 -- today's unsharded path.
            try zero(out_full, a);
            if (tmax_full) |tm| try zero(tm, a);
            try tri_index.indexScore(t, q.ptr, k, wt.ptr, vis.ptr, N, out_full.ptr, null, keyed, if (tmax_full) |tm| tm.ptr else null, false, ROWS, IH, ID);
            try stream.synchronize();

            // Row-sharded: four separate 512-row calls at row offsets 0, 512, 1024, 1536 into the SAME buffers --
            // exactly what indexer()'s block loop will do when bounded to [owned.start, owned.start + owned.n).
            try zero(out_shard, a);
            if (tmax_shard) |tm| try zero(tm, a);
            for (0..NQ) |qi| {
                const r0 = qi * QROWS;
                const qoff = q.ptr + @as(u64, @intCast(r0 * IH * ID * 2));
                const woff = wt.ptr + @as(u64, @intCast(r0 * IH * 2));
                const voff = vis.ptr + @as(u64, @intCast(r0 * 8));
                const ooff = out_shard.ptr + @as(u64, @intCast(r0 * N * out_elem));
                const tmoff = if (tmax_shard) |tm| tm.ptr + @as(u64, @intCast(r0 * nt * 8)) else null;
                try tri_index.indexScore(t, qoff, k, woff, voff, N, ooff, null, keyed, tmoff, false, QROWS, IH, ID);
            }
            try stream.synchronize();

            const bf = try a.alloc(u8, out_full.len);
            try out_full.download(0, bf);
            const bs = try a.alloc(u8, out_shard.len);
            try out_shard.download(0, bs);
            const label = if (keyed) "keyed" else "plain";
            if (std.mem.eql(u8, bf, bs)) {
                std.debug.print("ok   RB={d} {s}: OUT bit-identical, full call vs 4 quarter calls at row offsets 0/512/1024/1536\n", .{ rb, label });
            } else {
                var nd: usize = 0;
                var first: ?usize = null;
                if (keyed) {
                    const vf = std.mem.bytesAsSlice(i64, bf);
                    const vs = std.mem.bytesAsSlice(i64, bs);
                    for (vf, vs, 0..) |x, y, i| {
                        if (x != y) {
                            nd += 1;
                            if (first == null) first = i;
                        }
                    }
                } else {
                    const vf = std.mem.bytesAsSlice(u32, bf);
                    const vs = std.mem.bytesAsSlice(u32, bs);
                    for (vf, vs, 0..) |x, y, i| {
                        if (x != y) {
                            nd += 1;
                            if (first == null) first = i;
                        }
                    }
                }
                std.debug.print("FAIL RB={d} {s}: OUT differs, {d}/{d} elements, first at {d}\n", .{ rb, label, nd, bf.len / out_elem, first.? });
                ok = false;
            }
            if (keyed) {
                const tf = try a.alloc(u8, tmax_full.?.len);
                try tmax_full.?.download(0, tf);
                const ts = try a.alloc(u8, tmax_shard.?.len);
                try tmax_shard.?.download(0, ts);
                if (std.mem.eql(u8, tf, ts)) {
                    std.debug.print("ok   RB={d} {s}: TMAX bit-identical, full call vs 4 quarter calls\n", .{ rb, label });
                } else {
                    std.debug.print("FAIL RB={d} {s}: TMAX differs\n", .{ rb, label });
                    ok = false;
                }
            }
        }
    }
    if (ok) {
        std.debug.print("WP6(b): _index_score_rows at a row-sharded quarter's own offset == the full-chunk call (OUT and TMAX, keyed and plain, RB=1 and RB=2)\n", .{});
        return 0;
    }
    return 1;
}
