//! WP6 (docs/prefill-plan.md): the indexer's prompt-chunk scoring kernel (_index_score_rows) called once with
//! TF_DS_IDX_RB's kernel-side RB=1 (today's default: the kit dequantizes its 64-key tile once a row) and once
//! with RB=2 (the kit's own captured alternative: once per 2 rows) on identical input, then compares the score
//! output (and, in the "keyed" mode, the tile maxima) bit for bit. Real AOT kernel, no checkpoint: this tests
//! only that RB=2 is the same arithmetic as RB=1, not anything about the model's actual indexer weights.
//! Two "keys" modes (prompt.zig's two indexScore call sites: ch.keys with tmax, or plain ch.score without) and
//! two row counts, one a multiple of 2 and one odd (the last RB=2 group then covers a single valid row), since
//! the real engine's row-block size is not guaranteed even.
//!   tf-dsv41-idx-replay AOT_DIR
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const tri_index = dsv41.tri_index;

const IH: usize = 32;
const ID: usize = 128;
const N: usize = 256; // compressed keys visible: 4 tiles of 64

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

/// K's packed FP4 codes (random nibbles: every bit pattern is a valid E2M1 code, no NaN/Inf to worry about) and
/// E8M0 scales (random bytes in [23, 127]: check_native.py's "fast scaling" range, powers of two in [2^-104, 1]).
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

/// VIS: strictly increasing with row index (causal: row r sees up to roughly its own position), varied so an
/// RB=2 group's two rows do not always share the same visible count -- the per-row masking must stay per-row
/// even when the tile dequant is shared.
fn fillVis(buf: cuda.DeviceBuffer, rows: usize, a: std.mem.Allocator) !void {
    const host = try a.alloc(i64, rows);
    defer a.free(host);
    for (host, 0..) |*v, r| v.* = @intCast(@min(N, 16 + r * 3));
    try buf.upload(0, std.mem.sliceAsBytes(host));
}

fn zero(buf: cuda.DeviceBuffer, a: std.mem.Allocator) !void {
    const z = try a.alloc(u8, buf.len);
    @memset(z, 0);
    try buf.upload(0, z);
}

fn runCase(t: dsv41.tri.Tri, stream: cuda.Stream, a: std.mem.Allocator, q: cuda.DeviceBuffer, kcodes: cuda.DeviceBuffer, kscales: cuda.DeviceBuffer, wt: cuda.DeviceBuffer, vis: cuda.DeviceBuffer, out: cuda.DeviceBuffer, tmax: ?cuda.DeviceBuffer, keyed: bool, rows: usize) !void {
    try zero(out, a);
    if (tmax) |tm| try zero(tm, a);
    const k: tri_index.IndexK = .{ .fp4 = .{ .codes = kcodes.ptr, .scales = kscales.ptr } };
    try tri_index.indexScore(t, q.ptr, k, wt.ptr, vis.ptr, N, out.ptr, null, keyed, if (tmax) |tm| tm.ptr else null, false, rows, IH, ID);
    try stream.synchronize();
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-idx-replay AOT_DIR\n", .{});
        return 2;
    }

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(init.gpa, io, &driver, ctx.device, args[1]);
    const t: dsv41.tri.Tri = .{ .set = &set, .stream = stream, .pdl = true };
    const d = &driver;
    var rng = std.Random.DefaultPrng.init(0xc0ffee);

    const nt = (N + 63) / 64;
    var ok = true;
    for ([_]usize{ 64, 65 }) |rows| {
        for ([_]bool{ true, false }) |keyed| {
            var q = try cuda.DeviceBuffer.alloc(d, rows * IH * ID * 2);
            defer q.free();
            var kcodes = try cuda.DeviceBuffer.alloc(d, N * (ID / 2));
            defer kcodes.free();
            var kscales = try cuda.DeviceBuffer.alloc(d, N * (ID / 32));
            defer kscales.free();
            var wt = try cuda.DeviceBuffer.alloc(d, rows * IH * 2);
            defer wt.free();
            var vis = try cuda.DeviceBuffer.alloc(d, rows * 8);
            defer vis.free();
            const out_elem: usize = if (keyed) 8 else 4;
            var out1 = try cuda.DeviceBuffer.alloc(d, rows * N * out_elem);
            defer out1.free();
            var out2 = try cuda.DeviceBuffer.alloc(d, rows * N * out_elem);
            defer out2.free();
            var tmax1: ?cuda.DeviceBuffer = if (keyed) try cuda.DeviceBuffer.alloc(d, rows * nt * 8) else null;
            defer if (tmax1) |*b| b.free();
            var tmax2: ?cuda.DeviceBuffer = if (keyed) try cuda.DeviceBuffer.alloc(d, rows * nt * 8) else null;
            defer if (tmax2) |*b| b.free();

            try fillBf16(q, &rng, a);
            try fillK(kcodes, kscales, &rng, a);
            try fillBf16(wt, &rng, a);
            try fillVis(vis, rows, a);

            tri_index.idx_rb = 1;
            try runCase(t, stream, a, q, kcodes, kscales, wt, vis, out1, tmax1, keyed, rows);
            tri_index.idx_rb = 2;
            try runCase(t, stream, a, q, kcodes, kscales, wt, vis, out2, tmax2, keyed, rows);

            const b1 = try a.alloc(u8, out1.len);
            try out1.download(0, b1);
            const b2 = try a.alloc(u8, out2.len);
            try out2.download(0, b2);
            const label = if (keyed) "keyed" else "plain";
            if (std.mem.eql(u8, b1, b2)) {
                std.debug.print("ok   rows={d} {s}: OUT bit-identical RB=1 vs RB=2\n", .{ rows, label });
            } else {
                var nd: usize = 0;
                var first: ?usize = null;
                if (keyed) {
                    const v1 = std.mem.bytesAsSlice(i64, b1);
                    const v2 = std.mem.bytesAsSlice(i64, b2);
                    for (v1, v2, 0..) |x, y, i| {
                        if (x != y) {
                            nd += 1;
                            if (first == null) first = i;
                        }
                    }
                } else {
                    const v1 = std.mem.bytesAsSlice(u32, b1);
                    const v2 = std.mem.bytesAsSlice(u32, b2);
                    for (v1, v2, 0..) |x, y, i| {
                        if (x != y) {
                            nd += 1;
                            if (first == null) first = i;
                        }
                    }
                }
                std.debug.print("FAIL rows={d} {s}: OUT differs, {d}/{d} elements, first at {d}\n", .{ rows, label, nd, b1.len / out_elem, first.? });
                ok = false;
            }
            if (keyed) {
                const t1 = try a.alloc(u8, tmax1.?.len);
                try tmax1.?.download(0, t1);
                const t2 = try a.alloc(u8, tmax2.?.len);
                try tmax2.?.download(0, t2);
                if (std.mem.eql(u8, t1, t2)) {
                    std.debug.print("ok   rows={d} {s}: TMAX bit-identical RB=1 vs RB=2\n", .{ rows, label });
                } else {
                    std.debug.print("FAIL rows={d} {s}: TMAX differs\n", .{ rows, label });
                    ok = false;
                }
            }
        }
    }
    if (ok) {
        std.debug.print("WP6: _index_score_rows RB=2 == RB=1 (OUT and TMAX, keyed and plain, even and odd row counts)\n", .{});
        return 0;
    }
    return 1;
}
