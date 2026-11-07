//! WP4 offline differential test: hc_pre_pf (tri_hc.hcPrePf) called once as a 2048-row launch (the reference) and
//! once as four 512-row launches that match the REAL ENGINE's actual quarter-HC buffer handling exactly -- not
//! tf-dsv41-hc-replay's "every buffer freshly sized tight" version. h, pre_in, pre_out, post, comb and x are
//! POINTER OFFSETS into the same full-2048 buffers the reference used (prompt.zig's attnMixes/ffnMixes do this,
//! and tf-dsv41-hc-replay already proved it is safe for exactly these buffers). `gathered` is repacked into a
//! fresh, tightly-strided [world, 512, D] buffer per quarter (qhc_g in prompt.zig; RS = rows * D makes a pointer
//! offset into the full-cap gathered buffer wrong). `part` is a fresh, tightly-sized [512, 40, 32] buffer, SHARED
//! across quarters and reused call to call (qhc_part in prompt.zig) -- not sliced from the full-cap part buffer,
//! per tf-dsv41-hc-replay's own comment that part's layout is undocumented scratch, not row-major.
//!   tf-dsv41-wp4-replay AOT_DIR
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");
const tri_hc = dsv41.tri_hc;

const D: usize = 5120;
const HC: usize = 4;
const WORLD: usize = 2;
const ROWS: usize = 2048;
const QROWS: usize = 512;
const NQ: usize = ROWS / QROWS;

fn fillF32(buf: cuda.DeviceBuffer, rng: *std.Random.DefaultPrng, a: std.mem.Allocator) !void {
    const n = buf.len / 4;
    const host = try a.alloc(f32, n);
    defer a.free(host);
    for (host) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
    try buf.upload(0, std.mem.sliceAsBytes(host));
}

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

fn zero(buf: cuda.DeviceBuffer, off: usize, n: usize, a: std.mem.Allocator) !void {
    const z = try a.alloc(u8, n);
    @memset(z, 0);
    try buf.upload(off, z);
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-dsv41-wp4-replay AOT_DIR [seed]\n", .{});
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

    var rng = std.Random.DefaultPrng.init(seed);
    const d = &driver;

    var h = try cuda.DeviceBuffer.alloc(d, ROWS * HC * D * 2);
    defer h.free();
    var h_out = try cuda.DeviceBuffer.alloc(d, ROWS * HC * D * 2);
    defer h_out.free();
    var out = try cuda.DeviceBuffer.alloc(d, ROWS * D * 2);
    defer out.free();
    var part = try cuda.DeviceBuffer.alloc(d, ROWS * 40 * 32 * 4); // full-cap part, as the real ch.part is sized
    defer part.free();
    var pre_in = try cuda.DeviceBuffer.alloc(d, ROWS * HC * 4);
    defer pre_in.free();
    var pre_out = try cuda.DeviceBuffer.alloc(d, ROWS * HC * 4);
    defer pre_out.free();
    var post = try cuda.DeviceBuffer.alloc(d, ROWS * HC * 4);
    defer post.free();
    var comb = try cuda.DeviceBuffer.alloc(d, ROWS * HC * HC * 4);
    defer comb.free();

    var fn_w = try cuda.DeviceBuffer.alloc(d, 24 * HC * D * 4);
    defer fn_w.free();
    var scale = try cuda.DeviceBuffer.alloc(d, 3 * 4);
    defer scale.free();
    var base = try cuda.DeviceBuffer.alloc(d, 24 * 4);
    defer base.free();
    var norm_w = try cuda.DeviceBuffer.alloc(d, D * 2);
    defer norm_w.free();

    var gathered_full = try cuda.DeviceBuffer.alloc(d, WORLD * ROWS * D * 4);
    defer gathered_full.free();

    for ([_]cuda.DeviceBuffer{ h, norm_w }) |buf| try fillBf16(buf, &rng, a);
    for ([_]cuda.DeviceBuffer{ part, pre_in, post, comb, fn_w, scale, base, gathered_full }) |buf| try fillF32(buf, &rng, a);
    try zero(h_out, 0, h_out.len, a);
    try zero(out, 0, out.len, a);
    try zero(pre_out, 0, pre_out.len, a);
    try zero(post, 0, post.len, a);
    try zero(comb, 0, comb.len, a);

    // Reference: the full 2048-row launch.
    _ = try tri_hc.hcPrePf(t, h.ptr, fn_w.ptr, scale.ptr, base.ptr, pre_in.ptr, norm_w.ptr, 1e-20, 1e-6, 20, out.ptr, pre_out.ptr, post.ptr, comb.ptr, part.ptr, .{ .gathered = gathered_full.ptr, .world = WORLD, .h_out = h_out.ptr }, ROWS, D);
    try stream.synchronize();

    const full_h_out = try a.alloc(u8, h_out.len);
    try h_out.download(0, full_h_out);
    const full_out = try a.alloc(u8, out.len);
    try out.download(0, full_out);
    const full_pre_out = try a.alloc(u8, pre_out.len);
    try pre_out.download(0, full_pre_out);
    const full_post = try a.alloc(u8, post.len);
    try post.download(0, full_post);
    const full_comb = try a.alloc(u8, comb.len);
    try comb.download(0, full_comb);

    // WP4's actual pattern: pointer-offset h/pre_in/pre_out/post/comb/out into the SAME full buffers (not fresh
    // per-quarter ones); a tight, repacked `gathered` (qhc_g); a tight, SHARED `part` scratch (qhc_part).
    var gq = try cuda.DeviceBuffer.alloc(d, WORLD * QROWS * D * 4);
    defer gq.free();
    var part_q = try cuda.DeviceBuffer.alloc(d, QROWS * 40 * 32 * 4); // qhc_part: tight, shared across quarters
    defer part_q.free();

    var ok = true;
    for (0..NQ) |qi| {
        const row0 = qi * QROWS;
        for (0..WORLD) |w| {
            const src = gathered_full.ptr + @as(u64, @intCast((w * ROWS + row0) * D * 4));
            const dst = gq.ptr + @as(u64, @intCast(w * QROWS * D * 4));
            try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, QROWS * D * 4, stream.handle), "cuMemcpyDtoDAsync");
        }
        try stream.synchronize();
        const hq = h.ptr + @as(u64, @intCast(row0 * HC * D * 2));
        const hoq = h_out.ptr + @as(u64, @intCast(row0 * HC * D * 2));
        const outq = out.ptr + @as(u64, @intCast(row0 * D * 2));
        const preiq = pre_in.ptr + @as(u64, @intCast(row0 * HC * 4));
        const preoq = pre_out.ptr + @as(u64, @intCast(row0 * HC * 4));
        const postq = post.ptr + @as(u64, @intCast(row0 * HC * 4));
        const combq = comb.ptr + @as(u64, @intCast(row0 * HC * HC * 4));
        try zero(h_out, row0 * HC * D * 2, QROWS * HC * D * 2, a);
        try zero(out, row0 * D * 2, QROWS * D * 2, a);
        try zero(pre_out, row0 * HC * 4, QROWS * HC * 4, a);
        try zero(post, row0 * HC * 4, QROWS * HC * 4, a);
        try zero(comb, row0 * HC * HC * 4, QROWS * HC * HC * 4, a);
        _ = try tri_hc.hcPrePf(t, hq, fn_w.ptr, scale.ptr, base.ptr, preiq, norm_w.ptr, 1e-20, 1e-6, 20, outq, preoq, postq, combq, part_q.ptr, .{ .gathered = gq.ptr, .world = WORLD, .h_out = hoq }, QROWS, D);
        try stream.synchronize();

        const qbytes_hc = QROWS * HC * D * 2;
        const qbytes_d = QROWS * D * 2;
        const qbytes_hc4 = QROWS * HC * 4;
        const qbytes_hc2_4 = QROWS * HC * HC * 4;
        var buf = try a.alloc(u8, qbytes_hc);
        try h_out.download(row0 * HC * D * 2, buf);
        if (!std.mem.eql(u8, buf, full_h_out[row0 * HC * D * 2 ..][0..qbytes_hc])) {
            std.debug.print("DIFFERS: h_out quarter {d}\n", .{qi});
            ok = false;
        }
        buf = try a.alloc(u8, qbytes_d);
        try out.download(row0 * D * 2, buf);
        if (!std.mem.eql(u8, buf, full_out[row0 * D * 2 ..][0..qbytes_d])) {
            const qu16 = std.mem.bytesAsSlice(u16, buf);
            const ru16 = std.mem.bytesAsSlice(u16, full_out[row0 * D * 2 ..][0..qbytes_d]);
            var n_diff: usize = 0;
            var first: ?usize = null;
            for (qu16, ru16, 0..) |qv, rv, idx| {
                if (qv == rv) continue;
                n_diff += 1;
                if (first == null) first = idx;
            }
            std.debug.print("DIFFERS: x (normed output) quarter {d}: {d}/{d} bf16 elements differ; first at {d}: quarter=0x{x:0>4} full=0x{x:0>4}\n", .{ qi, n_diff, qu16.len, first.?, qu16[first.?], ru16[first.?] });
            ok = false;
        }
        buf = try a.alloc(u8, qbytes_hc4);
        try pre_out.download(row0 * HC * 4, buf);
        if (!std.mem.eql(u8, buf, full_pre_out[row0 * HC * 4 ..][0..qbytes_hc4])) {
            std.debug.print("DIFFERS: pre_out quarter {d}\n", .{qi});
            ok = false;
        }
        buf = try a.alloc(u8, qbytes_hc4);
        try post.download(row0 * HC * 4, buf);
        if (!std.mem.eql(u8, buf, full_post[row0 * HC * 4 ..][0..qbytes_hc4])) {
            std.debug.print("DIFFERS: post quarter {d}\n", .{qi});
            ok = false;
        }
        buf = try a.alloc(u8, qbytes_hc2_4);
        try comb.download(row0 * HC * HC * 4, buf);
        if (!std.mem.eql(u8, buf, full_comb[row0 * HC * HC * 4 ..][0..qbytes_hc2_4])) {
            std.debug.print("DIFFERS: comb quarter {d}\n", .{qi});
            ok = false;
        }
    }
    if (ok) {
        std.debug.print("WP4 phase 1: the real engine's exact buffer pattern (offset h/pre/post/comb, tight g, tight shared part) matches the full launch bit for bit\n", .{});
    }

    // Phase 2 (Codex's proposed experiment, the layer-20 boundary): after the quarter loop above, `post`/`comb`
    // hold the FULL, correctly-reassembled state (each quarter wrote only its own slice onto zeroed ground,
    // proven bit-identical to the reference above) -- exactly what qhcResyncBoundary's fix produces. Now run the
    // boundary layer's own full-row, pending=true hcPrePf call (the real attnMixes(li == c.layers/2) path) twice
    // on identical h/pre_in/gathered: once reading this correct post/comb ("fixed"), once reading post/comb with
    // 3 of 4 quarters overwritten by unrelated random data ("buggy": qhcResyncBoundary never ran). If the bug
    // diagnosis is right, the two boundary runs diverge even though every other input is identical.
    {
        const fixed_post = try a.alloc(u8, post.len);
        try post.download(0, fixed_post);
        const fixed_comb = try a.alloc(u8, comb.len);
        try comb.download(0, fixed_comb);

        var fn_w2 = try cuda.DeviceBuffer.alloc(d, 24 * HC * D * 4);
        defer fn_w2.free();
        var scale2 = try cuda.DeviceBuffer.alloc(d, 3 * 4);
        defer scale2.free();
        var base2 = try cuda.DeviceBuffer.alloc(d, 24 * 4);
        defer base2.free();
        var norm_w2 = try cuda.DeviceBuffer.alloc(d, D * 2);
        defer norm_w2.free();
        var gathered2 = try cuda.DeviceBuffer.alloc(d, WORLD * ROWS * D * 4);
        defer gathered2.free();
        for ([_]cuda.DeviceBuffer{norm_w2}) |buf| try fillBf16(buf, &rng, a);
        for ([_]cuda.DeviceBuffer{ fn_w2, scale2, base2, gathered2 }) |buf| try fillF32(buf, &rng, a);

        var h_out2 = try cuda.DeviceBuffer.alloc(d, ROWS * HC * D * 2);
        defer h_out2.free();
        var out2 = try cuda.DeviceBuffer.alloc(d, ROWS * D * 2);
        defer out2.free();
        var pre_out2 = try cuda.DeviceBuffer.alloc(d, ROWS * HC * 4);
        defer pre_out2.free();

        // Run A: post/comb = the fixed, fully-reassembled state (qhcResyncBoundary's fix).
        try post.upload(0, fixed_post);
        try comb.upload(0, fixed_comb);
        try zero(h_out2, 0, h_out2.len, a);
        try zero(out2, 0, out2.len, a);
        try zero(pre_out2, 0, pre_out2.len, a);
        _ = try tri_hc.hcPrePf(t, h.ptr, fn_w2.ptr, scale2.ptr, base2.ptr, pre_in.ptr, norm_w2.ptr, 1e-20, 1e-6, 20, out2.ptr, pre_out2.ptr, post.ptr, comb.ptr, part.ptr, .{ .gathered = gathered2.ptr, .world = WORLD, .h_out = h_out2.ptr }, ROWS, D);
        try stream.synchronize();
        const a_out = try a.alloc(u8, out2.len);
        try out2.download(0, a_out);
        const a_h_out = try a.alloc(u8, h_out2.len);
        try h_out2.download(0, a_h_out);

        // Run B ("buggy"): post/comb = quarter 0's real slice kept, quarters 1-3 overwritten with unrelated
        // random data (simulating qhcResyncBoundary never having run -- stale leftover content, not layer 19's).
        try post.upload(0, fixed_post);
        try comb.upload(0, fixed_comb);
        for (1..NQ) |qi| {
            const row0 = qi * QROWS;
            const stale_post = try a.alloc(f32, QROWS * HC);
            for (stale_post) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
            try post.upload(row0 * HC * 4, std.mem.sliceAsBytes(stale_post));
            const stale_comb = try a.alloc(f32, QROWS * HC * HC);
            for (stale_comb) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
            try comb.upload(row0 * HC * HC * 4, std.mem.sliceAsBytes(stale_comb));
        }
        try zero(h_out2, 0, h_out2.len, a);
        try zero(out2, 0, out2.len, a);
        try zero(pre_out2, 0, pre_out2.len, a);
        _ = try tri_hc.hcPrePf(t, h.ptr, fn_w2.ptr, scale2.ptr, base2.ptr, pre_in.ptr, norm_w2.ptr, 1e-20, 1e-6, 20, out2.ptr, pre_out2.ptr, post.ptr, comb.ptr, part.ptr, .{ .gathered = gathered2.ptr, .world = WORLD, .h_out = h_out2.ptr }, ROWS, D);
        try stream.synchronize();
        const b_out = try a.alloc(u8, out2.len);
        try out2.download(0, b_out);
        const b_h_out = try a.alloc(u8, h_out2.len);
        try h_out2.download(0, b_h_out);

        const diverges = !std.mem.eql(u8, a_out, b_out) or !std.mem.eql(u8, a_h_out, b_h_out);
        if (diverges) {
            std.debug.print("WP4 phase 2 CONFIRMS the diagnosis: the boundary layer's x/h_out change when post/comb for 3 of 4 quarters are stale instead of resynced -- qhcResyncBoundary must gather post and comb, not just h and pre.\n", .{});
        } else {
            std.debug.print("WP4 phase 2: boundary layer's x/h_out did NOT change when post/comb were corrupted for 3 of 4 quarters -- post/comb resync is not needed; the diagnosis is wrong, keep looking.\n", .{});
        }
        if (!ok or !diverges) return 1;
        return 0;
    }
}
