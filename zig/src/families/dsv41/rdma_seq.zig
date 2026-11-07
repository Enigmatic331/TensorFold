//! The RDMA ring's sequence bookkeeping (rdma.zig's proxy thread), apart from the verbs so it can be tested: a
//! slot's parts over the devices, a peer's parts back to whole sequences, and when a slot's writes have all been sent.
//! A write's immediate carries its sequence's low 28 bits, and for a relayed part (a ring without a switch: a peer
//! across the ring is reached through the two neighbours) a relay bit and the rank whose slot it is.
const std = @import("std");

pub const max_ranks = 8;
pub const max_devices = 4;
pub const max_slots = 16;

const seq_bits = 28;
const seq_mask: u64 = (1 << seq_bits) - 1;
const relay_bit: u32 = 1 << 31;

/// Device `d`'s part of an `n`-byte slot over `nd` devices: whole 16-byte units each, the last device takes the rest.
pub fn part(n: u64, nd: u32, d: u32) struct { off: u64, len: u64 } {
    const unit = n / 16 / nd * 16;
    const off = unit * d;
    return .{ .off = off, .len = if (d == nd - 1) n - off else unit };
}

/// The immediate a write carries, in network order: its sequence's low 28 bits, and for a part relayed on from rank
/// `src`'s slot the relay bit and src (bits 28-30).
pub fn immediate(q: u64, relayed_from: ?u32) u32 {
    var w: u32 = @intCast(q & seq_mask);
    if (relayed_from) |s| w |= relay_bit | (s << seq_bits);
    return std.mem.nativeToBig(u32, w);
}

/// An immediate's parts: its sequence's low bits, and the rank whose slot a relayed part is (null: the sender's own).
pub fn decode(imm: u32) struct { low: u32, relayed_from: ?u32 } {
    const w = std.mem.bigToNative(u32, imm);
    return .{ .low = @intCast(w & seq_mask), .relayed_from = if (w & relay_bit != 0) (w >> seq_bits) & 7 else null };
}

/// A receive completion's sequence bits back to a sequence. A lane delivers a peer's parts in order, so it extends
/// `last`, the sequence that lane delivered from that peer before.
pub fn extend(last: u64, low: u32) u64 {
    var q = (last & ~seq_mask) | low;
    if (q <= last) q += 1 << seq_bits;
    return q;
}

/// What each peer has delivered: the last sequence each of its lanes brought (a lane: a device toward a peer, or for a
/// relayed peer one relaying neighbour) and the last one published in the flags. lanes[p]: peer p's lane count.
pub const Arrivals = struct {
    lanes: [max_ranks]u32 = @splat(1),
    got: [max_ranks][max_devices]u64 = @splat(@splat(0)),
    published: [max_ranks]u64 = @splat(0),

    /// A receive completion from `peer` on lane `l`: `seq` the sequence that lane brought, and `from`..`to` (inclusive;
    /// none when from > to) whose every part is now in, to publish in order (a peer may already be a gather ahead on
    /// one lane).
    pub fn arrive(a: *Arrivals, peer: usize, l: usize, low: u32) struct { seq: u64, from: u64, to: u64 } {
        a.got[peer][l] = extend(a.got[peer][l], low);
        var in = a.got[peer][0];
        for (a.got[peer][1..a.lanes[peer]]) |g| in = @min(in, g);
        const from = a.published[peer] + 1;
        if (in > a.published[peer]) a.published[peer] = in;
        return .{ .seq = a.got[peer][l], .from = from, .to = in };
    }
};

/// Write completions a gather's slot still owes: one a peer and device.
pub const Sends = struct {
    need: u32,
    slots: u32,
    seq: [max_slots]u64 = @splat(0),
    done: [max_slots]u32 = @splat(0),

    /// One completed write of sequence `q`: q when it was the last its slot owed (the slot may be staged again), else null.
    pub fn complete(s: *Sends, q: u64) ?u64 {
        const slot: usize = @intCast(q % s.slots);
        if (s.seq[slot] != q) {
            s.seq[slot] = q;
            s.done[slot] = 0;
        }
        s.done[slot] += 1;
        return if (s.done[slot] == s.need) q else null;
    }
};

/// A relaying rank's progress: the sequences whose every relayed write has completed, in order (`upto`), from the
/// completions of each sequence's `need` relays, which may finish out of order across sequences.
pub const Relays = struct {
    sends: Sends,
    done_at: [max_slots]u64 = @splat(0),
    upto: u64 = 0,

    pub fn init(need: u32, slots: u32) Relays {
        return .{ .sends = .{ .need = need, .slots = slots } };
    }

    pub fn complete(r: *Relays, q: u64) void {
        if (r.sends.complete(q)) |d| r.done_at[@intCast(d % r.sends.slots)] = d;
        while (r.done_at[@intCast((r.upto + 1) % r.sends.slots)] == r.upto + 1) r.upto += 1;
    }
};

test "a slot's parts: whole 16-byte units a device, the rest on the last" {
    const a = part(120_000, 2, 0);
    const b = part(120_000, 2, 1);
    try std.testing.expectEqual(@as(u64, 0), a.off);
    try std.testing.expectEqual(@as(u64, 60_000), a.len);
    try std.testing.expectEqual(@as(u64, 60_000), b.off);
    try std.testing.expectEqual(@as(u64, 60_000), b.len);
    const c = part(48, 2, 1);
    try std.testing.expectEqual(@as(u64, 16), c.off);
    try std.testing.expectEqual(@as(u64, 32), c.len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 0).len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 1).len);
    try std.testing.expectEqual(@as(u64, 4096), part(4096, 1, 0).len);
}

test "immediates extend a lane's last sequence, across the 28-bit wrap, and carry a relayed part's rank" {
    const low = struct {
        fn of(q: u64) u32 {
            return decode(immediate(q, null)).low;
        }
    }.of;
    try std.testing.expectEqual(@as(u64, 1), extend(0, low(1)));
    try std.testing.expectEqual(@as(u64, 7), extend(6, low(7)));
    try std.testing.expectEqual(@as(u64, 1 << 28), extend((1 << 28) - 1, low(1 << 28)));
    try std.testing.expectEqual(@as(u64, (1 << 32) + 3), extend((1 << 32) + 2, low((1 << 32) + 3)));
    try std.testing.expectEqual(@as(?u32, null), decode(immediate(5, null)).relayed_from);
    const d = decode(immediate((1 << 30) + 9, 3));
    try std.testing.expectEqual(@as(?u32, 3), d.relayed_from);
    try std.testing.expectEqual(@as(u32, 9), d.low);
    try std.testing.expectEqual(@as(?u32, 0), decode(immediate(1, 0)).relayed_from);
    // the wire bytes are big-endian
    try std.testing.expectEqual(@as(u32, 0x0102_0304), std.mem.bigToNative(u32, immediate(0x0102_0304, null)));
}

test "a sequence is published once every lane has delivered its part" {
    var a: Arrivals = .{};
    a.lanes[1] = 2;
    a.lanes[3] = 2;
    const low = struct {
        fn of(q: u64) u32 {
            return decode(immediate(q, null)).low;
        }
    }.of;
    var r = a.arrive(1, 0, low(1));
    try std.testing.expect(r.from > r.to);
    try std.testing.expectEqual(@as(u64, 1), r.seq);
    r = a.arrive(1, 0, low(2));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 1, low(1));
    try std.testing.expectEqual(@as(u64, 1), r.from);
    try std.testing.expectEqual(@as(u64, 1), r.to);
    r = a.arrive(1, 1, low(2));
    try std.testing.expectEqual(@as(u64, 2), r.from);
    try std.testing.expectEqual(@as(u64, 2), r.to);
    // another peer is independent
    r = a.arrive(3, 1, low(1));
    try std.testing.expect(r.from > r.to);
    // one lane: every completion publishes its sequence
    var b: Arrivals = .{};
    r = b.arrive(0, 0, low(1));
    try std.testing.expectEqual(@as(u64, 1), r.to);
}

test "a slot is sent after its last write: peers x devices completions" {
    var s: Sends = .{ .need = 6, .slots = 4 };
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(1));
    try std.testing.expectEqual(@as(?u64, 1), s.complete(1));
    // sequence 5 reuses slot 1 and starts its own count
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(5));
    try std.testing.expectEqual(@as(?u64, null), s.complete(2));
    try std.testing.expectEqual(@as(?u64, 5), s.complete(5));
}

test "relays complete in order even when sequences finish out of order" {
    var r = Relays.init(2, 4);
    r.complete(2);
    r.complete(2);
    try std.testing.expectEqual(@as(u64, 0), r.upto);
    r.complete(1);
    try std.testing.expectEqual(@as(u64, 0), r.upto);
    r.complete(1);
    try std.testing.expectEqual(@as(u64, 2), r.upto);
    r.complete(3);
    r.complete(3);
    try std.testing.expectEqual(@as(u64, 3), r.upto);
}
