//! Engram rows kept in host memory: a set-associative cache of packed records by row id, shared by every reader
//! thread. It holds the bytes a read returned, so a hit is the row a read would give.
const std = @import("std");

pub const Cache = struct {
    rec: usize,
    ways: usize,
    sets: usize,
    keys: []u64, // a way's row id + 1 (0: empty)
    data: []u8, // [sets * ways] records
    next: []u8, // each set's next way to fill
    locks: []std.atomic.Value(u32), // one a set
    hits: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),

    /// About `bytes` of records of `rec` bytes, `ways` a set (bytes too few for one set: error.CacheTooSmall).
    pub fn init(gpa: std.mem.Allocator, bytes: usize, rec: usize, ways: usize) !Cache {
        if (rec == 0 or ways == 0 or ways > 255) return error.BadCache;
        const sets = bytes / rec / ways;
        if (sets == 0) return error.CacheTooSmall;
        const keys = try gpa.alloc(u64, sets * ways);
        errdefer gpa.free(keys);
        const data = try gpa.alloc(u8, sets * ways * rec);
        errdefer gpa.free(data);
        const next = try gpa.alloc(u8, sets);
        errdefer gpa.free(next);
        const locks = try gpa.alloc(std.atomic.Value(u32), sets);
        @memset(keys, 0);
        @memset(next, 0);
        @memset(locks, .init(0));
        return .{ .rec = rec, .ways = ways, .sets = sets, .keys = keys, .data = data, .next = next, .locks = locks };
    }

    pub fn deinit(c: *Cache, gpa: std.mem.Allocator) void {
        gpa.free(c.keys);
        gpa.free(c.data);
        gpa.free(c.next);
        gpa.free(c.locks);
        c.* = undefined;
    }

    fn lock(c: *Cache, s: usize) void {
        while (c.locks[s].cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }

    fn unlock(c: *Cache, s: usize) void {
        c.locks[s].store(0, .release);
    }

    /// Row `id`'s record into `out` (rec bytes) when held.
    pub fn get(c: *Cache, id: u64, out: []u8) bool {
        const s: usize = @intCast(id % c.sets);
        c.lock(s);
        defer c.unlock(s);
        for (0..c.ways) |w| {
            const k = s * c.ways + w;
            if (c.keys[k] == id + 1) {
                @memcpy(out[0..c.rec], c.data[k * c.rec ..][0..c.rec]);
                _ = c.hits.fetchAdd(1, .monotonic);
                return true;
            }
        }
        _ = c.misses.fetchAdd(1, .monotonic);
        return false;
    }

    /// Row `id`'s record (rec bytes) kept, over the set's oldest fill (a row held already stays as it is).
    pub fn put(c: *Cache, id: u64, row: []const u8) void {
        const s: usize = @intCast(id % c.sets);
        c.lock(s);
        defer c.unlock(s);
        for (0..c.ways) |i| if (c.keys[s * c.ways + i] == id + 1) return;
        const w: usize = c.next[s] % c.ways;
        c.next[s] = @intCast((w + 1) % c.ways);
        const k = s * c.ways + w;
        @memcpy(c.data[k * c.rec ..][0..c.rec], row[0..c.rec]);
        c.keys[k] = id + 1;
    }
};

test "a cached record is the one put, and a full set gives up its oldest fill" {
    const a = std.testing.allocator;
    var c = try Cache.init(a, 2 * 2 * 4, 4, 2); // two sets of two ways
    defer c.deinit(a);
    var out: [4]u8 = undefined;
    try std.testing.expect(!c.get(6, &out));
    c.put(6, &.{ 1, 2, 3, 4 });
    try std.testing.expect(c.get(6, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    c.put(6, &.{ 9, 9, 9, 9 }); // held already: unchanged
    try std.testing.expect(c.get(6, &out));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &out);
    // set 0 holds 6; 8 and 10 fill it again, the first over 6
    c.put(8, &.{ 5, 5, 5, 5 });
    c.put(10, &.{ 7, 7, 7, 7 });
    try std.testing.expect(!c.get(6, &out));
    try std.testing.expect(c.get(8, &out) and c.get(10, &out));
    try std.testing.expectEqual(@as(u64, 4), c.hits.load(.monotonic));
    try std.testing.expectError(error.CacheTooSmall, Cache.init(a, 7, 4, 2));
}
