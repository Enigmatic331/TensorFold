//! CUDA-graph capture failures are an optimization failure, not a request failure: keep the eager result and
//! temporarily stop trying to capture new shapes. Adapted for the DSv4.1 graph paths from MiaAI-Lab's Flash Next
//! fallback (Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold, release 1.0.0).

const std = @import("std");

pub const Fault = struct {
    pause: u32 = 0,
    streak: u32 = 0,
    total: u64 = 0,

    /// Whether this miss should stay eager. Exactly `pause` misses are skipped after a failure.
    pub fn skip(f: *Fault) bool {
        if (f.pause == 0) return false;
        f.pause -= 1;
        return true;
    }

    /// Record one failed capture round on every rank. Repeated failures back off 64, 128, ... 4096 misses.
    pub fn failed(f: *Fault) u32 {
        f.total += 1;
        f.streak +|= 1;
        f.pause = @as(u32, 64) << @intCast(@min(f.streak - 1, 6));
        return f.pause;
    }

    /// A completed capture breaks the failure streak; historical totals remain available for logs.
    pub fn succeeded(f: *Fault) void {
        f.streak = 0;
    }
};

test "capture fault: exponential pause, exact skips and success resets the streak" {
    var f: Fault = .{};
    try std.testing.expect(!f.skip());
    try std.testing.expectEqual(@as(u32, 64), f.failed());
    for (0..64) |_| try std.testing.expect(f.skip());
    try std.testing.expect(!f.skip());
    try std.testing.expectEqual(@as(u32, 128), f.failed());
    f.succeeded();
    try std.testing.expectEqual(@as(u32, 64), f.failed());
    try std.testing.expectEqual(@as(u64, 3), f.total);
}

test "capture fault: pause saturates at 4096 misses" {
    var f: Fault = .{};
    var pause: u32 = 0;
    for (0..20) |_| pause = f.failed();
    try std.testing.expectEqual(@as(u32, 4096), pause);
}
