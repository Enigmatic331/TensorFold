//! Late admission and per-stream prompt state through the served lane host.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const gpa = std.testing.allocator;

const Action = enum { none, cancel_long, cancel_short, refuse_short, full };

const Seen = struct {
    action: Action = .none,
    engine: api.Engine = undefined,
    short: *const api.Request,
    long_chunks: usize = 0,
    short_at: ?usize = null,
    submitted: bool = false,
    tokens: [2]std.ArrayList(u32) = @splat(.empty),
    reasons: [2]?api.Reason = .{ null, null },
    done: std.atomic.Value(u32) = .init(0),

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        if (s.prompt_len != 32) return;
        x.long_chunks += 1;
        if (x.long_chunks == 1) switch (x.action) {
            .cancel_long => x.engine.cancel(1),
            .cancel_short => x.engine.cancel(2),
            else => {},
        };
        if (!x.submitted) {
            x.submitted = true;
            x.engine.submit(2, x.short, .{ .ctx = x, .event = event }) catch unreachable;
        }
    }

    fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        const i: usize = @intCast(id - 1);
        switch (e.*) {
            .tokens => |t| {
                if (id == 2 and x.short_at == null) x.short_at = x.long_chunks;
                x.tokens[i].appendSlice(gpa, t) catch unreachable;
            },
            .finished => |f| {
                x.reasons[i] = f.reason;
                _ = x.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }

    fn wait(x: *Seen) !void {
        for (0..2000) |_| {
            if (x.done.load(.acquire) == 2) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.RequestsDidNotFinish;
    }

    fn deinit(x: *Seen) void {
        for (&x.tokens) |*t| t.deinit(gpa);
    }
};

fn reference(cfg: *const lanes.Config, prompt: []const u32) ![]u32 {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, cfg, target.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "reference", .prompt = prompt, .max_new = 8, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    while (core.activeCount() > 0) try core.step();
    return gpa.dupe(u32, s.emitted());
}

test "a late short request completes before another prompt finishes filling, with the same tokens as solo" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const long: [32]u32 = @splat(3);
    const short = [_]u32{ 1, 2, 3 };
    const short_request: api.Request = .{ .prompt = &short, .max_tokens = 8, .drafts = false };
    var seen: Seen = .{ .short = &short_request };
    defer seen.deinit();
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 8, .prefill_hook = Seen.hook, .prefill_hook_ctx = &seen };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    seen.engine = host.engine();
    try host.start();
    defer host.stop();
    const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
    try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
    try seen.wait();
    try std.testing.expect(seen.short_at.? < 8);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[0].?);
    try std.testing.expectEqual(api.Reason.length, seen.reasons[1].?);
    const want_long = try reference(&cfg, &long);
    defer gpa.free(want_long);
    const want_short = try reference(&cfg, &short);
    defer gpa.free(want_short);
    try std.testing.expectEqualSlices(u32, want_long, seen.tokens[0].items);
    try std.testing.expectEqualSlices(u32, want_short, seen.tokens[1].items);
}

test "partial admission releases cancelled prompts, isolates refusal, and retries a full pool with another filler" {
    for ([_]Action{ .cancel_long, .cancel_short, .refuse_short, .full }) |action| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        const long: [32]u32 = @splat(3);
        const short = [_]u32{ 1, 2, 3 };
        const short_request: api.Request = .{
            .prompt = &short,
            .max_tokens = 8,
            .drafts = false,
            .sampling = if (action == .refuse_short) .{ .temperature = 1, .seed = 7 } else null,
        };
        var seen: Seen = .{ .short = &short_request, .action = action, .submitted = true };
        defer seen.deinit();
        var target: lanes.fake.Fake = .{
            .gpa = gpa,
            .prefill_hook = Seen.hook,
            .prefill_hook_ctx = &seen,
            .refuse_sampled = action == .refuse_short,
            .refuse_full = if (action == .full) 1 else 0,
        };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.chunkedBackend(), clock.clock());
        defer core.deinit();
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        seen.engine = host.engine();
        const request: api.Request = .{ .prompt = &long, .max_tokens = 8, .drafts = false };
        try seen.engine.submit(1, &request, .{ .ctx = &seen, .event = Seen.event });
        try seen.engine.submit(2, &short_request, .{ .ctx = &seen, .event = Seen.event });
        try host.start();
        defer host.stop();
        try seen.wait();
        const want_long: api.Reason = if (action == .cancel_long) .cancelled else .length;
        const want_short: api.Reason = switch (action) {
            .cancel_short => .cancelled,
            .refuse_short => .failed,
            else => .length,
        };
        try std.testing.expectEqual(want_long, seen.reasons[0].?);
        try std.testing.expectEqual(want_short, seen.reasons[1].?);
        if (want_long == .length) {
            const want = try reference(&cfg, &long);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[0].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[0].items.len);
        if (want_short == .length) {
            const want = try reference(&cfg, &short);
            defer gpa.free(want);
            try std.testing.expectEqualSlices(u32, want, seen.tokens[1].items);
        } else try std.testing.expectEqual(@as(usize, 0), seen.tokens[1].items.len);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
    }
}
