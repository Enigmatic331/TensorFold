//! Draft depth by expected tokens a millisecond (Python family_depth.py DraftDepth and FamilyRounds._head_depth).
const std = @import("std");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;
const alloc = @import("allocate.zig");

pub const Constants = struct {
    depth_rate: f64 = 0.15, // weight of the newest round in a depth's acceptance
    depth_probe_every: u64 = 8, // rounds between probes one depth deeper
    cost_rate: f64 = 0.2, // weight of the newest round in a depth's measured time
    plain_margin: f64 = 1.05, // a drafted depth must beat plain rounds' tokens a ms by this much
    plain_wait_most: u64 = 128, // plain rounds between probes at most
    prepass_probe: u64 = 8, // prepass: shared passes a stream sits out before it joins one again to remeasure its gain
    draft_slack: i64 = 2, // drafts a stream offers past what its last shared round granted it
};

/// A stream's acceptance by depth and probe counters (Python `_depth_state[stream]`).
pub const State = struct {
    p: []f64,
    rounds: u64 = 0,
    plain: ?u64 = null,
    wait: ?u64 = null,
    ms: ?Table = null, // plain_guard: the stream's own round times
    skipped: u64 = 0, // prepass: shared passes sat out in a row

    pub fn deinit(st: *State, gpa: Allocator) void {
        gpa.free(st.p);
        if (st.ms) |*t| t.deinit(gpa);
    }
};

/// What the rule reads of a stream when it decides.
pub const Who = struct {
    state: *?State,
    draft_room: i64,
    finished: bool,
    forced: bool,
    granted: ?u32 = null,
    gain: ?f64 = null,
    gain_at: [9]f64 = @splat(0),
    stream: ?*anyopaque = null,
};

/// Whether a copied continuation is ahead of a stream (the round loop proposes; Python `_copy_proposal`).
pub const Probe = struct {
    ptr: *anyopaque,
    copy: *const fn (ptr: *anyopaque, who: *const Who) bool,
};

pub const Overhead = struct { streams: u32, ms: f64 };

pub const Rule = struct {
    gpa: Allocator,
    k: Constants = .{},
    prior: []const f64,
    most_drafts: u32,
    family_costs: *const Table,
    shared_costs: *const Table,
    mtp_step_ms: f64,
    plain_guard: bool,
    prepass: bool = false,
    pass_ms: [2]f64 = .{ 0, 0 }, // prepass: a batched head pass's ms, a + b x streams (measured; TF_DS_PASS_MS)
    round_streams: usize = 0, // prepass: the shared round's streams, drafting or not (roundStreams sets it)
    draft_row_ms: f64 = 0.0, // A3 (TF_DS_DRAFT_ROW_MS): a draft row's ms apart from the streams' rows (0: the table alone)
    node_probabilities: bool,
    batch_rows: u32,
    round_ms: Table = .{}, // a round's wall time by drafts, shared by every stream
    overhead: std.ArrayList(Overhead) = .empty, // a shared round's ms past its forward, by streams (insertion order)

    pub fn deinit(r: *Rule) void {
        r.round_ms.deinit(r.gpa);
        r.overhead.deinit(r.gpa);
    }

    pub fn rates(r: *Rule, who: Who) ![]f64 {
        if (who.state.* == null) {
            const n = @min(r.prior.len, @max(1, r.most_drafts));
            who.state.* = .{ .p = try r.gpa.dupe(f64, r.prior[0..n]) };
        }
        return who.state.*.?.p;
    }

    /// Draft j was tried when drafts 1 .. j - 1 were kept; its estimate moves toward whether it was kept.
    pub fn observeDepth(r: *Rule, who: Who, proposed: i64, accepted: i64) !void {
        const p = try r.rates(who);
        const n = @min(@max(proposed, 0), @as(i64, @intCast(p.len)));
        var j: i64 = 0;
        while (j < n) : (j += 1) {
            if (accepted < j) break;
            const hit: f64 = if (accepted > j) 1.0 else 0.0;
            const at: usize = @intCast(j);
            p[at] += r.k.depth_rate * (hit - p[at]);
        }
    }

    /// Round times by depth: the stream's own under `plain_guard` (its sampling sets a row's host work), else shared.
    fn costs(r: *Rule, who: ?Who) !*Table {
        const w = who orelse return &r.round_ms;
        if (!r.plain_guard) return &r.round_ms;
        _ = try r.rates(w);
        const st = &w.state.*.?;
        if (st.ms == null) st.ms = .{};
        return &st.ms.?;
    }

    /// A stream's first round carries the prefill-to-decode switch: its time stays out of the costs.
    pub fn observeCost(r: *Rule, drafts: i64, ms: f64, initializing: bool, who: ?Who) !void {
        if (drafts < 0 or (drafts == 0 and !r.plain_guard) or initializing) return;
        const table = try r.costs(who);
        const before = table.get(drafts);
        try table.put(r.gpa, drafts, if (before) |b| b + r.k.cost_rate * (ms - b) else ms);
    }

    /// Measured round time when there is one, else the load-time forward cost plus head steps.
    pub fn roundCost(r: *Rule, drafts: i64, who: ?Who) !?f64 {
        const table = try r.costs(who);
        if (table.get(drafts)) |v| return v;
        const forward = r.family_costs.get(drafts + 1) orelse return null;
        var modeled = forward + r.mtp_step_ms * @as(f64, @floatFromInt(drafts));
        if (r.plain_guard) {
            // untimed: its model plus the least time a row of a timed round took beyond its model
            var least: ?f64 = null;
            for (table.slots, 0..) |slot, d| {
                const ms = slot orelse continue;
                const fc = r.family_costs.get(@intCast(d + 1)) orelse continue;
                const beyond = (ms - fc - r.mtp_step_ms * @as(f64, @floatFromInt(d))) / @as(f64, @floatFromInt(d + 1));
                least = if (least) |l| (if (beyond < l) beyond else l) else beyond;
            }
            const floor = least orelse 0.0;
            modeled += @as(f64, @floatFromInt(drafts + 1)) * (if (floor > 0.0) floor else 0.0);
        }
        return modeled;
    }

    /// The most expected tokens a ms, probing one depth farther every `depth_probe_every` rounds.
    pub fn depth(r: *Rule, who: Who) !i64 {
        const most = @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1));
        if (most <= 0) return 0;
        const p = try r.rates(who);
        if (r.family_costs.empty()) {
            const want: i64 = if (p[0] < 0.8) 1 else if (p[0] < 0.9) 2 else 3;
            return @max(1, @min(most, want));
        }
        var best: i64 = 1;
        var best_rate: f64 = -1.0;
        if (r.plain_guard) {
            if (try r.roundCost(0, who)) |plain| {
                if (plain != 0.0) { // a draft must beat plain by plain_margin: estimates near the line are noise
                    best = 0;
                    best_rate = r.k.plain_margin / plain;
                }
            }
        }
        var expected: f64 = 1.0;
        var run: f64 = 1.0;
        var d: i64 = 1;
        while (d <= most) : (d += 1) {
            const cost = (try r.roundCost(d, who)) orelse break;
            const at: usize = @intCast(d - 1);
            run *= if (at < p.len) p[at] else p[p.len - 1];
            expected += run;
            if (expected / cost > best_rate) {
                best = d;
                best_rate = expected / cost;
            }
        }
        const st = &who.state.*.?;
        st.rounds += 1;
        if (best == 0) return @intFromBool(r.probePlain(st));
        st.plain = 0;
        st.wait = r.k.depth_probe_every;
        if (best < most and st.rounds % r.k.depth_probe_every == 0) best += 1;
        return best;
    }

    /// The decode plan's A2 (docs/UNIFIED-DECODE-IMPROVEMENTS.md), `prepass` on (TF_DS_PREPASS=1), shared rounds:
    /// which streams join the next head pass, chosen before it. The pass costs its rows whether or not a draft is
    /// verified after it, so the streams in it and their drafts are priced together: each stream's expected landed
    /// drafts `gain` and rows `granted` from its last allocation, a round's ms as the shared table at the round's rows
    /// plus the pass's measured `pass_ms[0] + pass_ms[1] x streams in it` (counted once). Streams join in order of gain
    /// a row, the best prefix of that order by expected tokens a ms (each stream's pending row: one token, drafted or not).
    /// Streams with no gain measured yet, and a stream that sat out `prepass_probe` passes in a row (its gain stale),
    /// always join. Decided anew every round at the round's stream count. depths[i] = 0 for a stream that sits out:
    /// its windows plain, its drafter history still absorbed by every verify. Replies are the same tokens either way
    /// (drafts only decide how many rows a round verifies).
    pub fn prepassSelect(r: *Rule, whos: []Who, depths: []i64) !void {
        if (!r.prepass or whos.len == 0 or (r.pass_ms[0] == 0 and r.pass_ms[1] == 0)) return;
        const n = whos.len;
        const in = try r.gpa.alloc(bool, n);
        defer r.gpa.free(in);
        const order = try r.gpa.alloc(usize, n);
        defer r.gpa.free(order);
        var m: usize = 0; // candidates: drafting streams with a measured gain and not due to probe
        var members: usize = 0;
        // every stream of the round has its pending row and token, drafting or not
        const all = @max(n, r.round_streams);
        var rows: f64 = @floatFromInt(all);
        var tokens: f64 = @floatFromInt(all);
        for (whos, depths, in, 0..) |w, d, *x, i| {
            x.* = false;
            if (d == 0) continue;
            const st = if (w.state.*) |*s| s else {
                x.* = true;
                members += 1;
                continue;
            };
            if (w.gain == null or w.granted == null or st.skipped >= r.k.prepass_probe) {
                x.* = true;
                members += 1;
                if (w.granted != null) {
                    const c = cut(w, d);
                    tokens += w.gain_at[c];
                    rows += @floatFromInt(c);
                }
                continue;
            }
            order[m] = i;
            m += 1;
        }
        // the candidates by gain a row (rows at least 1: a pass row's share), highest first, ties by stream order
        const Ctx = struct {
            w: []Who,
            d: []i64,
            fn density(c: @This(), i: usize) f64 {
                const k = cut(c.w[i], c.d[i]);
                return c.w[i].gain_at[k] / @as(f64, @floatFromInt(@max(k, 1)));
            }
            fn lt(c: @This(), a: usize, b: usize) bool {
                const da = c.density(a);
                const db = c.density(b);
                return da > db or (da == db and a < b);
            }
        };
        std.sort.insertion(usize, order[0..m], Ctx{ .w = whos, .d = depths }, Ctx.lt);
        // every prefix of that order, the best kept (as allocate keeps its best prefix: the pass's fixed part can make
        // one stream lose and several win)
        var best = r.passRate(rows, tokens, members);
        var take: usize = 0;
        for (order[0..m], 1..) |i, j| {
            const c = cut(whos[i], depths[i]);
            rows += @floatFromInt(c);
            tokens += whos[i].gain_at[c];
            members += 1;
            const now = r.passRate(rows, tokens, members);
            if (now > best) {
                best = now;
                take = j;
            }
        }
        for (order[0..take]) |i| in[i] = true;
        for (whos, depths, in) |w, *d, x| {
            if (d.* == 0) continue;
            const st = if (w.state.*) |*s| s else continue;
            if (x) {
                st.skipped = 0;
            } else {
                st.skipped += 1;
                d.* = 0;
            }
        }
    }

    /// The drafts a stream would verify next round as prepassSelect prices them: its last grant, but no more than the
    /// depth asked of the pass now (room, a copy ahead) nor than gain_at holds.
    fn cut(w: Who, d: i64) usize {
        const g: usize = w.granted orelse 0;
        const dd: usize = @intCast(@max(d, 0));
        return @min(@min(g, dd), w.gain_at.len - 1);
    }

    /// prepassSelect's objective: expected tokens over the round's ms (the shared table at these rows, else the
    /// nearest timed width, plus the pass for `members` streams when any join it).
    fn passRate(r: *const Rule, rows: f64, tokens: f64, members: usize) f64 {
        const total: i64 = @intFromFloat(@ceil(rows));
        const last: i64 = @as(i64, @intCast(r.shared_costs.slots.len)) - 1;
        const streams: i64 = @intCast(r.round_streams);
        const fwd = if (r.draft_row_ms > 0.0 and streams > 0 and total >= streams)
            (r.shared_costs.get(@min(streams, last)) orelse return 0.0) + r.draft_row_ms * @as(f64, @floatFromInt(total - streams))
        else
            r.shared_costs.get(@min(total, last)) orelse return 0.0;
        const pass = if (members == 0) 0.0 else r.pass_ms[0] + r.pass_ms[1] * @as(f64, @floatFromInt(members));
        return tokens / (fwd + pass);
    }

    /// One draft after `wait` plain rounds in a row; each probe doubles the wait until drafting wins again.
    fn probePlain(r: *Rule, st: *State) bool {
        st.plain = (st.plain orelse 0) + 1;
        const wait = st.wait orelse r.k.depth_probe_every;
        if (st.plain.? < wait) return false;
        st.plain = 0;
        st.wait = @min(2 * wait, r.k.plain_wait_most);
        return true;
    }

    /// Python `_head_depth`: the next round's head drafts for a stream (0 when forced, 1 with a copy ahead).
    pub fn headDepth(r: *Rule, who: *const Who, budget: ?i64, probe: Probe) !i64 {
        const d: i64 = if (who.finished) 0 else if (budget) |b| b else if (r.node_probabilities)
            @min(@as(i64, r.most_drafts), @max(1, who.draft_room - 1))
        else
            try r.depth(who.*);
        if (who.forced) return 0;
        if (d != 0 and probe.copy(probe.ptr, who)) return 1;
        return d;
    }

    /// Head drafts for the streams of a shared round, sized by allocation over their chain chances.
    pub fn budgets(r: *Rule, whos: []Who, probe: Probe, out: []i64) !void {
        if (r.node_probabilities) {
            for (whos, out) |*w, *o| {
                const inner = try r.headDepth(w, null, probe);
                const granted: i64 = if (w.granted) |g| g else r.most_drafts;
                o.* = try r.headDepth(w, @min(inner, granted + r.k.draft_slack), probe);
            }
            try r.prepassSelect(whos, out);
            return;
        }
        const most = try r.gpa.alloc(i64, whos.len);
        defer r.gpa.free(most);
        const probs = try r.gpa.alloc([]f64, whos.len);
        defer r.gpa.free(probs);
        var made: usize = 0;
        defer for (probs[0..made]) |p| r.gpa.free(p);
        for (whos, most, probs) |w, *m, *p| {
            m.* = @min(@as(i64, r.most_drafts), @max(1, w.draft_room - 1));
            p.* = try alloc.chainProbabilities(r.gpa, try r.rates(w), @intCast(m.*));
            made += 1;
        }
        const ones = try r.gpa.alloc(u32, whos.len);
        defer r.gpa.free(ones);
        @memset(ones, 1);
        const counts = try alloc.allocate(r.gpa, ones, probs, r.shared_costs, r.overheadMs(whos.len), @max(whos.len, r.batch_rows));
        defer r.gpa.free(counts);
        if (r.plain_guard) {
            // no draft where none pays; a probe after a growing run of plain rounds
            for (whos, counts) |w, *c| {
                const st = &w.state.*.?;
                st.rounds += 1;
                if (c.* != 0) {
                    st.plain = 0;
                    st.wait = r.k.depth_probe_every;
                } else c.* = @intFromBool(r.probePlain(st));
            }
            for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @as(i64, c)), probe);
            return;
        }
        for (whos, most, counts, out) |*w, m, c, *o| o.* = try r.headDepth(w, @min(m, @max(1, @as(i64, c))), probe);
    }

    /// A shared round's ms beyond its forward at this many streams (the nearest measured count, first seen on ties).
    pub fn overheadMs(r: *const Rule, streams: usize) f64 {
        if (r.overhead.items.len == 0) return 8.0;
        var best = r.overhead.items[0];
        for (r.overhead.items[1..]) |o| {
            if (distance(o.streams, streams) < distance(best.streams, streams)) best = o;
        }
        return best.ms;
    }

    pub fn observeOverhead(r: *Rule, streams: u32, rows: u64, ms: f64) !void {
        const forward = if (r.draft_row_ms > 0.0 and rows >= streams)
            (r.shared_costs.get(@intCast(streams)) orelse return) + r.draft_row_ms * @as(f64, @floatFromInt(rows - streams))
        else
            r.shared_costs.get(@intCast(rows)) orelse return;
        const extra = if (ms - forward > 0.0) ms - forward else 0.0;
        for (r.overhead.items) |*o| {
            if (o.streams == streams) {
                o.ms = o.ms + 0.2 * (extra - o.ms);
                return;
            }
        }
        try r.overhead.append(r.gpa, .{ .streams = streams, .ms = extra });
    }
};

fn distance(a: u32, b: usize) usize {
    const x: usize = a;
    return if (x > b) x - b else b - x;
}

test "depth picks the most tokens a ms and probes deeper" {
    const gpa = std.testing.allocator;
    var fc: Table = .{};
    defer fc.deinit(gpa);
    for (1..5) |w| try fc.put(gpa, @intCast(w), 5.0 + @as(f64, @floatFromInt(w)));
    const shared: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = false, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 3), try rule.depth(who));
}
