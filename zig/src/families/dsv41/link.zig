//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (node to node, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        var p = try Ports.open(&.{ip}, port);
        defer p.close();
        return p.accept(1);
    }

    /// A socket bound to `ip`:`port` and listening (a backlog of one: the port is one peer's).
    fn bound(ip: [4]u8, port: u16) !c.fd_t {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        return fd;
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        var tries: u32 = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            var a = addr(ip, port);
            if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return .{ .fd = fd };
            _ = c.close(fd);
            if (tries >= seconds * 4) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(250), .awake) catch {};
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// Rank 0's ports for its followers: follower g's (g = 1 to n) is `port + g - 1` on `ips[g - 1]`. All of them listen
/// from `open` on, so a follower that is ready early connects into its port's backlog (the kernel completes the
/// handshake) and waits there while rank 0 loads its weights or takes the followers before it.
pub const Ports = struct {
    fds: [3]c.fd_t = .{ -1, -1, -1 },

    pub fn open(ips: []const [4]u8, port: u16) !Ports {
        if (ips.len > 3) return error.TooManyFollowers;
        var p: Ports = .{};
        errdefer p.close();
        for (ips, 0..) |ip, i| p.fds[i] = try Link.bound(ip, port + @as(u16, @intCast(i)));
        return p;
    }

    /// Follower g's link (g from 1): its connection, waited for if it has not come yet; its port then closes.
    pub fn accept(p: *Ports, g: usize) !Link {
        const fd = p.fds[g - 1];
        if (fd < 0) return error.NoSuchFollower;
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        p.fds[g - 1] = -1;
        Link.tune(peer);
        return .{ .fd = peer };
    }

    pub fn close(p: *Ports) void {
        for (&p.fds) |*fd| if (fd.* >= 0) {
            _ = c.close(fd.*);
            fd.* = -1;
        };
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("10.200.30"));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

test "followers that are ready first wait in their ports' backlogs" {
    // followers 3, 2 and 1 connect in that order, each giving up after a second, before rank 0 takes any of them:
    // with one port listening at a time (each opened after the follower before it was in), 3 and 2 would give up
    const io = std.testing.io;
    const lo: [4]u8 = .{ 127, 0, 0, 1 };
    const port: u16 = 29600 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 300)));
    var ports = try Ports.open(&.{ lo, lo, lo }, port);
    defer ports.close();
    const Follower = struct {
        fn run(p: u16, g: u8, out: *?Link) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p + g - 1, 1) catch return;
            l.send(&.{g}) catch {
                l.close();
                return;
            };
            out.* = l;
        }
    };
    var theirs: [3]?Link = .{ null, null, null };
    defer for (&theirs) |*l| if (l.*) |*x| x.close();
    for ([_]u8{ 3, 2, 1 }) |g| {
        const t = try std.Thread.spawn(.{}, Follower.run, .{ port, g, &theirs[g - 1] });
        t.join();
    }
    for (theirs) |l| try std.testing.expect(l != null);
    var buf: [4]u8 = undefined;
    for (1..4) |g| {
        var l = try ports.accept(g);
        defer l.close();
        try std.testing.expectEqualSlices(u8, &.{@as(u8, @intCast(g))}, try l.recv(&buf));
    }
    try std.testing.expectError(error.NoSuchFollower, ports.accept(1));
}
