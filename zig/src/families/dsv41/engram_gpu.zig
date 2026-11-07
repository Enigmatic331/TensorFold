//! Engram's packed records to bf16 rows on the GPU (zig/kernels/cuda/dsv41/engram.cu, cuda.kernels.dsv41_engram), each
//! value as engram.decodeRow gives it on the host, so the rows a layer reads are the host decode's bits.
const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    module: cuda.Module,
    decode: cuda.Function,

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .decode = try m.function("tf_dse_decode_kernel") };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }

    /// Records (r, lo + c) of `rec` ([rows][slots] records of dim + dim / 32 bytes) into bf16 rows (r, c) of `out`
    /// ([rows][k][dim]) for c below k.
    pub fn decodeRows(k: *const Kernels, s: cuda.Stream, rec: u64, out: u64, rows: usize, slots: usize, lo: usize, n: usize, dim: usize) !void {
        const total = rows * n * dim;
        if (total == 0) return;
        if (lo + n > slots or dim % 32 != 0) return error.BadEngramRows;
        var a: cuda.Args = .{};
        a.add(rec);
        a.add(out);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(slots)));
        a.add(@as(c_int, @intCast(lo)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(dim)));
        try cuda.launch.launch(k.decode, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }
};
