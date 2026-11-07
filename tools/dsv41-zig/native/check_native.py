"""Checks of zig/kernels/cuda/dsv41/native.cu against DeepSeek's reference arithmetic (inference/model.py `linear`,
`Expert`, `MoE`; kernel.py `act_quant`, `fp8_gemm`, `fp4_gemm`), written here in plain torch, on real tensors of
deepseek-ai/DeepSeek-V4.1-Flash-1017. Runs in nvcr.io/nvidia/pytorch:26.07-py3 on a GB10:

    python check_native.py <checkpoint dir> [layer] [index json, when the checkpoint's own is not there yet]

It builds native_host.cu into a shared library (nvcc), then checks:
  1. act_quant: codes and scales equal the reference's bit for bit;
  2. the FP8 and MXFP4 GEMMs against an fp64 sum of the same quantized operands (relative error at fp32 rounding),
     at several row counts, K splits and row tiles, and that a row's bits never depend on the launch (row-invariance);
  3. the grouped experts (gate / up, SwiGLU with the clamps and route weights, down, routed sum) against MoE.forward's
     arithmetic for a layer's experts;
  4. the bf16 linears (wo_a's grouped einsum, the head) against fp64;
  5. the prompt chunks' prefill tile GEMMs (tf_dsn_pf_gemm / gu / dn / bf16): the same bits as the tile kernels at
     every row count (so the same bounds), and a row's bits never depending on the launch, the grouped experts with
     the shared FP8 expert and experts spanning several row tiles.
Exit status 0 when every check holds.
"""
import ctypes
import json
import math
import os
import struct
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cuda")
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                         dtype=torch.float64)
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- the library
def build():
    so = os.path.join("/tmp", "dsn_native_host.so")
    src = os.path.join(HERE, "native_host.cu")
    subprocess.check_call(["nvcc", "-arch=sm_121a", "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-o", so, src])
    lib = ctypes.CDLL(so)
    return lib


P = ctypes.c_void_p
L = ctypes.c_long
I = ctypes.c_int
F = ctypes.c_float


def ptr(t):
    return P(t.data_ptr()) if t is not None else P(0)


# ---------------------------------------------------------------- safetensors
class Shards:
    def __init__(self, root, index=None):
        self.root = root
        self.map = json.load(open(index or os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.headers = {}

    def header(self, f):
        if f not in self.headers:
            with open(os.path.join(self.root, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (json.loads(fh.read(n)), 8 + n)
        return self.headers[f]

    def get(self, name):
        f = self.map[name]
        h, base = self.header(f)
        e = h[name]
        a, b = e["data_offsets"]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            raw = bytearray(fh.read(b - a))
        dt = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.uint8, "BF16": torch.bfloat16,
              "F32": torch.float32, "F16": torch.float16}[e["dtype"]]
        return torch.frombuffer(raw, dtype=dt).reshape(e["shape"])


# ---------------------------------------------------------------- reference arithmetic (kernel.py, model.py)
def ref_act_quant(x):
    """act_quant(x.bf16, 32, 'ue8m0'): codes (float8_e4m3fn) and power-of-two scales (fp32)."""
    x = x.to(torch.bfloat16).float()
    m, k = x.shape
    g = x.view(m, k // 32, 32)
    amax = g.abs().amax(-1).clamp_min(1e-4)
    v = amax * torch.tensor(1 / 448.0, dtype=torch.float32)
    bits = v.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s = torch.ldexp(torch.ones_like(v), e)
    q = (g / s.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(m, k)
    return q, s


def e8m0(s):
    return torch.ldexp(torch.ones(s.shape, dtype=torch.float64, device=s.device), s.to(torch.int32) - 127)


def dequant_fp8(w, ws):
    n, k = w.shape
    return (w.to(torch.float64).view(n // 32, 32, k // 32, 32) * e8m0(ws)[:, None, :, None]).view(n, k)


def dequant_fp4(w, ws):
    n, k2 = w.shape
    b = w.to(torch.int64)
    vals = torch.stack([FP4_TABLE.to(w.device)[b & 0xF], FP4_TABLE.to(w.device)[(b >> 4) & 0xF]], -1).view(n, 2 * k2)
    return (vals.view(n, -1, 32) * e8m0(ws)[:, :, None]).view(n, 2 * k2)


def ref_linear64(q, s, wdq):
    """the quantized activations times the dequantized weights, exactly (fp64)."""
    m, k = q.shape
    xdq = (q.to(torch.float64).view(m, k // 32, 32) * s.to(torch.float64)[:, :, None]).view(m, k)
    return xdq @ wdq.T


# ---------------------------------------------------------------- 1. act_quant
def check_act_quant(lib):
    torch.manual_seed(0)
    for dtype, code in ((torch.bfloat16, 1), (torch.float32, 2), (torch.float16, 0)):
        x = (torch.randn(37, 5120, device=dev) * torch.logspace(-6, 3, 5120, device=dev)).to(dtype)
        x[3, :32] = 0  # an all-zero group: amax floored at 1e-4
        q = torch.empty(37, 5120, dtype=torch.uint8, device=dev)
        s = torch.empty(37, 160, dtype=torch.float32, device=dev)
        assert lib.dsn_act_quant(ptr(x), code, 5120, 37, 5120, ptr(q), 5120, ptr(s), 160, P(0), 0) == 0
        rq, rs = ref_act_quant(x)
        check(f"act_quant {dtype}: codes", torch.equal(q, rq.view(torch.uint8)))
        check(f"act_quant {dtype}: scales", torch.equal(s, rs))
        q2 = torch.empty_like(q)
        s2 = torch.empty_like(s)
        assert lib.dsn_dec_quant(ptr(x), code, 5120, 37, 5120, ptr(q2), 5120, ptr(s2), 160) == 0
        check(f"dec_quant {dtype}: the codes and scales of act_quant", torch.equal(q2, q) and torch.equal(s2, s))
    fake = torch.empty(37, 5120, dtype=torch.bfloat16, device=dev)
    assert lib.dsn_act_quant(ptr(x), 0, 5120, 37, 5120, P(0), 0, P(0), 0, ptr(fake), 5120) == 0
    rq, rs = ref_act_quant(x)
    rf = (rq.float().view(37, 160, 32) * rs[:, :, None]).view(37, 5120).to(torch.bfloat16)
    check("act_quant fake (quant-dequant to bf16)", torch.equal(fake, rf))


# ---------------------------------------------------------------- 2. the GEMMs
def run_mx(lib, q, s, w, ws, fmt, n, k, splits, mt, rows=None, m=None):
    m = m if m is not None else q.shape[0]
    y = torch.empty(m, n, dtype=torch.float32, device=dev)
    z = torch.empty(max(splits, 1) * m * n, dtype=torch.float32, device=dev)
    ldw = w.shape[1]
    assert lib.dsn_mx_gemm(ptr(q), ptr(s), ptr(rows), k, k // 32, ptr(w), ptr(ws), ldw, ws.shape[1], fmt, m, n, k,
                           splits, mt, ptr(z), ptr(y), n, 2, 0) == 0
    return y


def run_mx_dt(lib, q, s, w, ws, fmt, n, k, splits, mt, y_dtype, m=None):
    """run_mx with the output dtype (0 f16, 1 bf16, 2 f32)."""
    m = m if m is not None else q.shape[0]
    dt = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}[y_dtype]
    y = torch.empty(m, n, dtype=dt, device=dev)
    z = torch.empty(max(splits, 1) * m * n, dtype=torch.float32, device=dev)
    assert lib.dsn_mx_gemm(ptr(q), ptr(s), P(0), k, k // 32, ptr(w), ptr(ws), w.shape[1], ws.shape[1], fmt, m, n, k,
                           splits, mt, ptr(z), ptr(y), n, y_dtype, 0) == 0
    return y


def run_pf(lib, q, s, w, ws, n, k, splits, y_dtype=2, y=None, accumulate=0):
    """the prefill tile GEMM (FP8) of q's rows."""
    m = q.shape[0]
    dt = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}[y_dtype]
    if y is None:
        y = torch.empty(m, n, dtype=dt, device=dev)
    assert lib.dsn_pf_gemm(ptr(q), ptr(s), q.stride(0), s.stride(0), ptr(w), ptr(ws), w.shape[1], ws.shape[1], m, n, k,
                           splits, ptr(y), n, y_dtype, accumulate) == 0
    return y


def check_pf_gemm(lib, name, w, ws, wdq):
    """the prefill GEMM against the tile kernel (bits) and fp64, at several row counts and splits."""
    n, k = w.shape[0], wdq.shape[1]
    if n % 128 or k % 64 or k > 8192:
        return
    torch.manual_seed(5)
    for M in (1, 77, 130, 300):
        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16) * 3
        q, s = ref_act_quant(x)
        q = q.view(torch.uint8).contiguous()
        ref = ref_linear64(q.view(torch.float8_e4m3fn), s, wdq)
        for splits in sorted({1, 4, splits_of(k, n)}):
            old = run_mx(lib, q, s, w, ws, 0, n, k, splits, 2)
            new = run_pf(lib, q, s, w, ws, n, k, splits)
            check(f"{name} prefill GEMM M={M} splits={splits}: same bits as the tile kernel", torch.equal(new, old))
            r = rel(new, ref)
            check(f"{name} prefill GEMM M={M} splits={splits}: vs fp64", r < 2e-6, f"max rel {r:.2e}")
        if M == 300:
            sp = splits_of(k, n)
            full = run_pf(lib, q, s, w, ws, n, k, sp)
            alone = torch.cat([run_pf(lib, q[i:i + 1].contiguous(), s[i:i + 1].contiguous(), w, ws, n, k, sp) for i in (0, 131, 299)])
            check(f"{name} prefill GEMM: a row alone equals its batch row", torch.equal(alone, full[[0, 131, 299]]))
            # bf16 output, and fp32 accumulate (y +=), as the tile kernel's
            ob = run_mx_dt(lib, q, s, w, ws, 0, n, k, sp, 2, 1)
            nb = run_pf(lib, q, s, w, ws, n, k, sp, y_dtype=1)
            check(f"{name} prefill GEMM bf16 output: same bits as the tile kernel", torch.equal(ob, nb))
            base = torch.randn(M, n, device=dev)
            acc = run_pf(lib, q, s, w, ws, n, k, sp, y=base.clone(), accumulate=1)
            check(f"{name} prefill GEMM accumulate: y + its result", torch.equal(acc, base + full))


DEC_SPLITS = ((8, 1), (4, 1), (2, 2), (8, 2), (8, 4))
DEC_CNT = None


def run_dec(lib, fmt, q, s, w, ws, n, k, ks, cs, rows=None, m=None, y_dtype=2, scale=1.0, round_first=0, multi=1):
    """the decode tensor-core kernel (fmt 0 FP8, 1 FP4: q, s quantized; 2 bf16: q the bf16 rows, s None)."""
    global DEC_CNT
    if DEC_CNT is None:
        DEC_CNT = torch.zeros(65536, dtype=torch.int32, device=dev)
    m = m if m is not None else q.shape[0]
    y = torch.empty(multi * m, n, dtype=torch.float32 if y_dtype == 2 else torch.bfloat16, device=dev)
    z = torch.empty(multi * max(cs, 1) * m * n, dtype=torch.float32, device=dev)
    ldw = w.shape[1] * w.element_size()
    assert lib.dsn_dec_dense(fmt, ptr(q), ptr(s), ptr(rows), q.shape[1], k // 32, ptr(w), ptr(ws), ldw,
                             ws.shape[1] if ws is not None else 0, m, n, k, ks, cs, ptr(z), ptr(DEC_CNT), ptr(y), n,
                             y_dtype, 0, F(scale), round_first, multi) == 0
    assert int(DEC_CNT.abs().sum().item()) == 0, "the split counters are left zero"
    return y


def slab_dec(lib, fmt, q, s, rows, ldq, lds, w, ws, n, k, ks, y, y_dtype, accumulate, m, gn=0, gq=0, gs=0):
    """mx.decDenseMulti past 64 rows: launches of 64 rows at the slab's offsets into one y (decSlab's arithmetic: codes
    ldq bytes a row (bf16: ldq elements), scales lds floats, a rows list 4 bytes an entry, y ldy elements of its type)."""
    qsz = 2 if fmt == 2 else 1
    ysz = 4 if (accumulate or y_dtype == 2) else 2
    for r0 in range(0, m, 64):
        mm = min(64, m - r0)
        qp = q.data_ptr() + (0 if rows is not None else r0 * ldq * qsz)
        sp = 0 if s is None else s.data_ptr() + (0 if rows is not None else r0 * lds * 4)
        rp = 0 if rows is None else rows.data_ptr() + r0 * 4
        assert lib.dsn_dec_dense_g(fmt, P(qp), P(sp), P(rp), ldq, lds, ptr(w), ptr(ws), w.shape[1] * w.element_size(),
                                   ws.shape[1] if ws is not None else 0, mm, n, k, ks, P(0), ptr(DEC_CNT),
                                   P(y.data_ptr() + r0 * y.stride(0) * ysz), y.stride(0), y_dtype, accumulate, gn, gq, gs) == 0


def check_dec_slabs_prod(lib, sh, layer):
    """G0.1 with the engine's own slab offsets into ONE output (not a fresh output a slab): FP8 into fp32 with accumulate,
    a gathered rows list, and wo_a's FP8 twin as the block-diagonal groups (gn, gq, gs), at R = 80 against each row's
    launch alone (R = 1) and inside one 64-row launch."""
    global DEC_CNT
    if DEC_CNT is None:
        DEC_CNT = torch.zeros(65536, dtype=torch.int32, device=dev)
    pre = f"layers.{layer}."
    torch.manual_seed(5)
    # FP8 (wq_a) into fp32 rows that already hold values, accumulate=1
    w = sh.get(pre + "attn.wq_a.weight").to(dev).view(torch.uint8).contiguous()
    ws = sh.get(pre + "attn.wq_a.scale").to(dev)
    n, k = w.shape[0], w.shape[1]
    x = torch.randn(80, k, device=dev, dtype=torch.bfloat16) * 3
    q, s = ref_act_quant(x)
    ref = ref_linear64(q, s, dequant_fp8(sh.get(pre + "attn.wq_a.weight").to(dev), ws))
    q = q.view(torch.uint8).contiguous()
    y0 = torch.randn(80, n, device=dev)
    for ks in (8, 4, 2):
        y = y0.clone()
        slab_dec(lib, 0, q, s, None, k, k // 32, w, ws, n, k, ks, y, 2, 1, 80)
        alone = y0.clone()
        for i in (0, 63, 64, 79):
            slab_dec(lib, 0, q[i:i + 1], s[i:i + 1], None, k, k // 32, w, ws, n, k, ks, alone[i:i + 1], 2, 1, 1)
        y64 = y0.clone()
        slab_dec(lib, 0, q[16:80], s[16:80], None, k, k // 32, w, ws, n, k, ks, y64[16:80], 2, 1, 64)
        ok = all(torch.equal(y[i], alone[i]) for i in (0, 63, 64, 79)) and torch.equal(y[16:80], y64[16:80])
        check(f"attn.wq_a (fp8) dec ks={ks}: R = 80 slabs, accumulate into fp32: a row's bits at R = 1, 64, 80", ok)
        # a gathered rows list across the slab boundary
        rows = torch.tensor([79, 3, 64, 64, 10] * 16, dtype=torch.int32, device=dev)
        yg = torch.zeros(80, n, device=dev)
        slab_dec(lib, 0, q, s, rows, k, k // 32, w, ws, n, k, ks, yg, 2, 0, 80)
        yr = torch.zeros(80, n, device=dev)
        slab_dec(lib, 0, q, s, None, k, k // 32, w, ws, n, k, ks, yr, 2, 0, 80)
        check(f"attn.wq_a (fp8) dec ks={ks}: gathered rows across slabs equal their rows", torch.equal(yg, yr[rows.long()]))
        r = rel(yr, ref)
        check(f"attn.wq_a (fp8) dec ks={ks}: R = 80 slabs vs fp64", r < 2e-6, f"max rel {r:.2e}")
    # wo_a's FP8 twin: two groups of gk inputs, each half of the output columns reading its own group, bf16 out
    w = sh.get(pre + "attn.wo_a.weight").to(dev).view(torch.uint8).contiguous()
    ws = sh.get(pre + "attn.wo_a.scale").to(dev)
    n, gk = w.shape[0], w.shape[1]
    groups = 2
    x = torch.randn(80, groups * gk, device=dev, dtype=torch.bfloat16) * 3
    q8, s = ref_act_quant(x)
    q = q8.view(torch.uint8).contiguous()
    kk = groups * gk
    for ks in (8, 4):
        y = torch.zeros(80, n, device=dev, dtype=torch.bfloat16)
        slab_dec(lib, 0, q, s, None, kk, kk // 32, w, ws, n, gk, ks, y, 1, 0, 80, n // groups, gk, gk // 32)
        alone = torch.zeros(80, n, device=dev, dtype=torch.bfloat16)
        for i in (0, 63, 64, 79):
            slab_dec(lib, 0, q[i:i + 1], s[i:i + 1], None, kk, kk // 32, w, ws, n, gk, ks, alone[i:i + 1], 1, 0, 1, n // groups, gk, gk // 32)
        ok = all(torch.equal(y[i], alone[i]) for i in (0, 63, 64, 79))
        # and the groups as two plain launches on their own columns and inputs
        y2 = torch.zeros(80, n, device=dev, dtype=torch.bfloat16)
        for g in range(groups):
            slab_dec(lib, 0, q[:, g * gk:].contiguous()[:, :gk].contiguous(), s[:, g * gk // 32:(g + 1) * gk // 32].contiguous(), None, gk, gk // 32,
                     w[g * (n // groups):(g + 1) * (n // groups)].contiguous(), ws[g * (n // groups) // 32:(g + 1) * (n // groups) // 32].contiguous(),
                     n // groups, gk, ks, y2[:, g * (n // groups):], 1, 0, 80)
        check(f"attn.wo_a (fp8 twin) dec ks={ks}: R = 80 grouped slabs: a row's bits at R = 1 and as two plain launches", ok and torch.equal(y, y2))
        # an independent bound: each group's columns against fp64 of its own inputs (bf16 out: the rounding of the sum)
        wdq = dequant_fp8(sh.get(pre + "attn.wo_a.weight").to(dev), ws)
        refg = torch.cat([ref_linear64(q8[:, g * gk:(g + 1) * gk].contiguous(), s[:, g * gk // 32:(g + 1) * gk // 32].contiguous(), wdq[g * (n // groups):(g + 1) * (n // groups)])
                          for g in range(groups)], dim=1)
        same = (y.float() == refg.to(torch.bfloat16).float()).double().mean().item()
        check(f"attn.wo_a (fp8 twin) dec ks={ks}: grouped bf16 out rounds as the fp64 sum", same > 0.995, f"{same:.5f} equal")


def rel(a, b):
    return ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def check_dec_slabs(lib, name, fmt, q, s, w, ws, n, k, y_dtype=2):
    """G0.1 (docs/UNIFIED-DECODE-IMPROVEMENTS.md): past 64 rows mx.decDenseMulti runs slabs of 64 rows, each launch on
    its slab's rows (q + r0 ldq, s + r0 lds, y + r0 ldy). A row's bits must be the same at R = 1, inside R = 64 and as the
    remainder of R = 65 (and inside the first slab of R = 80)."""
    for ks, cs in DEC_SPLITS:
        def slabs(m, r0=0):
            return torch.cat([run_dec(lib, fmt, q[r0 + a:r0 + min(a + 64, m)], s[r0 + a:r0 + min(a + 64, m)] if s is not None else None,
                                      w, ws, n, k, ks, cs, y_dtype=y_dtype) for a in range(0, m, 64)])
        one = {i: run_dec(lib, fmt, q[i:i + 1], s[i:i + 1] if s is not None else None, w, ws, n, k, ks, cs, y_dtype=y_dtype) for i in (0, 63, 64, 79)}
        r64 = slabs(64)
        r65 = slabs(65)
        r80 = slabs(80)
        r64b = slabs(64, 1)  # rows 1 .. 64 in one launch: row 64 at position 63
        ok = (torch.equal(r65[:64], r64) and torch.equal(r80[:64], r64) and torch.equal(r65[64:65], one[64]) and
              torch.equal(r64[0:1], one[0]) and torch.equal(r64[63:64], one[63]) and torch.equal(r64b[63:64], one[64]) and
              torch.equal(r80[79:80], one[79]) and torch.equal(r80[64:65], one[64]))
        check(f"{name} dec {ks}x{cs}: 64-row slabs (R = 1, 64, 65, 80): a row's bits at every width", ok)


def check_gemm(lib, name, w, ws, fmt, wdq):
    n = w.shape[0]
    k = wdq.shape[1]
    torch.manual_seed(1)
    x = torch.randn(130, k, device=dev, dtype=torch.bfloat16) * 3
    q, s = ref_act_quant(x)
    q = q.view(torch.uint8).contiguous()
    ref = ref_linear64(q.view(torch.float8_e4m3fn), s, wdq)
    base = None
    for splits in (1, 4):
        for mt in (1, 2, 4, 8):
            y = run_mx(lib, q, s, w, ws, fmt, n, k, splits, mt)
            r = rel(y, ref)
            check(f"{name} splits={splits} mt={mt}: vs fp64", r < 2e-6, f"max rel {r:.2e}")
            if splits == 1:
                if base is None:
                    base = y
                else:
                    check(f"{name} splits=1 mt={mt}: same bits as mt=1", torch.equal(y, base))
    # the decode GEMV (a decode launch's few rows): the same bound, and each row its own bits
    xs = 48
    yv = run_mx(lib, q[:xs], s[:xs], w, ws, fmt, n, k, 1, -1)
    r = rel(yv, ref[:xs])
    check(f"{name} GEMV: vs fp64", r < 2e-6, f"max rel {r:.2e}")
    alone = torch.cat([run_mx(lib, q[i:i + 1], s[i:i + 1], w, ws, fmt, n, k, 1, -1) for i in (0, 17, 47)])
    check(f"{name} GEMV: a row alone equals its batch row", torch.equal(alone, yv[[0, 17, 47]]))
    rows = torch.tensor([47, 5, 5, 30], dtype=torch.int32, device=dev)
    g = run_mx(lib, q, s, w, ws, fmt, n, k, 1, -1, rows=rows, m=4)
    check(f"{name} GEMV: gathered rows equal their batch rows", torch.equal(g, yv[rows.long()]))
    # the decode swap-AB MMA: the same bound, and each row its own bits
    ya = run_mx(lib, q[:xs], s[:xs], w, ws, fmt, n, k, 1, -2)
    r = rel(ya, ref[:xs])
    check(f"{name} swap-AB: vs fp64", r < 2e-6, f"max rel {r:.2e}")
    alone = torch.cat([run_mx(lib, q[i:i + 1], s[i:i + 1], w, ws, fmt, n, k, 1, -2) for i in (0, 17, 47)])
    check(f"{name} swap-AB: a row alone equals its batch row", torch.equal(alone, ya[[0, 17, 47]]))
    g = run_mx(lib, q, s, w, ws, fmt, n, k, 1, -2, rows=rows, m=4)
    check(f"{name} swap-AB: gathered rows equal their batch rows", torch.equal(g, ya[rows.long()]))
    # the decode tensor-core kernels (tf_dsn_dec_*): the bound at each K split (warps a tile x CTAs), and each row its
    # own bits (48 rows run as 6 row groups, a row alone as 1; gathered rows)
    for ks, cs in DEC_SPLITS:
        yd = run_dec(lib, fmt, q[:xs], s[:xs], w, ws, n, k, ks, cs)
        r = rel(yd, ref[:xs])
        check(f"{name} dec {ks}x{cs}: vs fp64", r < 2e-6, f"max rel {r:.2e}")
        alone = torch.cat([run_dec(lib, fmt, q[i:i + 1], s[i:i + 1], w, ws, n, k, ks, cs) for i in (0, 17, 47)])
        check(f"{name} dec {ks}x{cs}: a row alone equals its batch row", torch.equal(alone, yd[[0, 17, 47]]))
        g = run_dec(lib, fmt, q, s, w, ws, n, k, ks, cs, rows=torch.tensor([47, 5, 5, 30], dtype=torch.int32, device=dev), m=4)
        check(f"{name} dec {ks}x{cs}: gathered rows equal their batch rows", torch.equal(g, yd[[47, 5, 5, 30]]))
        y3 = run_dec(lib, fmt, q[:xs], s[:xs], w, ws, n, k, ks, cs, multi=3)
        check(f"{name} dec {ks}x{cs}: three in one launch equal it alone", torch.equal(y3, torch.cat([yd, yd, yd])))
    check_dec_slabs(lib, name, fmt, q, s, w, ws, n, k)
    check_dec_slabs(lib, name + " bf16 out", fmt, q, s, w, ws, n, k, y_dtype=1)
    # row-invariance: each row alone, and rows gathered through a rows table, equal the batch's rows
    for splits in (1, 4):
        full = run_mx(lib, q, s, w, ws, fmt, n, k, splits, 2)
        alone = torch.cat([run_mx(lib, q[i:i + 1], s[i:i + 1], w, ws, fmt, n, k, splits, 1) for i in (0, 17, 129)])
        check(f"{name} splits={splits}: a row alone equals its batch row", torch.equal(alone, full[[0, 17, 129]]))
        rows = torch.tensor([129, 5, 5, 64], dtype=torch.int32, device=dev)
        g = run_mx(lib, q, s, w, ws, fmt, n, k, splits, 1, rows=rows, m=4)
        check(f"{name} splits={splits}: gathered rows equal their batch rows", torch.equal(g, full[rows.long()]))


def code(split, sp):
    """dsn_experts' split argument: 1 the MMA tiles, sp split K, -1 the GEMV, -2 swap-AB, -10 - ks the dec kernels."""
    return {"sab": -2, "gemv": -1, "dec1": -11, "dec2": -12, "dec4": -14}.get(split, sp if split is True else 1)


def splits_of(k, n, step=32):
    """mx.zig splits() / bfSplits(): a decode launch's K splits from its shape alone."""
    kb, ctas, s = k // step, (n // 32 + 3) // 4, 1
    while ctas * s < 96 and s < 16 and kb // (s * 2) >= 8:
        s *= 2
    return s


# ---------------------------------------------------------------- 3. the experts
def check_experts(lib, sh, layer, n_exp=24):
    pre = f"layers.{layer}.ffn.experts."
    torch.manual_seed(2)
    E = n_exp
    w1 = [sh.get(f"{pre}{e}.w1.weight").to(dev) for e in range(E)]
    w3 = [sh.get(f"{pre}{e}.w3.weight").to(dev) for e in range(E)]
    w2 = [sh.get(f"{pre}{e}.w2.weight").to(dev) for e in range(E)]
    s1 = [sh.get(f"{pre}{e}.w1.scale").to(dev) for e in range(E)]
    s3 = [sh.get(f"{pre}{e}.w3.scale").to(dev) for e in range(E)]
    s2 = [sh.get(f"{pre}{e}.w2.scale").to(dev) for e in range(E)]
    I_, D = w1[0].shape[0], w1[0].shape[1] * 2
    N = w2[0].shape[0]
    table = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    # the pointer tables live as long as the checks (a temporary would be freed before the launch)
    t1, t3, ts1, ts3, t2, ts2 = table(w1), table(w3), table(s1), table(s3), table(w2), table(s2)
    for R, S in ((1, 6), (5, 6), (70, 6)):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, device=dev)[:S] for _ in range(R)]).to(torch.int32)
        if R > 1:
            pick[1, 2] = E + 5  # another node's expert: skipped
        wts = torch.rand(R, S, device=dev, dtype=torch.float32) * 1.5
        cols = torch.arange(0, I_, 32, dtype=torch.int32, device=dev)
        RS = R * S
        scr = dict(q=torch.empty(R, D, dtype=torch.uint8, device=dev), s=torch.empty(R, D // 32, device=dev),
                   cnt=torch.empty(E, dtype=torch.int32, device=dev), off=torch.empty(E + 1, dtype=torch.int32, device=dev),
                   fill=torch.empty(E, dtype=torch.int32, device=dev), act=torch.empty(E, dtype=torch.int32, device=dev),
                   nact=torch.empty(1, dtype=torch.int32, device=dev), ent=torch.empty(RS, dtype=torch.int32, device=dev),
                   erow=torch.empty(RS, dtype=torch.int32, device=dev),
                   h=torch.empty(RS, I_, dtype=torch.bfloat16, device=dev),
                   hq=torch.empty(RS, I_, dtype=torch.uint8, device=dev), hs=torch.empty(RS, I_ // 32, device=dev),
                   out=torch.zeros(RS, N, device=dev), y=torch.empty(R, N, device=dev))
        gs, ds = splits_of(D, I_), splits_of(I_, N)
        z = torch.empty(max(gs * 2 * RS * I_, ds * RS * N), device=dev)
        res, sres, vres, ares, d1res, d2res, d4res = {}, {}, {}, {}, {}, {}, {}
        for mt in (1, 2, 4):
            for split, into in ((False, res), (True, sres), ("gemv", vres), ("sab", ares), ("dec1", d1res), ("dec2", d2res), ("dec4", d4res)):
                scr["out"].zero_()
                assert lib.dsn_experts(ptr(x), R, D, ptr(pick), ptr(wts), S, E, ptr(t1), ptr(t3),
                                       ptr(ts1), ptr(ts3), w1[0].shape[1], s1[0].shape[1], ptr(t2),
                                       ptr(ts2), w2[0].shape[1], s2[0].shape[1], 1, I_, ptr(cols), I_ // 32, N,
                                       F(10.0), *[ptr(scr[k]) for k in ("q", "s", "cnt", "off", "fill", "act", "nact", "ent",
                                                                         "erow", "h", "hq", "hs", "out", "y")], mt,
                                       code(split, gs), code(split, ds), ptr(z)) == 0
                into[mt] = scr["y"].clone()
        # reference: MoE.forward's routed part, expert ids ascending, each expert's arithmetic as Expert.forward
        q, s = ref_act_quant(x)
        y = torch.zeros(R, N, dtype=torch.float32, device=dev)
        hmax = 0.0
        for e in range(E):
            idx, top = torch.where(pick == e)
            if idx.numel() == 0:
                continue
            gate = ref_linear64(q[idx], s[idx], dequant_fp4(w1[e], s1[e])).to(torch.bfloat16).float()
            up = ref_linear64(q[idx], s[idx], dequant_fp4(w3[e], s3[e])).to(torch.bfloat16).float()
            up = up.clamp(-10, 10)
            gate = gate.clamp(max=10)
            h = (torch.nn.functional.silu(gate) * up * wts[idx, top, None]).to(torch.bfloat16)
            hq, hs = ref_act_quant(h)
            o = ref_linear64(hq, hs, dequant_fp4(w2[e], s2[e])).to(torch.bfloat16).float()
            y[idx] += o
            hmax = max(hmax, o.abs().max().item())
        r = rel(res[1], y.double())
        # bf16 roundings of g / u / h make single elements flip by one bf16 ulp against the fp64-summed reference:
        # the bound is a few bf16 ulps of the largest term
        check(f"experts R={R}: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
        close = ((res[1] - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
        check(f"experts R={R}: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        check(f"experts R={R}: mt=2 / mt=4 same bits as mt=1", torch.equal(res[1], res[2]) and torch.equal(res[1], res[4]))
        # split K (decode launches): the same bounds against the reference, and its own bits at every row tile
        r = rel(sres[1], y.double())
        check(f"experts R={R} split K {gs}/{ds}: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
        close = ((sres[1] - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
        check(f"experts R={R} split K: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        check(f"experts R={R} split K: mt=2 / mt=4 same bits as mt=1", torch.equal(sres[1], sres[2]) and torch.equal(sres[1], sres[4]))
        # the decode GEMV: the same bounds
        r = rel(vres[1], y.double())
        check(f"experts R={R} GEMV: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
        close = ((vres[1] - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
        check(f"experts R={R} GEMV: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        r = rel(ares[1], y.double())
        check(f"experts R={R} swap-AB: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
        close = ((ares[1] - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
        check(f"experts R={R} swap-AB: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        # Track C (TF_DS_EXPERT_EXIT): the decode kernels' rows past the active experts leaving at once give the
        # same bits (R = 5: 24 grid rows for fewer active experts)
        lib.dsn_set_expert_exit(1)
        for nm_, dres in (("dec1", d1res), ("dec2", d2res), ("dec4", d4res)):
            scr["out"].zero_()
            assert lib.dsn_experts(ptr(x), R, D, ptr(pick), ptr(wts), S, E, ptr(t1), ptr(t3),
                                   ptr(ts1), ptr(ts3), w1[0].shape[1], s1[0].shape[1], ptr(t2),
                                   ptr(ts2), w2[0].shape[1], s2[0].shape[1], 1, I_, ptr(cols), I_ // 32, N,
                                   F(10.0), *[ptr(scr[k]) for k in ("q", "s", "cnt", "off", "fill", "act", "nact", "ent",
                                                                     "erow", "h", "hq", "hs", "out", "y")], 1,
                                   code(nm_, gs), code(nm_, ds), ptr(z)) == 0
            na = int(scr["nact"].item())
            check(f"experts R={R} {nm_}: the early exit ({min(E, RS) - na} of {min(E, RS)} grid rows idle) gives the same bits", torch.equal(scr["y"], dres[1]))
        lib.dsn_set_expert_exit(0)
        for nm_, dres in (("dec1", d1res), ("dec2", d2res), ("dec4", d4res)):
            r = rel(dres[1], y.double())
            check(f"experts R={R} {nm_}: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
            close = ((dres[1] - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
            check(f"experts R={R} {nm_}: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        if R == 70:
            for split, ref_rows in ((False, res[1]), (True, sres[1]), ("gemv", vres[1]), ("sab", ares[1]), ("dec1", d1res[1]), ("dec2", d2res[1]), ("dec4", d4res[1])):
                alone = []
                for i in (0, 33, 69):
                    scr1 = {k: v for k, v in scr.items()}
                    y1 = torch.empty(1, N, device=dev)
                    out1 = torch.zeros(S, N, device=dev)
                    assert lib.dsn_experts(ptr(x[i:i + 1]), 1, D, ptr(pick[i:i + 1].contiguous()), ptr(wts[i:i + 1].contiguous()),
                                           S, E, ptr(t1), ptr(t3), ptr(ts1), ptr(ts3),
                                           w1[0].shape[1], s1[0].shape[1], ptr(t2), ptr(ts2), w2[0].shape[1],
                                           s2[0].shape[1], 1, I_, ptr(cols), I_ // 32, N, F(10.0),
                                           *[ptr(scr1[k]) for k in ("q", "s", "cnt", "off", "fill", "act", "nact", "ent",
                                                                    "erow", "h", "hq", "hs")], ptr(out1), ptr(y1), 1,
                                           code(split, gs), code(split, ds), ptr(z)) == 0
                    alone.append(y1)
                check(f"experts{' ' + str(split) if split else ''}: a row alone equals its batch row",
                      torch.equal(torch.cat(alone), ref_rows[[0, 33, 69]]))


def check_experts_pf(lib, sh, layer, n_exp=24):
    """the prefill grouped experts (tf_dsn_pf_gu / dn) against the tile kernels' bits and MoE.forward, the shared FP8
    expert (the last id) in every row's last slot, row counts up to several 64-row tiles an expert."""
    pre = f"layers.{layer}.ffn."
    torch.manual_seed(6)
    E = n_exp + 1
    get = lambda n: sh.get(pre + n).to(dev)
    w1 = [get(f"experts.{e}.w1.weight") for e in range(n_exp)] + [get("shared_experts.w1.weight").view(torch.uint8)]
    w3 = [get(f"experts.{e}.w3.weight") for e in range(n_exp)] + [get("shared_experts.w3.weight").view(torch.uint8)]
    w2 = [get(f"experts.{e}.w2.weight") for e in range(n_exp)] + [get("shared_experts.w2.weight").view(torch.uint8)]
    s1 = [get(f"experts.{e}.w1.scale") for e in range(n_exp)] + [get("shared_experts.w1.scale")]
    s3 = [get(f"experts.{e}.w3.scale") for e in range(n_exp)] + [get("shared_experts.w3.scale")]
    s2 = [get(f"experts.{e}.w2.scale") for e in range(n_exp)] + [get("shared_experts.w2.scale")]
    I_, D = w1[0].shape[0], w1[0].shape[1] * 2
    N = w2[0].shape[0]
    table = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    t1, t3, ts1, ts3, t2, ts2 = table(w1), table(w3), table(s1), table(s3), table(w2), table(s2)
    deq = [(dequant_fp4(w1[e], s1[e]), dequant_fp4(w3[e], s3[e]), dequant_fp4(w2[e], s2[e])) for e in range(n_exp)]
    deq.append((dequant_fp8(w1[-1].view(torch.float8_e4m3fn), s1[-1]), dequant_fp8(w3[-1].view(torch.float8_e4m3fn), s3[-1]),
                dequant_fp8(w2[-1].view(torch.float8_e4m3fn), s2[-1])))
    S = 7

    def run(x, pick, wts, pf, mt=2):
        R = x.shape[0]
        RS = R * S
        scr = dict(q=torch.empty(R, D, dtype=torch.uint8, device=dev), s=torch.empty(R, D // 32, device=dev),
                   cnt=torch.empty(E, dtype=torch.int32, device=dev), off=torch.empty(E + 1, dtype=torch.int32, device=dev),
                   fill=torch.empty(E, dtype=torch.int32, device=dev), act=torch.empty(E, dtype=torch.int32, device=dev),
                   nact=torch.empty(1, dtype=torch.int32, device=dev), ent=torch.empty(RS, dtype=torch.int32, device=dev),
                   erow=torch.empty(RS, dtype=torch.int32, device=dev), h=torch.empty(RS, I_, dtype=torch.bfloat16, device=dev),
                   hq=torch.empty(RS, I_, dtype=torch.uint8, device=dev), hs=torch.empty(RS, I_ // 32, device=dev),
                   out=torch.zeros(RS, N, device=dev), y=torch.empty(R, N, device=dev))
        assert lib.dsn_experts_pf(ptr(x), R, D, ptr(pick), ptr(wts), S, E, ptr(t1), ptr(t3), ptr(ts1), ptr(ts3),
                                  w1[0].shape[1], s1[0].shape[1], ptr(t2), ptr(ts2), w2[0].shape[1], s2[0].shape[1], I_, N,
                                  F(10.0), E - 1, D, D // 32, I_, I_ // 32,
                                  *[ptr(scr[k]) for k in ("q", "s", "cnt", "off", "fill", "act", "nact", "ent", "erow", "h",
                                                          "hq", "hs", "out", "y")], mt, pf) == 0
        return scr["y"].clone()

    for R in (1, 5, 70, 300):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        rows_ = []
        for i in range(R):
            p_ = torch.randperm(n_exp, device=dev)
            if R >= 70 and i < R // 2:  # one expert past a 64-row tile: 3 first, no row picks an expert twice
                p_ = torch.cat([torch.tensor([3], device=dev), p_[p_ != 3]])
            rows_.append(p_[:S - 1])
        routed = torch.stack(rows_).to(torch.int32)
        pick = torch.cat([routed, torch.full((R, 1), E - 1, dtype=torch.int32, device=dev)], 1).contiguous()
        if R > 1:
            pick[1, 2] = E + 5  # another node's expert: skipped
        wts = (torch.rand(R, S, device=dev, dtype=torch.float32) * 1.5).contiguous()
        old = run(x, pick, wts, 0)
        new = run(x, pick, wts, 1)
        check(f"prefill experts R={R}: same bits as the tile kernels", torch.equal(new, old))
        q, s = ref_act_quant(x)
        y = torch.zeros(R, N, dtype=torch.float32, device=dev)
        hmax = 0.0
        for e in range(E):
            idx, top = torch.where(pick == e)
            if idx.numel() == 0:
                continue
            g1, g3, g2 = deq[e]
            gate = ref_linear64(q[idx], s[idx], g1).to(torch.bfloat16).float().clamp(max=10)
            up = ref_linear64(q[idx], s[idx], g3).to(torch.bfloat16).float().clamp(-10, 10)
            h = (torch.nn.functional.silu(gate) * up * wts[idx, top, None]).to(torch.bfloat16)
            hq, hs = ref_act_quant(h)
            o = ref_linear64(hq, hs, g2).to(torch.bfloat16).float()
            y[idx] += o
            hmax = max(hmax, o.abs().max().item())
        r = rel(new, y.double())
        check(f"prefill experts R={R}: vs MoE.forward", r < 2e-2, f"max rel {r:.2e}")
        close = ((new - y).abs() <= 2 ** -6 * y.abs() + 1e-3 * hmax).float().mean().item()
        check(f"prefill experts R={R}: elements within a bf16 rounding", close > 0.999, f"{close:.5f}")
        if R == 300:
            alone = torch.cat([run(x[i:i + 1].contiguous(), pick[i:i + 1].contiguous(), wts[i:i + 1].contiguous(), 1) for i in (0, 151, 299)])
            check("prefill experts: a row alone equals its batch row", torch.equal(alone, new[[0, 151, 299]]))


# ---------------------------------------------------------------- 4. bf16 linears
def check_bf16(lib, sh, layer):
    torch.manual_seed(3)
    wo_a = sh.get(f"layers.{layer}.attn.wo_a.weight").to(dev)
    sc = sh.get(f"layers.{layer}.attn.wo_a.scale").to(dev)
    w = dequant_fp8(wo_a, sc)  # convert.py: wo_a dequantized to bf16
    wb = w.to(torch.bfloat16)
    G, R_, Dg = 8, 1024, wb.shape[1]
    M = 21
    x = torch.randn(M, G * Dg, device=dev).to(torch.bfloat16)
    y = torch.empty(M, G * R_, dtype=torch.bfloat16, device=dev)
    # group g: x[:, g * Dg:(g + 1) * Dg] . wb[g * R_:(g + 1) * R_]^T -> y[:, g * R_:]
    assert lib.dsn_bf16_gemm(ptr(x), G * Dg, Dg, ptr(wb), Dg, R_ * Dg, M, R_, Dg, G, ptr(y), G * R_, R_, 1, F(1.0), 0, 1, None) == 0
    ref = torch.einsum("mgd,grd->mgr", x.double().view(M, G, Dg), wb.double().view(G, R_, Dg)).reshape(M, G * R_)
    exact_bf16 = (y.double() == ref.to(torch.bfloat16).double()).float().mean().item()
    check("wo_a grouped bf16 einsum: rounds as the fp64 sum", exact_bf16 > 0.995, f"{exact_bf16:.5f} equal")
    y1 = torch.empty(1, G * R_, dtype=torch.bfloat16, device=dev)
    assert lib.dsn_bf16_gemm(ptr(x[5:6]), G * Dg, Dg, ptr(wb), Dg, R_ * Dg, 1, R_, Dg, G, ptr(y1), G * R_, R_, 1, F(1.0), 0, 1, None) == 0
    check("wo_a: a row alone equals its batch row", torch.equal(y1, y[5:6]))
    # split K (decode launches), one group at a time as mx.woA runs them
    sp = splits_of(Dg, R_, 16)
    z = torch.empty(sp * M * R_, device=dev)
    ys = torch.empty(M, G * R_, dtype=torch.bfloat16, device=dev)
    for g in range(G):
        assert lib.dsn_bf16_gemm(ptr(x[:, g * Dg:]), G * Dg, 0, ptr(wb[g * R_:]), Dg, 0, M, R_, Dg, 1, ptr(ys[:, g * R_:]),
                                 G * R_, 0, 1, F(1.0), 0, sp, ptr(z)) == 0
    exact_bf16 = (ys.double() == ref.to(torch.bfloat16).double()).float().mean().item()
    check(f"wo_a split K {sp}: rounds as the fp64 sum", exact_bf16 > 0.995, f"{exact_bf16:.5f} equal")
    y1s = torch.empty(1, R_, dtype=torch.bfloat16, device=dev)
    assert lib.dsn_bf16_gemm(ptr(x[5:6, 3 * Dg:]), G * Dg, 0, ptr(wb[3 * R_:]), Dg, 0, 1, R_, Dg, 1, ptr(y1s), R_, 0, 1,
                             F(1.0), 0, sp, ptr(z)) == 0
    check("wo_a split K: a row alone equals its batch row", torch.equal(y1s, ys[5:6, 3 * R_:4 * R_]))
    # the prefill bf16 GEMM: the tile kernel's bits at several row counts (wo_a's groups; the gate's fp32 logits)
    for M2 in (21, 300):
        x2 = torch.randn(M2, G * Dg, device=dev).to(torch.bfloat16)
        yo = torch.empty(M2, G * R_, dtype=torch.bfloat16, device=dev)
        yn = torch.empty(M2, G * R_, dtype=torch.bfloat16, device=dev)
        assert lib.dsn_bf16_gemm(ptr(x2), G * Dg, Dg, ptr(wb), Dg, R_ * Dg, M2, R_, Dg, G, ptr(yo), G * R_, R_, 1, F(1.0), 0, 1, None) == 0
        assert lib.dsn_pf_bf16(ptr(x2), G * Dg, Dg, ptr(wb), Dg, R_ * Dg, M2, R_, Dg, G, ptr(yn), G * R_, R_, 1, F(1.0), 0) == 0
        check(f"prefill bf16 GEMM wo_a M={M2}: same bits as the tile kernel", torch.equal(yo, yn))
        if M2 == 300:
            y1 = torch.empty(1, G * R_, dtype=torch.bfloat16, device=dev)
            assert lib.dsn_pf_bf16(ptr(x2[150:151]), G * Dg, Dg, ptr(wb), Dg, R_ * Dg, 1, R_, Dg, G, ptr(y1), G * R_, R_, 1, F(1.0), 0) == 0
            check("prefill bf16 GEMM: a row alone equals its batch row", torch.equal(y1, yn[150:151]))
    gw = sh.get(f"layers.{layer}.ffn.gate.weight").to(dev).to(torch.bfloat16)
    xg = torch.randn(300, gw.shape[1], device=dev).to(torch.bfloat16)
    lo = torch.empty(300, gw.shape[0], device=dev)
    ln = torch.empty(300, gw.shape[0], device=dev)
    K2 = gw.shape[1]
    assert lib.dsn_bf16_gemm(ptr(xg), K2, 0, ptr(gw), K2, 0, 300, gw.shape[0], K2, 1, ptr(lo), gw.shape[0], 0, 2, F(1.0), 0, 1, None) == 0
    assert lib.dsn_pf_bf16(ptr(xg), K2, 0, ptr(gw), K2, 0, 300, gw.shape[0], K2, 1, ptr(ln), gw.shape[0], 0, 2, F(1.0), 0) == 0
    check("prefill bf16 GEMM gate (fp32 logits): same bits as the tile kernel", torch.equal(lo, ln))
    # the decode tensor-core bf16 kernel, one group at a time (fp32 out: the head's; bf16 out: wo_a's), each K split
    ref3 = ref.view(M, G, R_)[:, 3]
    for ks, cs in DEC_SPLITS:
        yd = run_dec(lib, 2, x[:, 3 * Dg:4 * Dg].contiguous(), None, wb[3 * R_:4 * R_].contiguous(), None, R_, Dg, ks, cs)
        r = rel(yd, ref3)
        check(f"bf16 dec {ks}x{cs}: vs fp64", r < 2e-6, f"max rel {r:.2e}")
        yb = run_dec(lib, 2, x[:, 3 * Dg:4 * Dg].contiguous(), None, wb[3 * R_:4 * R_].contiguous(), None, R_, Dg, ks, cs, y_dtype=1)
        exact_bf16 = (yb.double() == ref3.to(torch.bfloat16).double()).float().mean().item()
        check(f"bf16 dec {ks}x{cs}: bf16 out rounds as the fp64 sum", exact_bf16 > 0.995, f"{exact_bf16:.5f} equal")
        alone = torch.cat([run_dec(lib, 2, x[i:i + 1, 3 * Dg:4 * Dg].contiguous(), None, wb[3 * R_:4 * R_].contiguous(), None, R_, Dg, ks, cs) for i in (0, 5, 20)])
        check(f"bf16 dec {ks}x{cs}: a row alone equals its batch row", torch.equal(alone, yd[[0, 5, 20]]))
        # the epilogue (weights_proj: bf16(acc) * scale)
        ye = run_dec(lib, 2, x[:, 3 * Dg:4 * Dg].contiguous(), None, wb[3 * R_:4 * R_].contiguous(), None, R_, Dg, ks, cs, scale=0.125, round_first=1)
        check(f"bf16 dec {ks}x{cs}: round_first / scale epilogue", torch.equal(ye, yd.to(torch.bfloat16).float() * 0.125))


def main():
    root = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    lib = build()
    lib.dsn_act_quant.argtypes = [P, I, L, I, I, P, L, P, L, P, L]
    lib.dsn_dec_quant.argtypes = [P, I, L, I, I, P, L, P, L]
    lib.dsn_mx_gemm.argtypes = [P, P, P, L, L, P, P, L, L, I, I, I, I, I, I, P, P, L, I, I]
    lib.dsn_experts.argtypes = [P, I, I, P, P, I, I, P, P, P, P, L, L, P, P, L, L, I, I, P, I, I, F] + [P] * 14 + [I, I, I, P]
    lib.dsn_bf16_gemm.argtypes = [P, L, L, P, L, L, I, I, I, I, P, L, L, I, F, I, I, P]
    lib.dsn_pf_gemm.argtypes = [P, P, L, L, P, P, L, L, I, I, I, I, P, L, I, I]
    lib.dsn_pf_bf16.argtypes = [P, L, L, P, L, L, I, I, I, I, P, L, L, I, F, I]
    lib.dsn_experts_pf.argtypes = [P, I, I, P, P, I, I, P, P, P, P, L, L, P, P, L, L, I, I, F, I, L, L, L, L] + [P] * 14 + [I, I]
    lib.dsn_dec_dense.argtypes = [I, P, P, P, L, L, P, P, L, L, I, I, I, I, I, P, P, P, L, I, I, F, I, I]
    lib.dsn_set_expert_exit.argtypes = [I]
    lib.dsn_dec_dense_g.argtypes = [I, P, P, P, L, L, P, P, L, L, I, I, I, I, P, P, P, L, I, I, I, L, L]
    sh = Shards(root, sys.argv[3] if len(sys.argv) > 3 else None)
    check_act_quant(lib)
    pre = f"layers.{layer}."
    for name in ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2"):
        w = sh.get(pre + name + ".weight").to(dev)
        ws = sh.get(pre + name + ".scale").to(dev)
        check_gemm(lib, name + " (fp8)", w.view(torch.uint8), ws, 0, dequant_fp8(w, ws))
        check_pf_gemm(lib, name + " (fp8)", w.view(torch.uint8), ws, dequant_fp8(w, ws))
    for name in ("ffn.experts.7.w1", "ffn.experts.7.w2"):
        w = sh.get(pre + name + ".weight").to(dev)
        ws = sh.get(pre + name + ".scale").to(dev)
        check_gemm(lib, name + " (mxfp4)", w, ws, 1, dequant_fp4(w, ws))
    # a 2D node's down: half the intermediate (K 1152, rows of 576 bytes: the dec kernels' short last stage)
    w = sh.get(pre + "ffn.experts.7.w2.weight").to(dev)[:, :576].contiguous()
    ws = sh.get(pre + "ffn.experts.7.w2.scale").to(dev)[:, :36].contiguous()
    check_gemm(lib, "ffn.experts.7.w2[:, :1152] (mxfp4)", w, ws, 1, dequant_fp4(w, ws))
    w = sh.get(pre + "ffn.shared_experts.w2.weight").to(dev)[:, :1152].contiguous()
    ws = sh.get(pre + "ffn.shared_experts.w2.scale").to(dev)[:, :36].contiguous()
    check_gemm(lib, "ffn.shared_experts.w2[:, :1152] (fp8)", w.view(torch.uint8), ws, 0, dequant_fp8(w, ws))
    check_experts(lib, sh, layer)
    check_experts_pf(lib, sh, layer)
    check_bf16(lib, sh, layer)
    # the router (ffn.gate, bf16, fp32 logits): the drafter's pass of 13+ streams reaches it past 64 rows (G0.1)
    gw = sh.get(pre + "ffn.gate.weight").to(dev).contiguous()
    torch.manual_seed(3)
    xg = (torch.randn(80, gw.shape[1], device=dev) * 2).to(torch.bfloat16)
    check_dec_slabs(lib, "ffn.gate (bf16)", 2, xg, None, gw, None, gw.shape[0], gw.shape[1])
    check_dec_slabs_prod(lib, sh, layer)
    print(f"{len(failures)} failed" if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
