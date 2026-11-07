"""Checks of NVFP4 in zig/kernels/cuda/dsv41/native.cu (tf_dsn_nv_quant, the decode and prefill expert kernels with fmt 2)
on real tensors of nvidia/DeepSeek-V4.1-Flash-1017-NVFP4. Runs in nvcr.io/nvidia/pytorch:26.07-py3 on a GB10:

    python check_nvfp4.py <NVFP4 checkpoint dir> [layer] [n experts] [source FP8 checkpoint dir]

  1. the scale pairs: a weight's E4M3 scales per 16 are equal in pairs (the kernels read one a block of 32);
  2. the weights dequantize to what the checkpoint's card says: code * scale * weight_scale_2 (and, given the source
     checkpoint, exactly to its MXFP4 weights);
  3. tf_dsn_nv_quant: codes and scales equal nvfp4q.cuh's rule bit for bit, by expert;
  4. the grouped experts, decode and prefill kernels, against MoE.forward's arithmetic in fp64 with NVFP4's quantization
     (a bf16 ulp of h may flip one quantized input, so a relative bound on the outputs, and h compared exactly in ulps);
  5. a row's bits never depend on the launch on the decode kernels (rows alone against rows together);
  6. the outputs sit near W4A8's (the source's FP8 activations against the same weights): alpha and g are the right way
     round, the quantization only coarser. Exit status 0 when every check holds.
"""
import ctypes
import json
import os
import struct
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_native import ref_act_quant, ptr  # noqa: E402

dev = torch.device("cuda")
TAB = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
P, L, I, F = ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_float
FAILS = []


def ok(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class Nv:
    def __init__(self, root):
        self.root = root
        self.map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.h:
            with open(os.path.join(self.root, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.h[f] = (json.loads(fh.read(n)), 8 + n)
        hh, base = self.h[f]
        e = hh[name]
        a, b = e["data_offsets"]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            raw = bytearray(fh.read(b - a))
        dt = {"U8": torch.uint8, "I8": torch.uint8, "F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8, "F32": torch.float32}[e["dtype"]]
        return torch.frombuffer(raw, dtype=dt).reshape(e["shape"])


def e4m3(b):
    return b.view(torch.float8_e4m3fn).to(torch.float64)


def nibbles(w):
    b = w.to(torch.int64)
    return torch.stack([b & 0xF, (b >> 4) & 0xF], -1).view(w.shape[0], -1)


def dq(w, ws, g):
    """code * e4m3 scale * global, fp64 [N, K]."""
    return TAB.to(w.device)[nibbles(w)] * e4m3(ws).repeat_interleave(16, 1) * float(g)


def ref_quant(x, g):
    """nvfp4q.cuh: per 16, sf = e4m3(g * amax / 6), codes e2m1(x * g / sf): codes [M, K] (ints), scale bytes [M, K/16]."""
    v = x.to(torch.bfloat16).float()
    m, k = v.shape
    b = v.view(m, k // 16, 16)
    amax = b.abs().amax(-1)
    sv = g.view(m, 1) * (amax * torch.tensor(1 / 6, dtype=torch.float32, device=v.device))
    sb = sv.clamp(max=448).to(torch.float8_e4m3fn)
    sf = sb.to(torch.float32)
    mul = torch.where(sf != 0, g.view(m, 1) / sf, torch.zeros_like(sf))
    a = (b * mul[:, :, None]).abs()
    c = torch.zeros_like(a, dtype=torch.int64)
    c[a > .25] = 1
    c[a >= .75] = 2
    c[a > 1.25] = 3
    c[a >= 1.75] = 4
    c[a > 2.5] = 5
    c[a >= 3.5] = 6
    c[a > 5] = 7
    neg = (b * mul[:, :, None]) < 0
    c = torch.where(neg & (c != 0), c | 8, c)
    return c.view(m, k), sb.view(torch.uint8)


def to_bf(v):
    return v.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def main():
    root = sys.argv[1]
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_exp = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    src = sys.argv[4] if len(sys.argv) > 4 else None
    so = "/tmp/dsn_nv_host.so"
    subprocess.check_call(["nvcc", "-arch=sm_121a", "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", "-o", so, os.path.join(HERE, "native_host.cu")])
    lib = ctypes.CDLL(so)
    lib.dsn_nv_quant.argtypes = [P, L, P, I, I, I, I, P, I, P, P]
    lib.dsn_nv_experts.argtypes = [P, I, I, P, P, I, I] + [P] * 11 + [I, I, F] + [P] * 14 + [I]
    nv = Nv(root)
    D, Iw, N = 5120, 2304, 5120
    p = f"layers.{layer}.ffn.experts"
    W = {k: [] for k in ("w1", "w3", "w2", "s1", "s3", "s2")}
    inp, glb = {}, {}
    for e in range(n_exp):
        for name in ("w1", "w3", "w2"):
            pre = f"{p}.{e}.{name}"
            w = nv.get(pre + ".weight")
            ws = nv.get(pre + ".weight_scale")
            eq = (ws[:, 0::2] == ws[:, 1::2]).all().item()
            if not eq:
                ok(False, f"{pre}: scales not equal in pairs")
            W[name].append(w.to(dev))
            W["s" + name[1]].append(ws.to(dev))
            inp[(e, name)] = nv.get(pre + ".input_scale").item()
            glb[(e, name)] = nv.get(pre + ".weight_scale_2").item()
    ok(all(inp[(e, "w1")] == inp[(e, "w3")] for e in range(n_exp)), "gate and up share an input_scale")
    ok(not FAILS, f"weight_scale pairs equal in {n_exp} experts of layer {layer}")
    if src:
        from check_native import Shards, dequant_fp4
        sh = Shards(src)
        for e in (0, n_exp - 1):
            for name in ("w1", "w2", "w3"):
                pre = f"{p}.{e}.{name}"
                a = dq(W[name][e], W["s" + name[1]][e], glb[(e, name)])
                b = dequant_fp4(sh.get(pre + ".weight").to(dev), sh.get(pre + ".scale").to(dev))
                ok(torch.equal(a, b.to(a.device)), f"{pre}: code * scale * weight_scale_2 == the source's MXFP4 weight")

    # the kernels' per-32 scale bytes
    def tab(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype=torch.uint64, device=dev)
    keep = dict(W)
    s32 = {k: [s[:, 0::2].contiguous() for s in W[k]] for k in ("s1", "s3", "s2")}
    tabs = {k: tab(v) for k, v in W.items() if k[0] == "w"}
    tabs.update({k: tab(v) for k, v in s32.items()})
    g13 = torch.tensor([1 / inp[(e, "w1")] for e in range(n_exp)], dtype=torch.float32, device=dev)
    g2 = torch.tensor([1 / inp[(e, "w2")] for e in range(n_exp)], dtype=torch.float32, device=dev)
    al = {n: torch.tensor([inp[(e, n)] * glb[(e, n)] for e in range(n_exp)], dtype=torch.float32, device=dev) for n in ("w1", "w3", "w2")}
    print(f"input_scale w1 {inp[(0, 'w1')]:.6g} w2 {inp[(0, 'w2')]:.6g} weight_scale_2 {glb[(0, 'w1')]:.6g} alpha {al['w1'][0].item():.6g}", flush=True)

    torch.manual_seed(1)
    R, S = 40, 6
    x = (torch.randn(R, D, device=dev) * 0.02 * torch.logspace(-1, 1, D, device=dev)[torch.randperm(D, device=dev)]).to(torch.bfloat16)
    x[5, :16] = 0
    pick = torch.stack([torch.randperm(n_exp, device=dev)[:S] for _ in range(R)]).to(torch.int32)
    wts = torch.rand(R, S, device=dev, dtype=torch.float32) * 0.5 + 0.05

    # 3. the quant
    RS = R * S
    nq = torch.empty(RS, D // 2, dtype=torch.uint8, device=dev)
    ns = torch.empty(RS, D // 16, dtype=torch.uint8, device=dev)
    assert lib.dsn_nv_quant(ptr(x), D, ptr(pick), S, 0, RS, D, ptr(g13), n_exp, ptr(nq), ptr(ns)) == 0
    entry_e = pick.view(-1).long()
    rx = x.repeat_interleave(S, 0)
    rc, rs = ref_quant(rx, g13[entry_e])
    ok(torch.equal(nibbles(nq), rc), "tf_dsn_nv_quant codes == nvfp4q.cuh's rule")
    ok(torch.equal(ns, rs), "tf_dsn_nv_quant scales == nvfp4q.cuh's rule")

    # 4. the grouped experts against fp64
    limit = 10.0
    def ref_experts():
        xhat = TAB.to(dev)[rc] * e4m3(rs).repeat_interleave(16, 1)
        h = torch.zeros(RS, Iw, dtype=torch.float64, device=dev)
        for e in range(n_exp):
            idx = (entry_e == e).nonzero()[:, 0]
            if idx.numel() == 0:
                continue
            w1 = dq(W["w1"][e], W["s1"][e], 1.0)
            w3 = dq(W["w3"][e], W["s3"][e], 1.0)
            ga = to_bf((xhat[idx] @ w1.T) * al["w1"][e].double())
            up = to_bf((xhat[idx] @ w3.T) * al["w3"][e].double())
            up = up.clamp(-limit, limit)
            ga = ga.clamp(max=limit)
            hv = (ga / (1 + torch.exp(-ga))) * up
            h[idx] = to_bf(hv * wts.view(-1)[idx].double()[:, None])
        hc, hs = ref_quant(h.to(torch.float32), g2[entry_e])
        hhat = TAB.to(dev)[hc] * e4m3(hs).repeat_interleave(16, 1)
        out = torch.zeros(RS, N, dtype=torch.float64, device=dev)
        for e in range(n_exp):
            idx = (entry_e == e).nonzero()[:, 0]
            if idx.numel() == 0:
                continue
            w2 = dq(W["w2"][e], W["s2"][e], 1.0)
            out[idx] = to_bf((hhat[idx] @ w2.T) * al["w2"][e].double())
        y = out.view(R, S, N).sum(1)
        return h, y
    rh, ry = ref_experts()

    def run(px, ppick, pw, path):
        r = px.shape[0]
        rs_ = r * S
        cnt = torch.empty(n_exp, dtype=torch.int32, device=dev)
        off = torch.empty(n_exp + 1, dtype=torch.int32, device=dev)
        fill = torch.empty(n_exp, dtype=torch.int32, device=dev)
        act = torch.empty(n_exp, dtype=torch.int32, device=dev)
        nact = torch.empty(1, dtype=torch.int32, device=dev)
        ent = torch.empty(rs_, dtype=torch.int32, device=dev)
        erow = torch.empty(rs_, dtype=torch.int32, device=dev)
        q = torch.empty(rs_, D // 2, dtype=torch.uint8, device=dev)
        s = torch.empty(rs_, D // 16, dtype=torch.uint8, device=dev)
        h = torch.empty(rs_, Iw, dtype=torch.bfloat16, device=dev)
        hq = torch.empty(rs_, Iw // 2, dtype=torch.uint8, device=dev)
        hs = torch.empty(rs_, Iw // 16, dtype=torch.uint8, device=dev)
        out = torch.empty(rs_, N, dtype=torch.float32, device=dev)
        y = torch.empty(r, N, dtype=torch.float32, device=dev)
        assert lib.dsn_nv_experts(ptr(px), r, D, ptr(ppick), ptr(pw), S, n_exp, ptr(tabs["w1"]), ptr(tabs["w3"]), ptr(tabs["s1"]), ptr(tabs["s3"]),
                                  ptr(tabs["w2"]), ptr(tabs["s2"]), ptr(g13), ptr(g2), ptr(al["w1"]), ptr(al["w3"]), ptr(al["w2"]),
                                  Iw, N, limit, ptr(cnt), ptr(off), ptr(fill), ptr(act), ptr(nact), ptr(ent), ptr(erow), ptr(q), ptr(s),
                                  ptr(h), ptr(hq), ptr(hs), ptr(out), ptr(y), path) == 0
        return h, y

    for path, name in ((0, "decode"), (1, "prefill")):
        h, y = run(x, pick, wts, path)
        dh = (h.to(torch.float64) - rh).abs()
        ulp = rh.abs().clamp_min(1e-30) * 2.0 ** -7
        ok((dh / ulp).max().item() <= 2.01, f"{name}: h within 2 bf16 ulps of fp64 (max {(dh / ulp).max().item():.3f} ulp, {100 * (dh > 0).double().mean().item():.4f} % differ)")
        rel = (y.double() - ry).norm() / ry.norm()
        ok(rel.item() < 2e-3, f"{name}: routed sum within 0.2 % of fp64 (relative error {rel.item():.2e})")
        ok(torch.isfinite(y).all().item() and y.abs().max().item() > 0, f"{name}: outputs finite and nonzero (max {y.abs().max().item():.4g})")

    # 5. a row alone
    h0, y0 = run(x, pick, wts, 0)
    same = True
    for r in (0, 5, 17, R - 1):
        hr, yr = run(x[r:r + 1], pick[r:r + 1], wts[r:r + 1], 0)
        same = same and torch.equal(yr[0], y0[r])
    ok(same, "decode kernels: a row's output has the same bits alone and with the others")
    hb, yb = run(x, pick, wts, 0)
    ok(torch.equal(yb, y0), "decode kernels: repeatable")

    # 6. W4A8 on the same weights
    q8, s8 = ref_act_quant(x)
    xh8 = (q8.to(torch.float64).view(R, D // 32, 32) * s8.double()[:, :, None]).view(R, D).repeat_interleave(S, 0)
    h8 = torch.zeros(RS, Iw, dtype=torch.float64, device=dev)
    for e in range(n_exp):
        idx = (entry_e == e).nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        ga = to_bf(xh8[idx] @ dq(W["w1"][e], W["s1"][e], glb[(e, "w1")]).T).clamp(max=limit)
        up = to_bf(xh8[idx] @ dq(W["w3"][e], W["s3"][e], glb[(e, "w3")]).T).clamp(-limit, limit)
        h8[idx] = to_bf((ga / (1 + torch.exp(-ga))) * up * wts.view(-1)[idx].double()[:, None])
    q8h, s8h = ref_act_quant(h8.to(torch.float32))
    hh8 = (q8h.to(torch.float64).view(RS, Iw // 32, 32) * s8h.double()[:, :, None]).view(RS, Iw)
    o8 = torch.zeros(RS, N, dtype=torch.float64, device=dev)
    for e in range(n_exp):
        idx = (entry_e == e).nonzero()[:, 0]
        if idx.numel():
            o8[idx] = to_bf(hh8[idx] @ dq(W["w2"][e], W["s2"][e], glb[(e, "w2")]).T)
    y8 = o8.view(R, S, N).sum(1)
    cos = torch.nn.functional.cosine_similarity(y0.double(), y8, dim=1)
    ratio = (y0.double().norm(dim=1) / y8.norm(dim=1))
    print(f"W4A8 vs W4A4 on the same weights: cosine min {cos.min().item():.4f} mean {cos.mean().item():.4f}; norm ratio {ratio.min().item():.3f} .. {ratio.max().item():.3f}", flush=True)
    ok(cos.mean().item() > 0.8 and 0.5 < ratio.mean().item() < 2.0, "the 4-bit activations stay near the 8-bit ones (alpha and g the right way round)")
    del keep
    print("FAILED: " + "; ".join(FAILS) if FAILS else "all NVFP4 checks hold", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
