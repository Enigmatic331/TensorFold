"""Check a native engine's layer dumps (TF_DS_DUMP) of node 0 (2D split, parity, TP2 rank 0 of pair 0) against
DeepSeek's reference arithmetic, step by step from each step's own dumped input:

    python check_dump.py <checkpoint dir> <dump dir> [layer] [index json] [peer dump dir of node 2]

Each line: the step, the largest relative difference, and the share of elements equal or within a bf16 rounding.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_native import Shards, ref_act_quant, dequant_fp8, dequant_fp4, ref_linear64  # noqa: E402

dev = torch.device("cuda")
D, QL, HD, OL = 5120, 1280, 512, 1024


def load(dirp, name, dtype, shape):
    raw = np.fromfile(os.path.join(dirp, f"n0-{name}.bin"), dtype=np.uint8)
    t = torch.from_numpy(raw.copy()).view(dtype)
    return t.reshape(shape).to(dev)


def cmp(name, got, ref):
    got = got.double()
    ref = ref.double()
    rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    close = ((got - ref).abs() <= 2 ** -7 * ref.abs() + 1e-6 * ref.abs().max()).float().mean().item()
    print(f"{name:10s} max rel {rel:9.2e}  within bf16 {close:.5f}  |ref| {ref.abs().max().item():.3e}", flush=True)


def mx(x, w, ws, fmt):
    q, s = ref_act_quant(x)
    wd = dequant_fp8(w, ws) if fmt == "fp8" else dequant_fp4(w, ws)
    return ref_linear64(q, s, wd)


def main():
    root, dd = sys.argv[1], sys.argv[2]
    li = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    sh = Shards(root, sys.argv[4] if len(sys.argv) > 4 else None)
    peer = sys.argv[5] if len(sys.argv) > 5 else None
    p = f"layers.{li}."
    g = lambda n: sh.get(p + n).to(dev)
    n = os.path.getsize(os.path.join(dd, "n0-x.bin")) // (D * 2)
    print(f"{n} rows")
    x = load(dd, "x", torch.bfloat16, (n, D))
    qa = load(dd, "qa", torch.bfloat16, (n, QL))
    cmp("wq_a", qa, mx(x, g("attn.wq_a.weight"), g("attn.wq_a.scale"), "fp8"))
    y = load(dd, "y", torch.bfloat16, (n, HD))
    cmp("wkv", y, mx(x, g("attn.wkv.weight"), g("attn.wkv.scale"), "fp8"))
    qr = load(dd, "qr", torch.bfloat16, (n, QL))
    f = qa.float()
    ref_qr = (g("attn.q_norm.weight") * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-20))).to(torch.bfloat16)
    cmp("q_norm", qr, ref_qr)
    q = load(dd, "q_norope", torch.bfloat16, (n, 16 * HD))
    w, ws = g("attn.wq_b.weight"), g("attn.wq_b.scale")
    cmp("wq_b", q, mx(qr, w[:8192], ws[:256], "fp8"))
    # wo_a: o quant-dequantized, then groups 0 and 1 (node 0's) as bf16 einsums
    o = load(dd, "o", torch.bfloat16, (n, 16 * HD))
    u = load(dd, "u", torch.bfloat16, (n, 2 * OL))
    qo, so = ref_act_quant(o)
    of = (qo.double().view(n, -1, 32) * so.double()[:, :, None]).view(n, -1).to(torch.bfloat16)
    woa = dequant_fp8(g("attn.wo_a.weight"), g("attn.wo_a.scale")).to(torch.bfloat16)  # [8192, 4096]
    if os.path.exists(os.path.join(dd, "n0-wo_a0.bin")):
        w0 = load(dd, "wo_a0", torch.bfloat16, (OL, 4096))
        cmp("wo_a0 weight", w0, woa[:OL].double())
        ofk = load(dd, "o_fake", torch.bfloat16, (n, 16 * HD))
        qo2, so2 = ref_act_quant(o)
        cmp("o fake", ofk, (qo2.double().view(n, -1, 32) * so2.double()[:, :, None]).view(n, -1))
    ref_u = torch.cat([of[:, k * 4096:(k + 1) * 4096].double() @ woa[k * OL:(k + 1) * OL].double().T for k in range(2)], 1)
    cmp("wo_a", u, ref_u)
    uf = load(dd, "u_full", torch.bfloat16, (n, 4 * OL))
    cmp("u_full[:2048] == u", uf[:, :2 * OL], u.double())
    pa = load(dd, "pa", torch.float32, (n, 2560))
    w, ws = g("attn.wo_b.weight"), g("attn.wo_b.scale")
    cmp("wo_b", pa, mx(uf, w[:2560, :4096], ws[:80, :128], "fp8"))
    if peer:
        u2 = torch.from_numpy(np.fromfile(os.path.join(peer, "n2-u.bin"), dtype=np.uint8).copy()).view(torch.bfloat16).reshape(n, 2 * OL).to(dev)
        cmp("u_full[2048:] == node 2's u", uf[:, 2 * OL:], u2.double())
    # the router
    xm = load(dd, "mx", torch.bfloat16, (n, D))
    lg = load(dd, "logits", torch.float32, (n, 384))
    cmp("gate", lg, xm.double() @ g("ffn.gate.weight").double().T)
    pick = load(dd, "pick", torch.int32, (n, 7))
    wts = load(dd, "wts", torch.float32, (n, 7))
    print("pick", pick.tolist())
    print("wts ", [[round(v, 4) for v in r] for r in wts.tolist()])
    # gate / up: node 0's pack (main columns 0..512, rest 1024..1152 for even experts and not the shared one)
    # the pack: every slot's main columns [P, 512], then every slot's rest columns [P, 128]
    xdr = load(dd, "xd", torch.bfloat16, (n * 7 * 640,))
    xd = torch.cat([xdr[:n * 7 * 512].view(n * 7, 512), xdr[n * 7 * 512:].view(n * 7, 128)], 1)
    q, s = ref_act_quant(xm)
    errs = []
    for r in range(n):
        for sl in range(7):
            e = int(pick[r, sl])
            pre = p + (f"ffn.experts.{e}." if e < 384 else "ffn.shared_experts.")
            fmt = "fp4" if e < 384 else "fp8"
            cols = [(0, 512)] + ([(1024, 1152)] if (e < 384 and e % 2 == 0) else [])
            outs = []
            for c0, c1 in cols:
                def part(nm):
                    wt, sc = sh.get(pre + nm + ".weight").to(dev), sh.get(pre + nm + ".scale").to(dev)
                    if fmt == "fp4":
                        return ref_linear64(q[r:r + 1], s[r:r + 1], dequant_fp4(wt[c0:c1], sc[c0:c1]))
                    return ref_linear64(q[r:r + 1], s[r:r + 1], dequant_fp8(wt[c0:c1], sc[c0 // 32:c1 // 32]))
                gt = part("w1").to(torch.bfloat16).float()
                up = part("w3").to(torch.bfloat16).float()
                up = up.clamp(-10, 10)
                gt = gt.clamp(max=10)
                outs.append((torch.nn.functional.silu(gt) * up * wts[r, sl]).to(torch.bfloat16))
            ref = torch.cat(outs, 1)
            got = xd[r * 7 + sl, :ref.shape[1]]
            errs.append(((got.double() - ref.double()).abs().max() / ref.double().abs().max().clamp_min(1e-30)).item())
    print(f"gate/up    max rel over the slots {max(errs):9.2e} (median {sorted(errs)[len(errs) // 2]:.2e})")
    xf = load(dd, "xd_full", torch.bfloat16, (n * 7, 1152))
    cmp("xd_full[:, :512] == xd main", xf[:, :512], xd[:, :512].double())
    pm = load(dd, "pm", torch.float32, (n, 2560))
    hq, hs = ref_act_quant(xf)
    ref = torch.zeros(n, 2560, dtype=torch.float64, device=dev)
    for r in range(n):
        order = sorted(range(7), key=lambda sl: int(pick[r, sl]))
        for sl in order:
            e = int(pick[r, sl])
            pre = p + (f"ffn.experts.{e}." if e < 384 else "ffn.shared_experts.")
            wt, sc = sh.get(pre + "w2.weight").to(dev), sh.get(pre + "w2.scale").to(dev)
            wd = dequant_fp4(wt[:2560, :576], sc[:2560, :36]) if e < 384 else dequant_fp8(wt[:2560, :1152], sc[:80, :36])
            ref[r] += ref_linear64(hq[r * 7 + sl:r * 7 + sl + 1], hs[r * 7 + sl:r * 7 + sl + 1], wd)[0]
    cmp("down", pm, ref)
    if os.path.exists(os.path.join(dd, "n0-head_x.bin")):
        hx = load(dd, "head_x", torch.bfloat16, (1, D))
        hl = load(dd, "head_l", torch.float32, (1, 32384))
        cmp("head", hl, hx.double() @ sh.get("head.weight")[:32384].to(dev).double().T)
        top = torch.topk(hl[0], 5)
        print("head top ids (node 0's columns)", top.indices.tolist(), [round(v, 2) for v in top.values.tolist()])


if __name__ == "__main__":
    main()
