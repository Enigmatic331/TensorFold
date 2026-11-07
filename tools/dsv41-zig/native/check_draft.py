"""Check a native engine's DSpark drafter pass (TF_DS_DRAFT_DUMP of node 0: 2D split, pair 0, TP2 rank 0) against
DeepSeek's reference DSparkBlock arithmetic (inference/model.py: forward_embed, DSparkBlock/Block.forward,
DSparkAttention, MoE, forward_head; kernel.py act_quant, sparse_attn, hc_split_sinkhorn), written here in plain torch:

    python check_draft.py <checkpoint dir> <dump dir>

Three parts:
  1. the stage rings: every window key recomputed from the dumped taps (main_proj, main_norm, each stage's wkv,
     kv_norm, RoPE, the fp8 round trip) against the engine's ring rows;
  2. each step of the pass from the engine's own inputs to that step (localizes a mismatch);
  3. the whole drafter from the taps alone (the reference's own values all the way): its drafts and confidences
     against the engine's.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_native import Shards, ref_act_quant, dequant_fp8, dequant_fp4  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
D, QL, HD, OL, RD, HC, V = 5120, 1280, 512, 1024, 64, 4, 129280
NH, NG, WIN = 64, 8, 128
EPS, HC_EPS, ITERS = 1e-20, 1e-6, 20
TOPK, NE, SCALE = 3, 128, 1.5
MAX_ROWS = 80
BF = torch.bfloat16


def raw(dd, name, dtype, shape=None):
    p = os.path.join(dd, f"{name}.bin")
    if not os.path.exists(p):
        return None
    t = torch.from_numpy(np.fromfile(p, dtype=np.uint8).copy()).view(dtype)
    return (t.reshape(shape) if shape is not None else t).to(dev)


def cmp(name, got, ref, tol=2 ** -7):
    got = got.double()
    ref = ref.double()
    rel = ((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    close = ((got - ref).abs() <= tol * ref.abs() + 1e-3 * ref.abs().max()).float().mean().item()
    print(f"  {name:22s} max rel {rel:9.2e}  close {close:.5f}  |ref| {ref.abs().max().item():.3e}", flush=True)
    return rel


# ------------------------------------------------------------------ reference arithmetic
def qdq(x):
    """act_quant(x, 32, ue8m0, inplace=True): the fp8 round trip back to bf16."""
    shp = x.shape
    q, s = ref_act_quant(x.reshape(-1, shp[-1]))
    m, k = q.shape
    return (q.float().view(m, k // 32, 32) * s[:, :, None]).view(shp).to(BF)


class W:
    def __init__(self, sh):
        self.sh = sh
        self.cache = {}

    def get(self, n):
        if n not in self.cache:
            self.cache[n] = self.sh.get(n).to(dev)
        return self.cache[n]

    def fp8(self, n):
        k = n + "#dq"
        if k not in self.cache:
            self.cache[k] = dequant_fp8(self.get(n + ".weight"), self.get(n + ".scale")).float()
        return self.cache[k]

    def fp4(self, n):
        return dequant_fp4(self.sh.get(n + ".weight").to(dev), self.sh.get(n + ".scale").to(dev)).float()


def lin8(x, wdq):
    """linear() on an fp8 weight: act_quant(x) then the GEMM, out in bf16."""
    shp = x.shape
    q, s = ref_act_quant(x.reshape(-1, shp[-1]))
    m, k = q.shape
    xdq = (q.float().view(m, k // 32, 32) * s[:, :, None]).view(m, k)
    return (xdq @ wdq.T).to(BF).view(*shp[:-1], -1)


def rmsnorm(x, w):
    f = x.float()
    return (w * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + EPS))).to(x.dtype)


def freqs(pos):
    fr = 1.0 / (10000.0 ** (torch.arange(0, RD, 2, dtype=torch.float32, device=dev) / RD))
    ang = torch.outer(pos.to(torch.float32).to(dev), fr)
    return torch.polar(torch.ones_like(ang), ang)


def rope(x, pos, inverse=False):
    """apply_rotary_emb on x[..., -RD:] (rows first: [R, ..., RD]), adjacent pairs."""
    y = x.clone()
    t = y[..., -RD:].float().unflatten(-1, (-1, 2))
    c = torch.view_as_complex(t.contiguous())
    f = freqs(pos)
    if inverse:
        f = f.conj()
    f = f.view(f.shape[0], *([1] * (c.ndim - 2)), f.shape[-1])
    y[..., -RD:] = torch.view_as_real(c * f).flatten(-2).to(y.dtype)
    return y


def hc_mixes(h, fn, scale, base):
    x = h.flatten(1).float()
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    mixes = (x @ fn.T) * rs
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = (mixes[:, 2 * HC:] * scale[2] + base[2 * HC:]).view(-1, HC, HC)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    return pre, post, comb


def hc_pre(h, pre):
    return (pre.unsqueeze(-1) * h.float()).sum(1).to(BF)


def hc_post(x, res, post, comb):
    y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + (comb.unsqueeze(-1) * res.float().unsqueeze(-2)).sum(1)
    return y.to(BF)


def attn_core(q, win, blk, sink, n_blk):
    """sparse_attn over the window keys (all valid) and the block rows' keys; q [R, H, HD]; P in bf16 for PV."""
    R = q.shape[0]
    out = torch.empty_like(q)
    for s in range(R // n_blk):
        rows = slice(s * n_blk, (s + 1) * n_blk)
        kv = torch.cat([win, blk[rows]], 0).float()  # [n, HD]
        sc = torch.einsum("rhd,nd->rhn", q[rows].float(), kv) * HD ** -0.5
        m = sc.amax(-1, keepdim=True)
        p = torch.exp(sc - m)
        den = p.sum(-1, keepdim=True) + torch.exp(sink.view(1, -1, 1) - m)
        o = torch.einsum("rhn,nd->rhd", p.to(BF).float(), kv) / den
        out[rows] = o.to(BF)
    return out


class Stage:
    def __init__(self, w, j, heads):
        self.w, self.p, self.heads = w, f"mtp.{j}.", heads

    def g(self, n):
        return self.w.get(self.p + n)

    def attention(self, x, pos, win, n_blk, h0=0, h1=NH, trace=None):
        """DSparkAttention.forward (decode) on heads h0..h1: the output projection's input u and output y."""
        w = self.w
        qa = lin8(x, w.fp8(self.p + "attn.wq_a"))
        qr = rmsnorm(qa, self.g("attn.q_norm.weight"))
        wqb = w.fp8(self.p + "attn.wq_b")[h0 * HD:h1 * HD]
        q = lin8(qr, wqb).view(-1, h1 - h0, HD)
        q = rope(q, pos)
        ykv = lin8(x, w.fp8(self.p + "attn.wkv"))
        kv = qdq(rope(rmsnorm(ykv, self.g("attn.kv_norm.weight")), pos))
        o = attn_core(q, win, kv, self.g("attn.attn_sink")[h0:h1], n_blk)
        o_rope = o
        o = rope(o, pos, inverse=True)
        R = x.shape[0]
        g0, g1 = h0 * NG // NH, h1 * NG // NH
        of = qdq(o.reshape(R, g1 - g0, -1))
        woa = w.fp8(self.p + "attn.wo_a").to(BF).float().view(NG, OL, -1)[g0:g1]
        u = torch.einsum("rgd,gkd->rgk", of.float(), woa).to(BF).reshape(R, -1)
        if trace is not None:
            trace.update(qa=qa, qr=qr, q=q, ykv=ykv, kvb=kv, o_rope=o_rope, o=o, u=u)
        return u

    def wo_b(self, u_full):
        return lin8(u_full, self.w.fp8(self.p + "attn.wo_b"))

    def gate(self, x):
        sc = x.float() @ self.g("ffn.gate.weight").float().T
        scores = F.softplus(sc).sqrt()
        idx = (scores + self.g("ffn.gate.bias")).topk(TOPK, dim=-1)[1]
        wt = scores.gather(1, idx)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * SCALE
        return sc, wt, idx

    def expert(self, x, pfx, wt, fp4):
        f = self.w.fp4 if fp4 else self.w.fp8
        w1, w2, w3 = f(pfx + ".w1"), f(pfx + ".w2"), f(pfx + ".w3")
        gate = lin8(x, w1).float()
        up = lin8(x, w3).float()
        up = up.clamp(-10.0, 10.0)
        gate = gate.clamp(max=10.0)
        y = F.silu(gate) * up
        if wt is not None:
            y = wt * y
        return lin8(y.to(BF), w2)

    def moe(self, x, routing=None):
        sc, wt, idx = self.gate(x) if routing is None else routing
        y = torch.zeros(x.shape[0], D, dtype=torch.float32, device=dev)
        for e in idx.unique().tolist():
            r, t = torch.where(idx == e)
            y[r] += self.expert(x[r], self.p + f"ffn.experts.{e}", wt[r, t, None], True).float()
        y += self.expert(x, self.p + "ffn.shared_experts", None, False).float()
        return y.to(BF)


def main():
    root, dd = sys.argv[1], sys.argv[2]
    meta = dict(l.split() for l in open(os.path.join(dd, "meta.txt")) if l.strip())
    meta = {k: int(v) for k, v in meta.items()}
    print(meta)
    N, steps, n, ring, slots = meta["N"], meta["steps"], meta["n"], meta["ring"], meta["slots"]
    world, head_n, vlo = meta["world"], meta["head_n"], meta["vocab_lo"]
    token, q0, slot = meta["token"], meta["q0"], meta["slot"]
    assert N == 1, "one stream"
    R = N * n
    sh = Shards(root)
    w = W(sh)
    hv = raw(dd, "rows-0", torch.int64, (6, MAX_ROWS))
    ids, pos = hv[0, :R], hv[1, :R]
    print("ids", ids.tolist(), "pos", pos.tolist())

    # ---------------------------------------------------------------- 1. the rings from the taps
    taps = raw(dd, "taps-0", BF).view(-1, 3 * D)
    tpos = raw(dd, "tpos-0", torch.int64)
    # the latest row of each position (a later window writes over the rows a round rejected)
    last = {}
    for i, p in enumerate(tpos.tolist()):
        last[p] = i
    want = [p for p in range(max(0, q0 - WIN), q0)]
    missing = [p for p in want if p not in last]
    print(f"taps rows {taps.shape[0]}, positions {min(last)}..{max(last)}, window {want[0]}..{want[-1]}, missing {missing}")
    sel = torch.tensor([last[p] for p in want], device=dev)
    wpos = torch.tensor(want, device=dev)
    main_x = rmsnorm(lin8(taps[sel], w.fp8("mtp.0.main_proj")), w.get("mtp.0.main_norm.weight"))
    ref_win = []
    eng_win = []
    stages = [Stage(w, j, NH) for j in range(3)]
    for j, st in enumerate(stages):
        kv = qdq(rope(rmsnorm(lin8(main_x, w.fp8(st.p + "attn.wkv")), st.g("attn.kv_norm.weight")), wpos))
        ref_win.append(kv)
        plane = raw(dd, f"attn_ring-{j}", BF, (slots * ring, HD))
        eng = plane[slot * ring + (wpos % ring)]
        eng_win.append(eng)
        print(f"stage {j} ring")
        cmp("window keys", eng, kv)

    # ---------------------------------------------------------------- 2. each step from the engine's inputs
    emb = w.get("embed.weight")
    h0 = emb[ids].unsqueeze(1).repeat(1, HC, 1)
    pre_in = torch.zeros(R, HC, device=dev)
    pre_in[:, 0] = 1
    eh = h0
    hh = 32  # node 0's heads (TP2 rank 0)
    for j, st in enumerate(stages):
        print(f"stage {j} (engine inputs)")
        h_a = raw(dd, f"h_a-{j}", BF, (R, HC, D))
        e_pre_in = raw(dd, f"pre_in-{j}", torch.float32, (R, HC))
        if j == 0:
            cmp("h_a == embed", h_a, eh)
        cmp("pre_in", e_pre_in, pre_in if j == 0 else e_pre_in)
        x_ref = rmsnorm(hc_pre(h_a, e_pre_in), st.g("attn_norm.weight"))
        x = raw(dd, f"attn_in-{j}", BF, (R, D))
        cmp("attn_in", x, x_ref)
        pa, post_a, comb_a = hc_mixes(h_a, st.g("hc_attn_fn"), st.g("hc_attn_scale"), st.g("hc_attn_base"))
        cmp("pre_a (mixes)", raw(dd, f"pre_a-{j}", torch.float32, (R, HC)), pa)
        tr = {}
        u = st.attention(x, pos, eng_win[j], n, 0, hh, tr)
        for k, shape in (("qa", (R, QL)), ("ykv", (R, HD)), ("qr", (R, QL)), ("q", (R, hh, HD)), ("kvb", (R, HD)),
                         ("o_rope", (R, hh, HD)), ("o", (R, hh, HD)), ("u", (R, 4 * OL))):
            got = raw(dd, f"{k}-{j}", BF, shape)
            if got is not None:
                cmp(k, got, tr[k])
        # the whole attention output: every head's u through wo_b, against the gathered partials' sum
        u_full = st.attention(x, pos, eng_win[j], n, 0, NH)
        y_ref = st.wo_b(u_full)
        ga = raw(dd, f"attn_gather-{j}", torch.float32, (world, R, D))
        cmp("attn out (gather sum)", ga.sum(0), y_ref.float())
        h_f = raw(dd, f"h_f-{j}", BF, (R, HC, D))
        cmp("h_f (attn post)", h_f, hc_post(ga.sum(0), h_a, post_a, comb_a))
        e_pre_a = raw(dd, f"pre_a-{j}", torch.float32, (R, HC))
        xf_ref = rmsnorm(hc_pre(h_f, e_pre_a), st.g("ffn_norm.weight"))
        xf = raw(dd, f"moe_in-{j}", BF, (R, D))
        cmp("moe_in", xf, xf_ref)
        pf, _, _ = hc_mixes(h_f, st.g("hc_ffn_fn"), st.g("hc_ffn_scale"), st.g("hc_ffn_base"))
        cmp("pre_f (mixes)", raw(dd, f"pre_f-{j}", torch.float32, (R, HC)), pf)
        sc, wt, idx = st.gate(xf)
        gl = raw(dd, f"gl-{j}", torch.float32, (R, NE))
        if gl is not None:
            cmp("gate logits", gl, sc)
        pick = raw(dd, f"pick-{j}", torch.int32, (R, TOPK + 1))
        mw = raw(dd, f"mw-{j}", torch.float32, (R, TOPK + 1))
        if pick is not None:
            ep = pick[:, :TOPK].long().sort(-1)[0]
            rp = idx.sort(-1)[0]
            print(f"  routing equal {bool((ep == rp).all())}  engine {pick.tolist()}  ref {idx.tolist()}")
            print(f"  engine weights {mw.tolist()}")
            print(f"  ref weights    {wt.tolist()}")
        gm = raw(dd, f"moe_gather-{j}", torch.float32, (world, R, D))
        cmp("moe out (gather sum)", gm.sum(0), st.moe(xf).float())
    hs = raw(dd, "stages_h-3", BF, (R, HC, D))
    ps = raw(dd, "stages_pre-3", torch.float32, (R, HC))
    xc = raw(dd, "xc-0", BF, (R, D))
    cmp("xc (collapse)", xc, hc_pre(hs, ps))
    xn = raw(dd, "xn-0", BF, (R, D))
    cmp("xn (norm)", xn, rmsnorm(xc, w.get("mtp.2.norm.weight")))
    head = w.get("head.weight")
    logits = xn.float() @ head.float().T
    local = raw(dd, "local-3", torch.float32, (R, head_n))
    cmp("local logits", local, logits[:, vlo:vlo + head_n])
    out_e = raw(dd, "out-0", torch.int64, (N, steps + 1))
    conf_e = raw(dd, "conf-0", torch.float32, (N, steps))
    me, mh = w.get("mtp.2.markov_head.embed.weight"), w.get("mtp.2.markov_head.head.weight").float()

    def markov(lg, x_pre):
        o = [token]
        embs = []
        for i in range(n):
            e = me[o[i]]
            embs.append(e)
            b = e.float() @ mh.T
            o.append(int((lg[i] + b).argmax()))
        cw = w.get("mtp.2.confidence_head.proj.weight").float().view(-1)
        conf = torch.cat([x_pre.float(), torch.stack(embs).float()], -1) @ cw
        return o, conf

    o_ref, c_ref = markov(logits, xc)
    print(f"drafts engine {out_e.tolist()}  ref from engine head input {o_ref}")
    print(f"conf engine {conf_e.tolist()}\n     ref    {c_ref.tolist()}")

    # ---------------------------------------------------------------- 3. the whole drafter from the taps
    print("reference end to end")
    h = h0
    pre = pre_in
    for j, st in enumerate(stages):
        res = h
        pa, post_a, comb_a = hc_mixes(h, st.g("hc_attn_fn"), st.g("hc_attn_scale"), st.g("hc_attn_base"))
        x = rmsnorm(hc_pre(h, pre), st.g("attn_norm.weight"))
        y = st.wo_b(st.attention(x, pos, ref_win[j], n))
        h = hc_post(y, res, post_a, comb_a)
        res = h
        pf, post_f, comb_f = hc_mixes(h, st.g("hc_ffn_fn"), st.g("hc_ffn_scale"), st.g("hc_ffn_base"))
        x = rmsnorm(hc_pre(h, pa), st.g("ffn_norm.weight"))
        h = hc_post(st.moe(x), res, post_f, comb_f)
        pre = pf
    cmp("stages h vs engine", hs, h)
    xc_r = hc_pre(h, pre)
    lg = rmsnorm(xc_r, w.get("mtp.2.norm.weight")).float() @ head.float().T
    o_full, c_full = markov(lg, xc_r)
    print(f"drafts engine {out_e.tolist()}  ref end to end {o_full}")
    print(f"conf ref end to end {c_full.tolist()}")
    top = lg.topk(5, -1)
    print("ref top5 per row (no markov):", top.indices.tolist())


if __name__ == "__main__":
    main()
