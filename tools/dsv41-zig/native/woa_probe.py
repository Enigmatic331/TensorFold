import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_native import Shards, ref_act_quant, dequant_fp8
dev = torch.device("cuda")
root, dd, idx = sys.argv[1], sys.argv[2], sys.argv[3]
sh = Shards(root, idx)
ld = lambda n, dt, shp: torch.from_numpy(np.fromfile(os.path.join(dd, f"n0-{n}.bin"), dtype=np.uint8).copy()).view(dt).reshape(shp).to(dev)
n = 5
o = ld("o", torch.bfloat16, (n, 8192)); u = ld("u", torch.bfloat16, (n, 2048))
W = sh.get("layers.0.attn.wo_a.weight").to(dev); S = sh.get("layers.0.attn.wo_a.scale").to(dev)
wd = dequant_fp8(W, S).to(torch.bfloat16)
qo, so = ref_act_quant(o); of = (qo.double().view(n, -1, 32) * so.double()[:, :, None]).view(n, -1)
def score(name, ref):
    rel = ((u.double() - ref).abs().max() / ref.abs().max()).item()
    print(f"{name:40s} {rel:.3e}")
for gs in [(0, 1), (2, 3), (4, 5), (6, 7)]:
    ref = torch.cat([of[:, k * 4096:(k + 1) * 4096] @ wd[g * 1024:(g + 1) * 1024].double().T for k, g in enumerate(gs)], 1)
    score(f"fake, groups {gs}", ref)
ref = torch.cat([o.double()[:, k * 4096:(k + 1) * 4096] @ wd[k * 1024:(k + 1) * 1024].double().T for k in range(2)], 1)
score("no fake, groups 0,1", ref)
# the weight as stored without its scale
w0 = W.to(torch.float64)
ref = torch.cat([of[:, k * 4096:(k + 1) * 4096] @ w0[k * 1024:(k + 1) * 1024].T for k in range(2)], 1)
score("fake, unscaled weight", ref)
# per-group norms
print("u norm per group", [u[:, k*1024:(k+1)*1024].float().norm().item() for k in range(2)])
ref = torch.cat([of[:, k * 4096:(k + 1) * 4096] @ wd[k * 1024:(k + 1) * 1024].double().T for k in range(2)], 1)
print("ref norm per group", [ref[:, k*1024:(k+1)*1024].norm().item() for k in range(2)])
print("ratio u/ref", (u.double() / ref).flatten()[:12].tolist())
