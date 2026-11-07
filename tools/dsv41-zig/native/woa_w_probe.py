import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_native import Shards, e8m0
root, dd, idx = sys.argv[1], sys.argv[2], sys.argv[3]
sh = Shards(root, idx)
w0 = torch.from_numpy(np.fromfile(os.path.join(dd, "n0-wo_a0.bin"), dtype=np.uint8).copy()).view(torch.bfloat16).reshape(1024, 4096).double()
W = sh.get("layers.0.attn.wo_a.weight")[:1024]; S = sh.get("layers.0.attn.wo_a.scale")[:32]
codes = W.view(torch.uint8)
vals = W.to(torch.float64)
sc = e8m0(S)  # [32, 128]
ref = (vals.view(32, 32, 128, 32) * sc[:, None, :, None]).view(1024, 4096)
print("equal", (w0 == ref).float().mean().item())
r = w0 / ref.where(ref != 0, torch.ones_like(ref))
print("ratio sample row0", r[0, :8].tolist())
print("w0 row0", w0[0, :8].tolist()); print("ref row0", ref[0, :8].tolist()); print("codes row0", codes[0, :8].tolist())
print("scale[0,:4]", S.view(torch.uint8)[0, :4].tolist())
# is w0 = vals * something per element?
print("w0/vals row0", (w0[0, :8] / vals[0, :8]).tolist())
