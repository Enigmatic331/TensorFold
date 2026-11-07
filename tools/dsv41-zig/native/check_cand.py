"""Track B's GPU candidates (zig/kernels/cuda/dsv41_ops.cu tf_ds_top_cand_rows_kernel) against sampling.topCandidates'
order written here in plain Python: a row's first `count` (value, id), the larger value first, of equal values the lower
id, +0 and -0 equal, the row's own values kept. A NaN row must be flagged 1, a row with more ties at the cut than the
tie buffer (2,048) holds flagged 2 (the host then reads the row whole). Runs in nvcr.io/nvidia/pytorch:26.07-py3 on a GB10:

    python check_cand.py

Exit status 0 when every check holds."""
import ctypes
import math
import os
import random
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cuda")
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def build():
    so = os.path.join("/tmp", "dsn_cand_host.so")
    subprocess.check_call(["nvcc", "-arch=sm_121a", "-O3", "--fmad=false", "--ftz=false", "-std=c++17", "-shared",
                           "-Xcompiler", "-fPIC", "-o", so, os.path.join(HERE, "cand_host.cu")])
    lib = ctypes.CDLL(so)
    P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
    lib.dsn_top_cand.argtypes = [P, U, I, I, I, P, P, P]
    return lib


def ref(row, count):
    """sampling.topCandidates: insertion keeping the first `count` by (value desc, id asc); -0 == +0."""
    vals = row.tolist()
    order = sorted(range(len(vals)), key=lambda i: (-(vals[i] if vals[i] != 0 else 0.0), i))
    k = min(count, len(vals))
    return [vals[i] for i in order[:k]], order[:k]


def run(lib, rows, count):
    rows = rows.contiguous().float().to(dev)
    r, n = rows.shape
    vals = torch.full((r, count), float("nan"), device=dev)
    ids = torch.full((r, count), -1, dtype=torch.int64, device=dev)
    flags = torch.full((r,), -1, dtype=torch.int32, device=dev)
    rc = lib.dsn_top_cand(ctypes.c_void_p(rows.data_ptr()), n, n, r, count, ctypes.c_void_p(vals.data_ptr()),
                          ctypes.c_void_p(ids.data_ptr()), ctypes.c_void_p(flags.data_ptr()))
    assert rc == 0, rc
    return vals.cpu(), ids.cpu(), flags.cpu()


def same_bits(a, b):
    return all(math.copysign(1, x) == math.copysign(1, y) and (x == y or (x != x and y != y)) for x, y in zip(a, b))


def main():
    lib = build()
    V = 129280
    g = torch.Generator().manual_seed(7)
    rows = []
    rows.append(torch.randn(V, generator=g) * 4)                         # ordinary logits
    t = torch.randn(V, generator=g); top = t.topk(40).values
    t[torch.randint(0, V, (50,), generator=g)] = top[27].item()          # ties at the 28th place
    rows.append(t)
    t = torch.randn(V, generator=g) - 10; t[torch.randint(0, V, (30,), generator=g)] = 0.0
    t[torch.randint(0, V, (30,), generator=g)] = -0.0                   # +0 / -0 ties at the top
    rows.append(t)
    t = torch.randn(V, generator=g); t[5] = float("inf"); t[9] = float("inf"); t[100:200] = float("-inf")
    rows.append(t)
    t = torch.randn(V, generator=g); t[:] = t.round(decimals=1)          # many ties everywhere
    rows.append(t)
    rows.append(torch.randn(V, generator=g).half().float())              # coarse values: ties
    batch = torch.stack(rows)
    for count in (1, 8, 28, 64):
        vals, ids, flags = run(lib, batch, count)
        for i in range(batch.shape[0]):
            rv, ri = ref(batch[i], count)
            got_v, got_i = vals[i, :len(rv)].tolist(), ids[i, :len(ri)].tolist()
            check(f"row {i} count {count}: the candidates and their order", flags[i].item() == 0 and got_i == ri and same_bits(got_v, rv),
                  f"flag {flags[i].item()}" if flags[i].item() else "")
    # one finite value and the rest -inf: count 1 finds it; any count past it cuts in 129,279 ties (flag 2: the host
    # reads the row whole); a short row of 1,990 -inf ties (within the tie buffer) is exact
    t = torch.full((1, V), float("-inf")); t[0, 3] = 1.0
    vals, ids, flags = run(lib, t, 1)
    check("one finite value, count 1: it alone", flags.tolist() == [0] and ids[0, 0].item() == 3 and vals[0, 0].item() == 1.0)
    for count in (8, 28, 64):
        _, _, flags = run(lib, t, count)
        check(f"one finite value, count {count}: the -inf ties past the buffer are flagged 2", flags.tolist() == [2])
    t = torch.full((1, 2000), float("-inf")); t[0, :10] = torch.randn(10, generator=g)
    vals, ids, flags = run(lib, t, 28)
    rv, ri = ref(t[0], 28)
    check("10 finite values and 1,990 -inf ties, count 28", flags.tolist() == [0] and ids[0].tolist() == ri and same_bits(vals[0].tolist(), rv))
    # a row shorter than the count
    short = torch.randn(3, 10, generator=g)
    vals, ids, flags = run(lib, short, 28)
    for i in range(3):
        rv, ri = ref(short[i], 28)
        check(f"short row {i}: all 10 in order", flags[i].item() == 0 and ids[i, :10].tolist() == ri and same_bits(vals[i, :10].tolist(), rv))
    # flags
    t = torch.randn(2, V, generator=g); t[0, 77] = float("nan")
    _, _, flags = run(lib, t, 28)
    check("a NaN row is flagged 1, its neighbour not", flags.tolist() == [1, 0])
    t = torch.zeros(1, V)
    _, _, flags = run(lib, t, 28)
    check("a row of ties past the tie buffer is flagged 2", flags.tolist() == [2])
    # many random rows, sampled-row counts
    for trial in range(20):
        b = torch.randn(48, V, generator=g) * random.Random(trial).uniform(0.5, 8)
        vals, ids, flags = run(lib, b, 28)
        ok = flags.eq(0).all().item()
        for i in range(0, 48, 7):
            rv, ri = ref(b[i], 28)
            ok = ok and ids[i].tolist() == ri and same_bits(vals[i].tolist(), rv)
        check(f"48 random rows, trial {trial}", ok)
    # the kernel's time (ms) a round of R rows at the served vocabulary, and the copy it replaces (R x V fp32 to pageable
    # host memory), for the record; not a check
    for R in (1, 3, 14, 48):
        b = (torch.randn(R, V, generator=g) * 4).to(dev)
        vals = torch.empty(R, 28, device=dev); ids = torch.empty(R, 28, dtype=torch.int64, device=dev)
        flags = torch.empty(R, dtype=torch.int32, device=dev)
        call = lambda: lib.dsn_top_cand(ctypes.c_void_p(b.data_ptr()), V, V, R, 28, ctypes.c_void_p(vals.data_ptr()),
                                        ctypes.c_void_p(ids.data_ptr()), ctypes.c_void_p(flags.data_ptr()))
        for _ in range(3):
            call()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record(); [call() for _ in range(20)]; t1.record(); torch.cuda.synchronize()
        host = torch.empty(R, V)
        c0 = torch.cuda.Event(enable_timing=True); c1 = torch.cuda.Event(enable_timing=True)
        c0.record(); [host.copy_(b) for _ in range(20)]; c1.record(); torch.cuda.synchronize()
        print(f"time R={R}: kernel {t0.elapsed_time(t1) / 20:.3f} ms (with its sync), the full copy {c0.elapsed_time(c1) / 20:.3f} ms")
    print(f"{len(failures)} failed" if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
