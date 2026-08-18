"""Agent 09 -- rank-correlation between arms' validation predictions + blends."""
from __future__ import annotations
import sys, json, itertools
import numpy as np
sys.path.insert(0, "/home/claude/sb/scripts")
from agent09_lib import Bench
from scipy.stats import rankdata

PRED = "/home/claude/sb/research/oof/agent09"


def main(files, fold=0, base="OBJ-binary", modules="m00_core,m03_dyn"):
    z = {}
    for fn in files:
        d = np.load(f"{PRED}/{fn}.npz")
        for k in d.files:
            n, f = k.split("|f")
            if int(f) == fold:
                z[n] = d[k]
    b = Bench(modules.split(","), folds=(fold,), screen=True)
    pack = b.get(fold)
    t = pack["tva"]
    # within-timestep rank transform (the only comparison the metric cares about)
    def wrank(p):
        o = np.lexsort((p, t)); r = np.empty(len(p))
        ts = t[o]
        gs = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]); ge = np.r_[gs[1:], len(p)]
        gl = ge - gs
        rr = (np.arange(len(p)) - np.repeat(gs, gl)) / np.maximum(np.repeat(gl, gl) - 1, 1)
        r[o] = rr
        return r
    names = sorted(z)
    W = {n: wrank(z[n].astype(np.float64)) for n in names}
    print("\nTS-AUC:")
    sc = {n: b.score(z[n].astype(np.float64), pack) for n in names}
    for n in sorted(names, key=lambda x: -sc[x]):
        print(f"  {sc[n]:.5f}  {n}")
    print("\nwithin-t rank correlation vs", base)
    if base in W:
        for n in sorted(names, key=lambda x: -sc[x]):
            c = np.corrcoef(W[base], W[n])[0, 1]
            print(f"  {c:.4f}  {n}   (TS-AUC {sc[n]:.5f})")
    print("\nblends with", base)
    for n in names:
        if n == base: continue
        for a in (0.3, 0.5, 0.7):
            bl = (1 - a) * W[base] + a * W[n]
            print(f"  blend {base}+{a:.1f}*{n:22s} {b.score(bl, pack):.5f}")
    print("\nfull pairwise corr matrix (within-t ranks)")
    print("        " + " ".join(f"{n[:9]:>9s}" for n in names))
    for n in names:
        print(f"{n[:8]:>8s} " + " ".join(f"{np.corrcoef(W[n],W[m])[0,1]:9.3f}" for m in names))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--files", default="sweepA")
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--base", default="OBJ-binary")
    ap.add_argument("--modules", default="m00_core,m03_dyn")
    a = ap.parse_args(); main(a.files.split(","), a.fold, a.base, a.modules)
