"""W6-E2 oracle block --- ORACLE / DIAGNOSTIC, NOT DEPLOYABLE.

Pre-registered in research/WAVE6_PREREG.md section 4.

WHAT THIS ANSWERS
    The oracle-frontier study found a boundary-aware model matched or beaten by
    the legal RT-300 through h = 150, but +0.0396 ahead at FULL.  That oracle
    used a GENERIC 150-tree feature bank, so the gap is either (a) it knows tau
    and we must estimate it, or (b) its features are better in that regime.
    Those imply opposite wave-6 programmes.  This block isolates (a): it hands
    OUR OWN 500-column engine the true break location and measures what that is
    worth, by post-break age.

WHY IT DOES NOT SIMPLY LEAK THE LABEL
    The online-step target is y[t] = 1[t >= tau], so an "is post-break" column
    IS the label.  Break series therefore get their true tau and NO-BREAK SERIES
    GET A PLACEBO CUT drawn from the break series' relative-tau distribution
    (seed 20260822, fixed and recorded).  Every series then has a cut, the cut
    reveals nothing about the label, and the model still has to decide whether
    the segment after the cut differs from what came before.  This is the same
    construction the wave-1 taxonomy and the oracle-frontier study used.

WHY IT IS NOT A REGISTERED MODULE
    It is never importable by `sbr.features.base.load_all()`, so it cannot reach
    a production manifest or a `crunch test`.  It writes its cache directly, and
    `sbr.pipeline.run` loads it by name like any other cached block.
"""
from __future__ import annotations

import json, os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
from sbr.store import load_store
from sbr.transforms import HistParams
from sbr.nullcal import NullCal
from sbr.features.base import make_ctx

NAME = "w6oracle"
SEED = 20260822
COLS = ["or_elapsed", "or_log_elapsed", "or_frac",
        "or_seg_mean_z", "or_seg_rabs_z", "or_seg_lag1_z",
        "or_seg_sq_z", "or_pre_post_mean", "or_pre_post_rabs"]


def build_series(h, o, cut, ctx):
    n = len(o)
    A = np.full((n, len(COLS)), np.nan)
    if cut < 0 or cut >= n:
        return A
    hp = ctx.hp
    z = (o - hp.mu) / hp.sd
    zr = (o - hp.med) / hp.mad
    lag = np.concatenate([[0.0], z[1:] * z[:-1]])[:n]
    cz, cr, cl, cs = (np.concatenate([[0.0], np.cumsum(v)])
                      for v in (z, np.abs(zr), lag, z * z))
    t = np.arange(n)
    m = t - cut + 1                                    # post-cut segment length
    ok = t >= cut
    if not ok.any():
        return A
    mm = m[ok].astype(np.float64)
    seg_mean = (cz[t[ok] + 1] - cz[cut]) / mm
    seg_rabs = (cr[t[ok] + 1] - cr[cut]) / mm
    seg_lag = (cl[t[ok] + 1] - cl[cut]) / mm
    seg_sq = (cs[t[ok] + 1] - cs[cut]) / mm
    nc = ctx.nc
    A[ok, 0] = mm
    A[ok, 1] = np.log1p(mm)
    A[ok, 2] = mm / (t[ok] + 1.0)
    A[ok, 3] = np.clip(nc.z("mean", mm, seg_mean), -12, 12)
    A[ok, 4] = np.clip(nc.z("rabs", mm, seg_rabs), -12, 12)
    A[ok, 5] = np.clip(nc.z("lag1", mm, seg_lag), -12, 12)
    A[ok, 6] = np.clip(nc.z("sq", mm, seg_sq), -12, 12)
    if cut >= 4:                                       # pre-cut contrast
        pre_mean = cz[cut] / cut
        pre_rabs = cr[cut] / cut
        A[ok, 7] = seg_mean - pre_mean
        A[ok, 8] = seg_rabs - pre_rabs
    return A


def main():
    t0 = time.time()
    st = load_store(f"{ROOT}/cache/store")
    meta = st.meta
    tau = meta.tau_index.to_numpy()
    n_on = meta.n_online.to_numpy()
    has = meta.has_break.to_numpy().astype(bool)

    rng = np.random.default_rng(SEED)
    rel = tau[has] / np.maximum(n_on[has], 1)
    rel = rel[(rel > 0) & (rel < 1)]
    cuts = np.where(has, tau, np.clip((rng.choice(rel, len(tau)) * n_on).astype(np.int64),
                                      1, np.maximum(n_on - 2, 1)))
    n_rows = int(n_on.sum())
    out = np.lib.format.open_memmap(f"{ROOT}/cache/features/{NAME}.npy", mode="w+",
                                    dtype=np.float32, shape=(n_rows, len(COLS)))
    json.dump({"cols": COLS, "version": "1", "owner": "claude-wave6-ORACLE"},
              open(f"{ROOT}/cache/features/{NAME}.cols.json", "w"))
    pos = 0
    for i in range(st.n_series):
        h, o = st.hist(i), st.online(i)
        ctx = make_ctx(h, o)
        out[pos:pos + len(o)] = build_series(h, o, int(cuts[i]), ctx).astype(np.float32)
        pos += len(o)
        if (i + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"{i+1}/{st.n_series}  {el:.0f}s  eta {el/(i+1)*(st.n_series-i-1):.0f}s",
                  flush=True)
    out.flush()
    np.save(f"{ROOT}/research/oof/w6oracle_cuts.npy", cuts)
    print(f"{NAME} -> {out.shape}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
