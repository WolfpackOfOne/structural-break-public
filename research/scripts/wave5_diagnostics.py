"""W5-D1/D2/D3 -- what the metric weights, what fools RT-600, and where.

DIAGNOSTIC ONLY.  Nothing here selects a model.  D2 characterises series with
statistics computed over the WHOLE series (including the future), which is
legitimate for forensics and is NEVER a feature: no output of this script is
readable by any production code path.

D1  metric geometry -- where the official n_pos*n_neg pair weight actually sits
D2  false-positive taxonomy -- which no-break series the champion ranks high
D3  break-age profile -- A / B / S TS-AUC by post-break age
"""
from __future__ import annotations

import json, time
import numpy as np
import pandas as pd

from wave5_lib import (Ctx, load_oof, SPECIALISTS, SEEDCLONES, FOLDS, REPORTS,
                       ROOT, AGE_BUCKETS, fmt)
from sbr.metric import ts_auc_flat


# ---------------------------------------------------------------- D1
def d1_metric_geometry(c):
    r = c.dev
    y, t = c.d.y[r], c.d.t[r]
    o = np.lexsort((y, t))
    ts, ys = t[o], y[o]
    g = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    e = np.r_[g[1:], len(ts)]
    npos = np.add.reduceat(ys.astype(np.int64), g)
    nall = e - g
    nneg = nall - npos
    w = (npos * nneg).astype(float)
    tv = ts[g]
    tot = w.sum()
    out = {"total_pair_weight": float(tot), "n_timesteps": int(len(tv))}
    print("=== D1  where the official pair weight sits (by online index t) ===")
    bands = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400),
             (400, 700), (700, 1000)]
    out["by_t"] = {}
    for lo, hi in bands:
        m = (tv >= lo) & (tv < hi)
        out["by_t"][f"{lo}-{hi}"] = {"share": float(w[m].sum() / tot),
                                     "n_series_alive_mean": float(nall[m].mean()) if m.any() else 0.0}
        print(f"  t {lo:4d}-{hi:<4d}  {100*w[m].sum()/tot:5.1f}% of pair weight   "
              f"mean alive {nall[m].mean() if m.any() else 0:6.0f}")
    cw = np.cumsum(w) / tot
    for q in (0.25, 0.50, 0.75, 0.90):
        i = int(np.searchsorted(cw, q))
        out[f"t_at_{int(q*100)}pct_weight"] = int(tv[min(i, len(tv) - 1)])
        print(f"  {int(q*100)}% of all pair weight is at t <= {tv[min(i,len(tv)-1)]}")

    # ---- and by POST-BREAK AGE (positives only carry age)
    a = c.age[r]
    pos = y == 1
    print("\n  post-break age share of POSITIVE rows (the pairs' pos side):")
    out["by_age"] = {}
    # each positive row at t contributes n_neg(t) pairs
    negs_at_t = np.zeros(int(t.max()) + 2)
    negs_at_t[tv] = nneg
    wpos = negs_at_t[t[pos]]
    ap = a[pos]
    for lo, hi in AGE_BUCKETS:
        m = (ap >= lo) & (ap < hi)
        k = f"{lo}-{hi if hi < 10**9 else ''}"
        out["by_age"][k] = {"row_share": float(m.mean()),
                            "pair_weight_share": float(wpos[m].sum() / wpos.sum())}
        print(f"  age {k:>7s}  rows {100*m.mean():5.1f}%   pair weight {100*wpos[m].sum()/wpos.sum():5.1f}%")
    return out


# ---------------------------------------------------------------- D2
def _series_forensics(c):
    """Transparent whole-series descriptors.  FORENSIC ONLY -- uses the future."""
    st = c.d.st
    n = st.n_series
    cols = {k: np.full(n, np.nan) for k in (
        "shock_max_absz", "shock_isolation", "burst_ratio", "burst_reverted",
        "trend_absslope", "tail_kurt_hist", "tail_rate_online", "dep_shift",
        "level_shift_end", "scale_shift_end", "spec_shift", "n_online", "n_hist")}
    for i in range(n):
        h, o = st.hist(i), st.online(i)
        if len(o) < 4:
            continue
        mu, sd = h.mean(), max(h.std(ddof=1), 1e-9)
        z = (o - mu) / sd
        az = np.abs(z)
        cols["n_online"][i], cols["n_hist"][i] = len(o), len(h)
        # transient shock: how extreme, and how isolated is the extreme
        cols["shock_max_absz"][i] = az.max()
        thr = max(3.0, np.quantile(np.abs((h - mu) / sd), 0.999))
        nex = int((az > thr).sum())
        cols["shock_isolation"][i] = 1.0 / max(nex, 1)
        # variance burst: worst 32-window variance ratio, and whether it reverted
        w = min(32, len(o))
        cs = np.concatenate([[0.0], np.cumsum(z * z)])
        rv = (cs[w:] - cs[:-w]) / w if len(o) >= w else np.array([np.mean(z * z)])
        cols["burst_ratio"][i] = rv.max() if len(rv) else np.nan
        tail_v = np.mean(z[-w:] ** 2)
        cols["burst_reverted"][i] = float(rv.max() / max(tail_v, 1e-9)) if len(rv) else np.nan
        # local trend
        x = np.arange(len(o)) - (len(o) - 1) / 2.0
        cols["trend_absslope"][i] = abs(float((x @ z) / max((x * x).sum(), 1e-9))) * len(o)
        # tails
        hz = (h - mu) / sd
        cols["tail_kurt_hist"][i] = float(np.mean(hz ** 4))
        q99 = np.quantile(np.abs(hz), 0.99)
        cols["tail_rate_online"][i] = float(np.mean(az > q99))
        # dependence shift
        def acf1(v):
            v = v - v.mean()
            d = float((v * v).sum())
            return float((v[1:] * v[:-1]).sum() / d) if d > 0 else 0.0
        cols["dep_shift"][i] = abs(acf1(o) - acf1(h))
        # persistent end-of-series level / scale displacement
        k = max(8, len(o) // 4)
        cols["level_shift_end"][i] = abs(float(z[-k:].mean()))
        cols["scale_shift_end"][i] = abs(np.log(max(float(z[-k:].std(ddof=1)), 1e-9)))
        # crude spectral shift: high-frequency energy share, online vs hist
        def hf(v):
            dv = np.diff(v)
            return float(np.var(dv) / max(np.var(v), 1e-12))
        cols["spec_shift"][i] = abs(hf(o) - hf(h))
    return pd.DataFrame(cols)


TAXONOMY = {
    "transient_shock":       ("shock_max_absz", "shock_isolation"),
    "heavy_tail_outlier":    ("tail_kurt_hist", "tail_rate_online"),
    "variance_burst":        ("burst_ratio", "burst_reverted"),
    "local_trend":           ("trend_absslope", None),
    "dependence_fluct":      ("dep_shift", None),
    "spectral_change":       ("spec_shift", None),
    "persistent_displacement": ("level_shift_end", "scale_shift_end"),
}


def d2_false_positives(c, Sv):
    print("\n=== D2  what fools the RT-600 architecture on no-break series ===")
    r = c.dev
    # within-timestep percentile rank of every online row, then per-series mean
    from wave4_ensemble import within_t_rank
    pr = within_t_rank(Sv[r], c.d.t[r])
    sid = c.d.sidx[r]
    order = np.argsort(sid, kind="stable")
    s_s, p_s = sid[order], pr[order]
    b = np.flatnonzero(np.r_[True, s_s[1:] != s_s[:-1]])
    e = np.r_[b[1:], len(s_s)]
    sev = np.full(c.d.st.n_series, np.nan)
    sev[s_s[b]] = [p_s[i:j].mean() for i, j in zip(b, e)]

    F = _series_forensics(c)
    dev_series = np.unique(sid)
    neg = dev_series[~c.has_break[dev_series]]
    pos = dev_series[c.has_break[dev_series]]
    sn = sev[neg]
    print(f"  {len(neg)} no-break dev series, {len(pos)} break dev series")
    print(f"  mean within-t percentile rank: negatives {np.nanmean(sn):.4f}  "
          f"positives {np.nanmean(sev[pos]):.4f}")

    out = {"n_neg": int(len(neg)), "n_pos": int(len(pos)),
           "mean_rank_neg": float(np.nanmean(sn)), "mean_rank_pos": float(np.nanmean(sev[pos])),
           "groups": {}}
    base = F.loc[neg]
    med = base.median()
    iqr = (base.quantile(.75) - base.quantile(.25)).replace(0, np.nan)

    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(round(frac * len(neg))))
        hard = neg[np.argsort(-sn)[:k]]
        easy = neg[np.argsort(sn)[:k]]
        H, E = F.loc[hard], F.loc[easy]
        # standardised gap of each descriptor, hard negatives vs all negatives
        gap = ((H.median() - med) / iqr).sort_values(ascending=False)
        # assign each hard negative to the taxonomy bucket it is most extreme on
        zs = ((F.loc[hard] - med) / iqr)
        lab = {}
        for name, (a, bb) in TAXONOMY.items():
            v = zs[a].copy()
            if bb:
                v = 0.5 * (v + zs[bb])
            lab[name] = v
        L = pd.DataFrame(lab)
        assign = L.idxmax(axis=1).value_counts()
        # the same assignment on ALL negatives, so shares are comparable
        zs_all = ((F.loc[neg] - med) / iqr)
        lab_all = {}
        for name, (a, bb) in TAXONOMY.items():
            v = zs_all[a].copy()
            if bb:
                v = 0.5 * (v + zs_all[bb])
            lab_all[name] = v
        base_assign = pd.DataFrame(lab_all).idxmax(axis=1).value_counts()

        print(f"\n  --- top {int(frac*100)}% hardest negatives (n={k}, "
              f"mean rank {sn[np.argsort(-sn)[:k]].mean():.4f} vs "
              f"{sn[np.argsort(sn)[:k]].mean():.4f} for the easiest) ---")
        print("    descriptor gap (hard vs all negatives, IQR units):")
        for nme, v in gap.head(6).items():
            print(f"      {nme:24s} {v:+.2f}")
        print("    mechanism mix (heuristic argmax assignment):")
        rows = []
        for nme in TAXONOMY:
            hs = int(assign.get(nme, 0)) / k
            bs = int(base_assign.get(nme, 0)) / len(neg)
            rows.append((nme, hs, bs, hs - bs))
            print(f"      {nme:24s} hard {100*hs:5.1f}%   all-neg {100*bs:5.1f}%   "
                  f"lift {100*(hs-bs):+5.1f}pp")
        out["groups"][f"top{int(frac*100)}pct"] = {
            "n": k,
            "mean_rank_hard": float(sn[np.argsort(-sn)[:k]].mean()),
            "mean_rank_easy": float(sn[np.argsort(sn)[:k]].mean()),
            "descriptor_gap_iqr": {k2: float(v) for k2, v in gap.items()},
            "mechanism_mix": {n2: {"hard_share": h, "all_neg_share": b2, "lift": d2}
                              for n2, h, b2, d2 in rows},
            "hard_series_ids": [int(x) for x in hard[:40]],
        }
    F.to_parquet(f"{REPORTS}/wave5_series_forensics.parquet")
    np.save(f"{ROOT}/research/oof/wave5_series_severity.npy", sev)
    return out


# ---------------------------------------------------------------- D3
def d3_age_profile(c, P, Sv, Bv):
    print("\n=== D3  TS-AUC by post-break age ===")
    arms = {"A_single": P["RT-300"], "B_seedclone7": Bv, "S_specialist7": Sv}
    res = {}
    for k, v in arms.items():
        res[k] = c.score_by_age(v)
    keys = list(res["A_single"])
    print(f"  {'age':>8s} " + "".join(f"{k:>14s}" for k in arms) +
          f"{'S-B':>10s}{'S-A':>10s}{'n_pos':>10s}")
    for kk in keys:
        a = res["A_single"][kk]["ts_auc"]; b = res["B_seedclone7"][kk]["ts_auc"]
        s = res["S_specialist7"][kk]["ts_auc"]
        print(f"  {kk:>8s} " + f"{a:>14.5f}{b:>14.5f}{s:>14.5f}"
              f"{s-b:>+10.5f}{s-a:>+10.5f}{res['A_single'][kk]['n_pos']:>10d}")
    return res


def main():
    t0 = time.time()
    c = Ctx()
    P = load_oof(sorted(set(SPECIALISTS) | set(SEEDCLONES)))
    Sv = np.load(f"{ROOT}/research/oof/wave5_S_specialist.npy")
    Bv = np.load(f"{ROOT}/research/oof/wave5_B_seedclone.npy")

    out = {"D1_metric_geometry": d1_metric_geometry(c),
           "D2_false_positives": d2_false_positives(c, Sv),
           "D3_age_profile": d3_age_profile(c, P, Sv, Bv)}
    json.dump(out, open(f"{REPORTS}/wave5_diagnostics.json", "w"), indent=2, default=float)
    print(f"\nwrote research/reports/wave5_diagnostics.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
