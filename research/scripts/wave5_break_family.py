"""W5-D4 -- performance by BREAK FAMILY.

HEURISTIC TAXONOMY, AND LABELLED AS SUCH.  The wave-1 taxonomy
(research/reports/break_taxonomy.md) built a proper artifact with a placebo-null
threshold, but `research/artifacts/break_taxonomy.parquet` did not survive the
container that produced it -- the report says so itself -- so this is a lighter
reconstruction of its `lead_family` column: an argmax over standardised family
effect sizes, with no threshold, assigned to every break series.

It is a DIAGNOSTIC.  It uses tau and the post-break segment, which no production
feature may do, and nothing here selects a model.

Family effects are POST vs PRE where at least 60 pre-break online points exist
and POST vs HIST otherwise -- the same rule the wave-1 taxonomy used, so the
family shares are comparable to its section J.1.
"""
from __future__ import annotations

import json, time
import numpy as np
import pandas as pd

from wave5_lib import Ctx, load_oof, ROOT, REPORTS, SPECIALISTS, SEEDCLONES
from sbr.metric import ts_auc_flat
from sbr.transforms import HistParams, ar_filter_causal

FAMILIES = ("location", "scale", "tails", "dependence", "trend", "spectral")


def _acf1(v):
    v = v - v.mean()
    d = float(v @ v)
    return float(v[1:] @ v[:-1] / d) if d > 0 else 0.0


def _hf(v):
    return float(np.var(np.diff(v)) / max(np.var(v), 1e-12)) if len(v) > 2 else 0.0


def classify(c):
    """Standardised family effect sizes for every dev series, break or not.

    No-break series get a PLACEBO cut drawn from the break series' relative-tau
    distribution, so the same statistics exist for both groups and the family
    shares have a matched null -- exactly as the wave-1 taxonomy did.
    """
    st = c.d.st
    rng = np.random.default_rng(20260818)
    dev = np.unique(c.d.sidx[c.dev])
    tau = c.tau.copy()
    rel = tau[c.has_break] / np.maximum(c.n_online[c.has_break], 1)
    rel = rel[(rel > 0) & (rel < 1)]
    rows = []
    for i in dev:
        n = int(c.n_online[i])
        if n < 24:
            continue
        h, o = st.hist(i), st.online(i)
        cut = int(tau[i]) if c.has_break[i] else int(np.clip(
            rng.choice(rel) * n, 6, n - 6))
        if cut < 4 or n - cut < 4:
            continue
        pre, post = o[:cut], o[cut:]
        hp = HistParams(h, ar_order=6)
        ref = pre if len(pre) >= 60 else h
        zr = (ref - hp.mu) / hp.sd
        zp = (post - hp.mu) / hp.sd
        er = ar_filter_causal((ref - hp.mu) / hp.sd, hp.ar_coef, (h[-6:] - hp.mu) / hp.sd)
        ep = ar_filter_causal(zp, hp.ar_coef, (ref[-6:] - hp.mu) / hp.sd)
        nr, npo = len(zr), len(zp)
        se = np.sqrt(1.0 / nr + 1.0 / npo)
        q99 = float(np.quantile(np.abs(zr), 0.99))
        def slope(v):
            x = np.arange(len(v)) - (len(v) - 1) / 2.0
            return float(x @ v / max(x @ x, 1e-9)) * len(v)
        eff = {
            "location": abs(zp.mean() - zr.mean()) / max(se * zr.std(), 1e-9),
            "scale": abs(np.log(max(ep.std(), 1e-9) / max(er.std(), 1e-9))) / max(se, 1e-9),
            "tails": abs(np.mean(np.abs(zp) > q99) - 0.01) / max(se, 1e-9),
            "dependence": abs(_acf1(zp) - _acf1(zr)) / max(se, 1e-9),
            "trend": abs(slope(zp) - slope(zr)) / max(se * max(zr.std(), 1e-9) * npo, 1e-9),
            "spectral": abs(_hf(zp) - _hf(zr)) / max(se, 1e-9),
        }
        rows.append({"sidx": int(i), "has_break": bool(c.has_break[i]),
                     "lead_family": max(eff, key=eff.get), **eff})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    c = Ctx()
    P = load_oof(sorted(set(SPECIALISTS) | set(SEEDCLONES)))
    Sv = np.load(f"{ROOT}/research/oof/wave5_S_specialist.npy")
    Bv = np.load(f"{ROOT}/research/oof/wave5_B_seedclone.npy")
    F = classify(c)
    F.to_parquet(f"{REPORTS}/wave5_break_family.parquet")

    br = F.loc[F.has_break]
    nb = F.loc[~F.has_break]
    print("=== W5-D4  break-family mix (HEURISTIC argmax, no threshold) ===")
    print(f"  {len(br)} break series, {len(nb)} no-break with a placebo cut\n")
    sh = br.lead_family.value_counts(normalize=True)
    shn = nb.lead_family.value_counts(normalize=True)
    print(f"  {'family':>12s}{'break %':>10s}{'placebo %':>12s}{'excess':>10s}")
    for f in FAMILIES:
        a, b = 100 * sh.get(f, 0.0), 100 * shn.get(f, 0.0)
        print(f"  {f:>12s}{a:>10.1f}{b:>12.1f}{a-b:>+10.1f}")

    r = c.dev
    y, t = c.d.y[r], c.d.t[r]
    sid = c.d.sidx[r]
    fam = dict(zip(br.sidx.tolist(), br.lead_family.tolist()))
    famrow = np.array([fam.get(int(s), "") for s in sid])
    neg = y == 0
    arms = {"A_single": P["RT-300"], "B_seedclone7": Bv, "S_specialist7": Sv}
    print("\n=== TS-AUC by lead family (negatives held fixed) ===")
    print(f"  {'family':>12s}{'n_pos':>9s}" + "".join(f"{k:>15s}" for k in arms) +
          f"{'S-B':>10s}{'S-A':>10s}")
    out = {}
    for f in FAMILIES:
        m = neg | ((y == 1) & (famrow == f))
        npos = int(((y == 1) & (famrow == f)).sum())
        if npos < 500:
            continue
        v = {k: float(ts_auc_flat(x[r][m], y[m], t[m])) for k, x in arms.items()}
        out[f] = {"n_pos": npos, **v}
        print(f"  {f:>12s}{npos:>9d}" + "".join(f"{v[k]:>15.5f}" for k in arms) +
              f"{v['S_specialist7']-v['B_seedclone7']:>+10.5f}"
              f"{v['S_specialist7']-v['A_single']:>+10.5f}")
    json.dump({"mix_break": {k: float(v) for k, v in sh.items()},
               "mix_placebo": {k: float(v) for k, v in shn.items()},
               "ts_auc_by_family": out},
              open(f"{REPORTS}/wave5_break_family.json", "w"), indent=2)
    print(f"\nwrote research/reports/wave5_break_family.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
