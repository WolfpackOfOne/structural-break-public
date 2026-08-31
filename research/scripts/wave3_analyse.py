"""Wave-3 analysis: everything the brief requires reported for a serious candidate.

For a treatment/control pair it reports
  * per-fold TS-AUC and the paired delta
  * a paired SERIES-level bootstrap of the delta (VALIDATION_V2 section 6)
  * TS-AUC by post-break age bucket -- where the metric mass actually is
  * within-timestep rank correlation between the two score vectors
  * the deployable ensemble delta under a per-series logit average

"Deployable" is the binding word: the blend is a fixed per-series function of
that series' own scores, so it can execute inside a series-sequential,
single-pass `infer()`.  A within-timestep cross-sectional rank blend is computed
too, but only ever labelled ORACLE / DIAGNOSTIC -- it cannot be submitted (see
the deployability trap).

usage:  wave3_analyse.py CONTROL_ID TREATMENT_ID [ANCHOR_ID]
"""
from __future__ import annotations

import os, sys, json
import numpy as np

ROOT = os.environ.get("SBR_ROOT", "/home/claude/sb")
sys.path.insert(0, f"{ROOT}/src")

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, OOF

FOLDS = (0, 1, 2, 3, 4)
AGE_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 10 ** 9)]


def load_oof(exp_id: str) -> np.ndarray:
    return np.load(f"{OOF}/{exp_id}.npy")


def per_fold(d: Data, s: np.ndarray):
    out = []
    for f in FOLDS:
        r = d.rows_for([f])
        out.append(float(ts_auc_flat(s[r], d.y[r], d.t[r])))
    return out


def post_break_age(d: Data) -> np.ndarray:
    """age = t - tau for post-break rows, -1 for every never-broken row."""
    tau = d.st.meta["tau_index"].to_numpy()
    age = np.full(len(d.y), -1, dtype=np.int64)
    n_on = d.st.meta["n_online"].to_numpy()
    for i in range(d.st.n_series):
        ti = tau[i]
        if ti >= 0:
            s = d.st.orow_off[i]
            age[s + ti: s + n_on[i]] = np.arange(n_on[i] - ti)
    return age


def auc_by_age(d: Data, s: np.ndarray, age: np.ndarray, rows: np.ndarray):
    """TS-AUC using only positives in an age band, against ALL negatives."""
    out = {}
    y, t = d.y[rows], d.t[rows]
    a, sc = age[rows], s[rows]
    neg = y == 0
    for lo, hi in AGE_BUCKETS:
        sel = neg | ((y == 1) & (a >= lo) & (a < hi))
        if (y[sel] == 1).sum() < 50:
            continue
        lab = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        out[lab] = dict(ts_auc=float(ts_auc_flat(sc[sel], y[sel], t[sel])),
                        n_pos=int((y[sel] == 1).sum()))
    return out


def within_timestep_rank(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rank percentile of each row inside its own online index (ORACLE only)."""
    order = np.lexsort((s, t))
    ts = t[order]
    n = len(s)
    gstart = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    gend = np.r_[gstart[1:], n]
    out = np.empty(n)
    r = np.empty(n)
    r[order] = np.arange(n)
    for a, b in zip(gstart, gend):
        idx = order[a:b]
        out[idx] = (np.arange(b - a) + 0.5) / (b - a)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def paired_bootstrap(d: Data, a: np.ndarray, b: np.ndarray, rows: np.ndarray,
                     n_boot: int = 300, seed: int = 0):
    """Resample whole SERIES with replacement; report the delta distribution."""
    rng = np.random.default_rng(seed)
    sidx = d.sidx[rows]
    uniq = np.unique(sidx)
    # row slices per series so a resample can be assembled without a join
    order = np.argsort(sidx, kind="stable")
    srt = sidx[order]
    bnd = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1]])
    ends = np.r_[bnd[1:], len(srt)]
    slices = {int(srt[b]): order[b:e] for b, e in zip(bnd, ends)}
    ya, ta = d.y[rows], d.t[rows]
    aa, bb = a[rows], b[rows]
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([slices[int(s)] for s in pick])
        da = ts_auc_flat(aa[idx], ya[idx], ta[idx])
        db = ts_auc_flat(bb[idx], ya[idx], ta[idx])
        deltas.append(float(db - da))
    q = np.percentile(deltas, [2.5, 50, 97.5])
    return dict(ci_lo=float(q[0]), median=float(q[1]), ci_hi=float(q[2]),
                frac_favouring_treatment=float(np.mean(np.array(deltas) > 0)),
                n_boot=n_boot)


def main():
    ctl_id, trt_id = sys.argv[1], sys.argv[2]
    anchor_id = sys.argv[3] if len(sys.argv) > 3 else None

    d = Data()
    rows = d.rows_for(list(FOLDS))
    age = post_break_age(d)
    ctl, trt = load_oof(ctl_id), load_oof(trt_id)

    rep: dict = {"control": ctl_id, "treatment": trt_id}
    pf_c, pf_t = per_fold(d, ctl), per_fold(d, trt)
    rep["per_fold"] = {ctl_id: pf_c, trt_id: pf_t}
    rep["mean"] = {ctl_id: float(np.mean(pf_c)), trt_id: float(np.mean(pf_t))}
    delta = [b - a for a, b in zip(pf_c, pf_t)]
    rep["per_fold_delta"] = delta
    rep["mean_delta"] = float(np.mean(delta))
    rep["folds_positive"] = int(sum(x > 0 for x in delta))
    rep["delta_sd"] = float(np.std(delta))

    # PRE-REGISTERED falsification conditions
    rep["falsified"] = {
        "mean_delta_le_0": bool(rep["mean_delta"] <= 0),
        "positive_on_le_2_of_5_folds": bool(rep["folds_positive"] <= 2),
    }
    rep["verdict"] = ("REJECTED" if any(rep["falsified"].values()) else "SURVIVES PRE-REGISTRATION")

    rep["paired_series_bootstrap"] = paired_bootstrap(d, ctl, trt, rows)

    rep["ts_auc_by_post_break_age"] = {
        ctl_id: auc_by_age(d, ctl, age, rows),
        trt_id: auc_by_age(d, trt, age, rows),
    }

    t_rows = d.t[rows]
    rc = within_timestep_rank(ctl[rows], t_rows)
    rt_ = within_timestep_rank(trt[rows], t_rows)
    rep["within_timestep_rank_corr"] = float(np.corrcoef(rc, rt_)[0, 1])

    # DEPLOYABLE blend: a fixed per-series function of that series' own scores
    blend = np.full(len(d.y), np.nan, dtype=np.float64)
    blend[rows] = 0.5 * (logit(ctl[rows]) + logit(trt[rows]))
    pf_b = per_fold(d, blend)
    rep["deployable_logit_blend"] = {
        "per_fold": pf_b, "mean": float(np.mean(pf_b)),
        "delta_vs_control": float(np.mean(pf_b) - np.mean(pf_c)),
        "label": "DEPLOYABLE (per-series function of own scores)",
    }
    if anchor_id:
        anc = load_oof(anchor_id)
        pf_a = per_fold(d, anc)
        b2 = np.full(len(d.y), np.nan)
        b2[rows] = 0.5 * (logit(anc[rows]) + logit(trt[rows]))
        pf_b2 = per_fold(d, b2)
        rep["blend_with_anchor"] = {
            "anchor": anchor_id, "anchor_mean": float(np.mean(pf_a)),
            "blend_mean": float(np.mean(pf_b2)),
            "delta_vs_anchor": float(np.mean(pf_b2) - np.mean(pf_a)),
            "within_timestep_rank_corr_anchor_vs_treatment":
                float(np.corrcoef(within_timestep_rank(anc[rows], t_rows), rt_)[0, 1]),
        }

    out = f"{ROOT}/research/reports/wave3_{ctl_id}_vs_{trt_id}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
