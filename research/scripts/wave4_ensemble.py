"""W4-E1: is the seven-stream ensemble's gain SPECIALIST DIVERSITY or BAGGING?

Pre-registered in research/RDOF_LEDGER.md before the first arm was trained.

Two seven-member sets, sharing member 1 (RT-300, the champion at seed 0):

  SPECIALIST  RT-300, RT-410..RT-415   modules/depth/rows/sampling/objective/boosting/seed all differ
  SEED        RT-300, RT-401..RT-406   ONLY the seed differs

Every blend is CROSS-FITTED: the calibration for validation fold k is fitted on
OOF scores from folds != k, so fold k's own score distribution never informs
fold k's ranking.  Every blend family is reported; none is selected from.

The oracle within-timestep rank average is computed for both arms as a
DIAGNOSTIC CEILING only.  It is not deployable -- the Crunch runner is
series-sequential and single-pass, so the live cross-section at time t does not
exist at inference.  It is never a champion.
"""
from __future__ import annotations

import itertools, json, os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
from sbr.metric import ts_auc_flat
from sbr.pipeline import Data
from wave4_cal import SCDF_T, SCDF_NSEEN, GlobalCDF, logit, sigmoid
from wave4_lib import SEED_SET, SPECIALIST_SET, SPECIALIST_ALIAS, SEED_CLONES

FOLDS = (0, 1, 2, 3, 4)
SINGLE = "RT-300"
OOFDIR = f"{ROOT}/research/oof"


def within_t_rank(scores, t):
    """Percentile rank inside each timestep.  ORACLE / DIAGNOSTIC ONLY."""
    order = np.lexsort((scores, t))
    s, tt = scores[order], t[order]
    n = len(s)
    gstart = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1]])
    glen = np.r_[gstart[1:], n] - gstart
    pos = np.arange(n) - np.repeat(gstart, glen)
    new = np.r_[True, (tt[1:] != tt[:-1]) | (s[1:] != s[:-1])]
    rs = np.flatnonzero(new)
    rl = np.r_[rs[1:], n] - rs
    avg = np.repeat(pos[rs] + (rl - 1) / 2.0, rl)
    out = np.empty(n)
    out[order] = avg / np.maximum(np.repeat(glen, glen) - 1, 1)
    return out


class Blends:
    """All calibration families, each cross-fitted, on one set of streams."""

    FAMILIES = ("raw", "logitmean", "gcdf", "scdf_t", "scdf_nseen")

    def __init__(self, P, d, rows):
        self.P, self.d, self.rows = P, d, rows

    def _fit(self, kind, streams, tr):
        t_tr = self.d.t[tr]
        if kind == "raw":
            return lambda va: np.column_stack([self.P[s][va] for s in streams]).mean(1)
        if kind == "logitmean":
            return lambda va: sigmoid(np.column_stack(
                [logit(self.P[s][va]) for s in streams]).mean(1))
        if kind == "gcdf":
            fs = [GlobalCDF(self.P[s][tr]) for s in streams]
            return lambda va: np.column_stack([f(self.P[s][va]) for f, s in zip(fs, streams)]).mean(1)
        cls = SCDF_T if kind == "scdf_t" else SCDF_NSEEN
        fs = [cls(self.P[s][tr], t_tr) for s in streams]
        return lambda va: np.column_stack(
            [f(self.P[s][va], self.d.t[va]) for f, s in zip(fs, streams)]).mean(1)

    def crossfit(self, kind, streams):
        """Full-length score vector, fold k calibrated on folds != k."""
        out = np.full(len(self.d.y), np.nan)
        for k in FOLDS:
            tr = np.concatenate([self.rows[g] for g in FOLDS if g != k])
            out[self.rows[k]] = self._fit(kind, streams, tr)(self.rows[k])
        return out


def score(v, d, rows):
    per = [float(ts_auc_flat(v[rows[k]], d.y[rows[k]], d.t[rows[k]])) for k in FOLDS]
    return float(np.mean(per)), per


def main():
    t0 = time.time()
    d = Data()
    rows = {f: d.rows_for([f]) for f in FOLDS}
    dev = d.rows_for(list(FOLDS))

    need = sorted(set(SEED_SET) | set(SPECIALIST_SET))
    missing = [e for e in need if not os.path.exists(f"{OOFDIR}/{e}.npy")]
    if missing:
        sys.exit(f"MISSING OOF: {missing}")
    P = {e: np.load(f"{OOFDIR}/{e}.npy") for e in need}

    res = {"folds": list(FOLDS), "arms": {},
           "seed_list_preregistered": [SEED_CLONES[e] for e in SEED_SET],
           "specialist_alias": SPECIALIST_ALIAS}

    # ---------------------------------------------------------- individuals
    print("=== individual streams ===")
    res["individual"] = {}
    for e in need:
        m, pf = score(P[e], d, rows)
        res["individual"][e] = {"mean": m, "per_fold": pf}
        tag = SPECIALIST_ALIAS.get(e, "") if e in SPECIALIST_SET else f"seed {SEED_CLONES.get(e)}"
        print(f"  {e}  {m:.5f}   {tag}")

    # ------------------------------------------------------------- the arms
    B = Blends(P, d, rows)
    for arm, streams in (("specialist", SPECIALIST_SET), ("seed_clone", SEED_SET)):
        print(f"\n=== {arm.upper()} arm ===")
        ind = [res["individual"][e]["mean"] for e in streams]
        a = {"streams": list(streams),
             "individual_mean": float(np.mean(ind)),
             "individual_std": float(np.std(ind)),
             "individual_min": float(np.min(ind)), "individual_max": float(np.max(ind)),
             "blends": {}, "vectors": {}}
        print(f"  members {np.mean(ind):.5f} +/- {np.std(ind):.5f} "
              f"[{np.min(ind):.5f}, {np.max(ind):.5f}]")
        for kind in Blends.FAMILIES:
            v = B.crossfit(kind, streams)
            m, pf = score(v, d, rows)
            a["blends"][kind] = {"mean": m, "per_fold": pf}
            a["vectors"][kind] = v
            print(f"  {kind:12s} {m:.5f}   " + " ".join(f"{x:.5f}" for x in pf))
        # oracle ceiling, diagnostic only
        orc = np.full(len(d.y), np.nan)
        orc[dev] = np.column_stack([within_t_rank(P[s][dev], d.t[dev]) for s in streams]).mean(1)
        m, pf = score(orc, d, rows)
        a["oracle_within_t_rank_ILLEGAL"] = {"mean": m, "per_fold": pf}
        print(f"  {'ORACLE(illegal)':12s} {m:.5f}")
        # correlations
        rk = {s: within_t_rank(P[s][dev], d.t[dev]) for s in streams}
        wt, gl = [], []
        pair = {}
        for x, y in itertools.combinations(streams, 2):
            w = float(np.corrcoef(rk[x], rk[y])[0, 1])
            g = float(np.corrcoef(P[x][dev], P[y][dev])[0, 1])
            pair[f"{x}|{y}"] = {"within_t_rank": w, "global_pearson": g}
            wt.append(w); gl.append(g)
        a["correlations"] = {"pairwise": pair,
                             "within_t_rank_mean": float(np.mean(wt)),
                             "within_t_rank_min": float(np.min(wt)),
                             "within_t_rank_max": float(np.max(wt)),
                             "global_pearson_mean": float(np.mean(gl))}
        print(f"  within-t rank corr  mean {np.mean(wt):.4f}  "
              f"range [{np.min(wt):.4f}, {np.max(wt):.4f}]")
        res["arms"][arm] = a

    # -------------------------------------------------- the headline verdict
    S = res["arms"]["specialist"]["blends"]["scdf_t"]
    C = res["arms"]["seed_clone"]["blends"]["scdf_t"]
    diff = S["mean"] - C["mean"]
    per = [a - b for a, b in zip(S["per_fold"], C["per_fold"])]
    single_m, single_pf = score(P[SINGLE], d, rows)
    res["verdict"] = {
        "specialist_scdf_t": S["mean"], "seed_clone_scdf_t": C["mean"],
        "difference": diff, "per_fold_difference": per,
        "n_folds_specialist_better": int(sum(x > 0 for x in per)),
        "single_champion_RT300": single_m,
        "specialist_gain_over_single": S["mean"] - single_m,
        "seed_clone_gain_over_single": C["mean"] - single_m,
        "bagging_share_of_gain": (C["mean"] - single_m) / max(S["mean"] - single_m, 1e-12),
        "prereg_bar": {"threshold": 0.0030, "min_folds": 4},
        "prereg_outcome": ("H1 CONFIRMED: specialist diversity beats bagging"
                           if diff > 0.0030 and sum(x > 0 for x in per) >= 4 else
                           "INDETERMINATE, resolved to the simpler system"
                           if diff > 0.0010 else
                           "H0 ACCEPTED: the gain is ordinary bagging"),
    }

    # --------------------------------------------- paired bootstrap, common RNG
    print("\n=== paired series bootstrap (200 reps, common random numbers) ===")
    sid = d.sidx[dev]
    order = np.argsort(sid, kind="stable")
    sid_s = sid[order]
    bnd = np.flatnonzero(np.r_[True, sid_s[1:] != sid_s[:-1]])
    ends = np.r_[bnd[1:], len(sid_s)]
    srows = {sid_s[b]: dev[order[b:e]] for b, e in zip(bnd, ends)}
    uniq = np.unique(sid)
    rng = np.random.default_rng(0)
    picks = [rng.choice(uniq, len(uniq), replace=True) for _ in range(200)]

    vec = {"specialist": res["arms"]["specialist"]["vectors"]["scdf_t"],
           "seed_clone": res["arms"]["seed_clone"]["vectors"]["scdf_t"],
           "single": P[SINGLE]}
    boot = {k: [] for k in ("spec_minus_seed", "spec_minus_single", "seed_minus_single")}
    for pk in picks:
        r = np.concatenate([srows[s] for s in pk])
        y, tt = d.y[r], d.t[r]
        a = ts_auc_flat(vec["specialist"][r], y, tt)
        b = ts_auc_flat(vec["seed_clone"][r], y, tt)
        c = ts_auc_flat(vec["single"][r], y, tt)
        boot["spec_minus_seed"].append(float(a - b))
        boot["spec_minus_single"].append(float(a - c))
        boot["seed_minus_single"].append(float(b - c))
    res["bootstrap"] = {}
    for k, v in boot.items():
        v = np.array(v)
        res["bootstrap"][k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                               "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                               "fraction_positive": float((v > 0).mean())}
        print(f"  {k:20s} {v.mean():+.5f}  CI [{np.quantile(v,.025):+.5f}, "
              f"{np.quantile(v,.975):+.5f}]  {100*(v>0).mean():.0f}% positive")

    print(f"\nVERDICT: specialist {S['mean']:.5f}  seed-clone {C['mean']:.5f}  "
          f"diff {diff:+.5f}  ({res['verdict']['n_folds_specialist_better']}/5 folds)")
    print(res["verdict"]["prereg_outcome"])

    os.makedirs(f"{ROOT}/research/reports", exist_ok=True)
    np.savez_compressed(f"{ROOT}/research/oof/wave4_blend_vectors.npz",
                        **{f"{arm}__{k}": v
                           for arm in res["arms"] for k, v in res["arms"][arm]["vectors"].items()})
    for arm in res["arms"]:
        del res["arms"][arm]["vectors"]
    json.dump(res, open(f"{ROOT}/research/reports/wave4_specialist_vs_bagging.json", "w"), indent=2)
    print(f"\nwrote research/reports/wave4_specialist_vs_bagging.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
