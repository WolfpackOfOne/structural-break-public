"""Pilot 1: specialist competence + failure-manifold diagnostics.

Diagnostic only: no RT ID, no candidate score, no RESULTS.csv row.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SBR_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sbr.metric import ts_auc_flat
from sbr.store import load_store
from wave5_lib import Ctx, SPECIALISTS, load_oof
from wave8_common import pair_repair_stats

OUTDIR = ROOT / "research" / "reports" / "new_avenues_2026"
CACHEDIR = ROOT / "cache" / "novel_streams"
CACHEDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

DOMINANT_T_MIN = 200
DOMINANT_AGE_MIN = 100
MIN_SERIES_WEIGHT = 500.0


def finite_float(x):
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): finite_float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_float(v) for v in x]
    return x


def rollmean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[w:] - c[:-w]) / w


def ar_coefs(z: np.ndarray, p: int) -> np.ndarray:
    if len(z) < 10 * p + 10:
        return np.zeros(p, dtype=np.float64)
    x = np.column_stack([z[p - k - 1 : len(z) - k - 1] for k in range(p)])
    y = z[p:]
    ridge = 1e-6 * np.eye(p) * len(y)
    return np.linalg.solve(x.T @ x + ridge, x.T @ y)


def perm_entropy(z: np.ndarray, m: int = 3) -> float:
    n = len(z) - m + 1
    if n < 10:
        return float("nan")
    w = np.column_stack([z[i : i + n] for i in range(m)])
    ranks = np.argsort(np.argsort(w, axis=1), axis=1)
    code = ranks[:, 0] * 9 + ranks[:, 1] * 3 + ranks[:, 2]
    cnt = np.bincount(code, minlength=27).astype(float)
    cnt = cnt[cnt > 0] / n
    return float(-(cnt * np.log(cnt)).sum() / np.log(6))


def history_fingerprints(force: bool = False) -> tuple[np.ndarray, list[str]]:
    cache = CACHEDIR / "pilot01_fingerprints.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return z["F"], list(z["names"])

    st = load_store(str(ROOT / "cache" / "store"))
    names = [
        "n_hist",
        "kurt",
        "skew",
        "hill",
        "ar1",
        "ar2",
        "ar3",
        "ar4",
        "ar5",
        "ar_sum",
        "acf1_sq",
        "acf1_abs",
        "vr10",
        "vr50",
        "spec_slope",
        "perm_ent",
        "turn_rate",
        "max_absz64",
        "exc_max_run64",
        "exc_n64",
        "max_logvr128",
        "q_ratio",
        "zerocross",
    ]
    f = np.full((len(st.meta), len(names)), np.nan, dtype=np.float64)
    t0 = time.time()
    for i in range(len(st.meta)):
        h = np.asarray(st.hist(i), dtype=np.float64)
        n = len(h)
        z = (h - h.mean()) / max(h.std(ddof=1), 1e-9)
        d = z - z.mean()
        v = float(d @ d / n)
        kurt = float(((d**4).mean()) / max(v * v, 1e-18))
        skew = float(((d**3).mean()) / max(v**1.5, 1e-18))
        a = np.sort(np.abs(z))[::-1]
        m = max(int(0.025 * n), 20)
        hill = float(m / np.log(a[:m] / a[m]).sum()) if m < len(a) and a[m] > 0 else np.nan
        ar = ar_coefs(z, 5)
        zs = z * z
        zs = zs - zs.mean()
        acf1sq = float((zs[1:] @ zs[:-1]) / max(zs @ zs, 1e-18))
        za = np.abs(z)
        za = za - za.mean()
        acf1abs = float((za[1:] @ za[:-1]) / max(za @ za, 1e-18))

        def vr(q):
            m2 = rollmean(z, q) * q
            return float(m2.var() / max(q * z.var(), 1e-18))

        nn = 1 << int(np.floor(np.log2(n)))
        p = np.abs(np.fft.rfft(z[:nn])) ** 2
        fq = np.arange(1, len(p))
        sel = fq <= max(len(fq) // 4, 8)
        spec_slope = float(np.polyfit(np.log(fq[sel]), np.log(np.maximum(p[1:][sel], 1e-30)), 1)[0])
        pe = perm_entropy(z[: min(n, 4000)])
        dz = np.diff(z)
        turn = float(np.mean(dz[1:] * dz[:-1] < 0))
        zerocross = float(np.mean(z[1:] * z[:-1] < 0))
        r64 = rollmean(z, 64)
        sd64 = r64.std()
        hot = np.abs(r64) > 2 * sd64
        if hot.any():
            ch = np.diff(np.r_[0, hot.astype(np.int8), 0])
            starts = np.flatnonzero(ch == 1)
            ends = np.flatnonzero(ch == -1)
            run = int((ends - starts).max())
            nev = len(starts)
        else:
            run, nev = 0, 0
        maxz64 = float(np.abs(r64).max() / max(sd64, 1e-12))
        lv = np.log(np.maximum(rollmean(z * z, 128), 1e-12))
        maxlvr = float(np.abs(lv - np.median(lv)).max())
        qs = np.quantile(z, [0.01, 0.25, 0.75, 0.99])
        qr = float((qs[3] - qs[0]) / max(qs[2] - qs[1], 1e-9))
        f[i] = [
            n,
            kurt,
            skew,
            hill,
            ar[0],
            ar[1],
            ar[2],
            ar[3],
            ar[4],
            ar.sum(),
            acf1sq,
            acf1abs,
            vr(10),
            vr(50),
            spec_slope,
            pe,
            turn,
            maxz64,
            run,
            nev,
            maxlvr,
            qr,
            zerocross,
        ]
        if i % 1000 == 0:
            print(f"fingerprints {i} {time.time() - t0:.0f}s", flush=True)
    np.savez(cache, F=f, names=np.array(names))
    return f, names


def cell_rows(c: Ctx, rows: np.ndarray) -> np.ndarray:
    y = c.d.y[rows]
    t = c.d.t[rows]
    age = c.age[rows]
    return rows[(t >= DOMINANT_T_MIN) & ((y == 0) | (age >= DOMINANT_AGE_MIN))]


def per_series_loss(score: np.ndarray, c: Ctx, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rr = cell_rows(c, rows)
    y = c.d.y[rr].astype(np.int8)
    t = c.d.t[rr]
    s = score[rr]
    sid = c.d.sidx[rr]

    order = np.lexsort((s, t))
    yo, to, so, sido = y[order], t[order], s[order], sid[order]
    gb = np.flatnonzero(np.r_[True, to[1:] != to[:-1]])
    ge = np.r_[gb[1:], len(to)]

    row_loss = np.zeros(len(to), dtype=np.float64)
    row_weight = np.zeros(len(to), dtype=np.float64)
    for lo, hi in zip(gb, ge):
        yy = yo[lo:hi]
        ss = so[lo:hi]
        pos = yy == 1
        neg = ~pos
        npos = int(pos.sum())
        nneg = int(neg.sum())
        if npos == 0 or nneg == 0:
            continue
        sp = ss[pos]
        sn = np.sort(ss[neg])
        left = np.searchsorted(sn, sp, side="left")
        right = np.searchsorted(sn, sp, side="right")
        conc_p = 0.5 * (left + right)
        idx = np.arange(lo, hi)
        row_loss[idx[pos]] = nneg - conc_p
        row_weight[idx[pos]] = nneg

        sp_sorted = np.sort(sp)
        left_n = np.searchsorted(sp_sorted, ss[neg], side="left")
        right_n = np.searchsorted(sp_sorted, ss[neg], side="right")
        conc_n = npos - 0.5 * (left_n + right_n)
        row_loss[idx[neg]] = npos - conc_n
        row_weight[idx[neg]] = npos

    n_series = int(c.d.sidx.max()) + 1
    sl = np.bincount(sido, weights=row_loss, minlength=n_series)
    sw = np.bincount(sido, weights=row_weight, minlength=n_series)
    return sl, sw


def within_t_rank(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    order = np.lexsort((x, t))
    out = np.empty(len(x), dtype=np.float64)
    ts = t[order]
    b = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    e = np.r_[b[1:], len(ts)]
    for lo, hi in zip(b, e):
        out[order[lo:hi]] = rankdata(x[order[lo:hi]]) / (hi - lo)
    return out


def qbins(x: np.ndarray, mask: np.ndarray, n: int = 5) -> np.ndarray:
    out = np.full(len(x), -1, dtype=np.int16)
    ok = mask & np.isfinite(x)
    qs = np.nanquantile(x[ok], np.linspace(0, 1, n + 1))
    qs = np.unique(qs)
    if len(qs) <= 2:
        out[ok] = 0
        return out
    b = np.digitize(x[ok], qs[1:-1], right=True)
    out[ok] = b
    return out


def standardize(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.array(x, dtype=np.float64, copy=True)
    med = np.nanmedian(z[mask], axis=0)
    scale = np.nanmedian(np.abs(z[mask] - med), axis=0) * 1.4826
    scale = np.where(scale > 1e-12, scale, np.nanstd(z[mask], axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (z - med) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    ok = np.isfinite(values) & (labels >= 0)
    if ok.sum() == 0:
        return float("nan")
    v = values[ok]
    lab = labels[ok]
    grand = float(v.mean())
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(lab):
        vg = v[lab == g]
        ss_between += len(vg) * float((vg.mean() - grand) ** 2)
    return ss_between / ss_total


def selector_by_fingerprint(
    fp: np.ndarray,
    valid_series: np.ndarray,
    folds: np.ndarray,
    losses: dict[str, np.ndarray],
    weights: np.ndarray,
    streams: dict[str, np.ndarray],
    c: Ctx,
) -> dict:
    out = {}
    dev_series = valid_series & (folds >= 0)
    base_rows = cell_rows(c, c.dev)
    base_y, base_t = c.d.y[base_rows], c.d.t[base_rows]
    base_auc = float(ts_auc_flat(streams["RT600"][base_rows], base_y, base_t))

    for j in range(fp.shape[1]):
        bins = qbins(fp[:, j], dev_series, 5)
        sel_score = np.full(len(c.d.y), np.nan, dtype=np.float64)
        chosen = {}
        for fold in range(5):
            for b in range(5):
                tr = dev_series & (folds != fold) & (bins == b)
                va_series = (folds == fold) & (bins == b)
                if weights[tr].sum() <= 0:
                    chosen[(fold, b)] = "RT600"
                    continue
                rates = {
                    s: float(losses[s][tr].sum() / max(weights[tr].sum(), 1.0))
                    for s in SPECIALISTS
                }
                pick = min(rates, key=rates.get)
                chosen[(fold, b)] = pick
                rows = c.rows[fold]
                m = va_series[c.d.sidx[rows]]
                sel_score[rows[m]] = streams[pick][rows[m]]
        # Conservative fallback: RT600 for any unassigned rows.
        miss = np.isnan(sel_score[c.dev])
        if miss.any():
            dev = c.dev
            sel_score[dev[miss]] = streams["RT600"][dev[miss]]
        auc = float(ts_auc_flat(sel_score[base_rows], base_y, base_t))
        out[j] = {
            "auc": auc,
            "delta_vs_rt600": auc - base_auc,
            "unique_selected_specialists": sorted(set(chosen.values())),
        }
    return out


def main() -> None:
    t0 = time.time()
    c = Ctx()
    folds = pd.read_parquet(ROOT / "research" / "folds" / "folds.parquet").fold.to_numpy()
    fp, fp_names = history_fingerprints()
    oof = load_oof(SPECIALISTS + ["RT-401"])
    rt600 = c.crossfit_blend(oof, SPECIALISTS)
    streams = {"RT600": rt600, **{s: oof[s] for s in SPECIALISTS}}

    rt600_mean, rt600_per_fold = c.score(rt600)
    rt600_pooled = c.pooled(rt600)
    dev_cell = cell_rows(c, c.dev)
    rt600_cell = float(ts_auc_flat(rt600[dev_cell], c.d.y[dev_cell], c.d.t[dev_cell]))

    print("computing per-series losses", flush=True)
    losses = {}
    weights = None
    for name, score in streams.items():
        sl, sw = per_series_loss(score, c, c.dev)
        losses[name] = sl
        if weights is None:
            weights = sw
    assert weights is not None

    dev_series = (folds >= 0) & (weights > MIN_SERIES_WEIGHT)
    never = dev_series & ~c.has_break
    brk = dev_series & c.has_break
    rt_rate = losses["RT600"] / np.maximum(weights, 1.0)

    # Specialist competence: positive advantage means specialist has lower loss than RT600.
    adv = np.column_stack(
        [(losses["RT600"] - losses[s]) / np.maximum(weights, 1.0) for s in SPECIALISTS]
    )
    winner = np.array(SPECIALISTS, dtype=object)[np.nanargmax(adv, axis=1)]
    best_adv = np.nanmax(adv, axis=1)

    selector = selector_by_fingerprint(fp, dev_series, folds, losses, weights, streams, c)

    rng = np.random.default_rng(0)
    fp_perm = fp.copy()
    for f in range(5):
        idx = np.flatnonzero((folds == f) & dev_series)
        fp_perm[idx] = fp_perm[rng.permutation(idx)]
    selector_perm = selector_by_fingerprint(fp_perm, dev_series, folds, losses, weights, streams, c)

    # Quintile competence summaries and permutation controls.
    qsum = []
    qsum_perm = []
    for j, nm in enumerate(fp_names):
        for arr, store in ((fp, qsum), (fp_perm, qsum_perm)):
            bins = qbins(arr[:, j], dev_series, 5)
            uniq_winners = set()
            spreads = []
            for b in range(5):
                m = dev_series & (bins == b)
                if weights[m].sum() <= 0:
                    continue
                rates = np.array([losses[s][m].sum() / max(weights[m].sum(), 1.0) for s in SPECIALISTS])
                uniq_winners.add(SPECIALISTS[int(np.argmin(rates))])
                spreads.append(float(rates.max() - rates.min()))
            store.append(
                {
                    "fingerprint": nm,
                    "unique_winners": len(uniq_winners),
                    "mean_spread": float(np.mean(spreads)) if spreads else 0.0,
                    "max_selector_delta": selector[j]["delta_vs_rt600"] if arr is fp else selector_perm[j]["delta_vs_rt600"],
                }
            )
    top_argmax = sorted(qsum, key=lambda x: (x["unique_winners"], x["mean_spread"]), reverse=True)[:8]
    top_selector = sorted(
        [{"fingerprint": fp_names[j], **v} for j, v in selector.items()],
        key=lambda x: x["delta_vs_rt600"],
        reverse=True,
    )[:8]
    top_selector_perm = sorted(
        [{"fingerprint": fp_names[j], **v} for j, v in selector_perm.items()],
        key=lambda x: x["delta_vs_rt600"],
        reverse=True,
    )[:8]

    monotone = []
    for j, nm in enumerate(fp_names):
        x = fp[:, j]
        for pop_name, mask in (("never_break", never), ("break", brk), ("all", dev_series)):
            ok = mask & np.isfinite(x)
            if ok.sum() < 100:
                continue
            rho_loss, p_loss = spearmanr(x[ok], rt_rate[ok])
            rho_best, p_best = spearmanr(x[ok], best_adv[ok])
            monotone.append(
                {
                    "fingerprint": nm,
                    "population": pop_name,
                    "rho_with_rt600_loss_rate": float(rho_loss),
                    "p_loss": float(p_loss),
                    "rho_with_best_specialist_advantage": float(rho_best),
                    "p_best_adv": float(p_best),
                }
            )
    top_loss_monotone = sorted(monotone, key=lambda x: abs(x["rho_with_rt600_loss_rate"]), reverse=True)[:12]
    top_adv_monotone = sorted(
        monotone, key=lambda x: abs(x["rho_with_best_specialist_advantage"]), reverse=True
    )[:12]

    # Hard never-break stratum check.
    nb_rate = rt_rate[never]
    nb_cut = float(np.quantile(nb_rate, 0.90))
    hard_nb = never.copy()
    hard_nb[never] = nb_rate >= nb_cut
    hard_univariate = []
    for j, nm in enumerate(fp_names):
        x = fp[:, j]
        ok = never & np.isfinite(x)
        if ok.sum() < 100:
            continue
        hard = hard_nb[ok]
        pooled_sd = max(np.nanstd(x[ok]), 1e-12)
        smd = float((np.nanmean(x[ok][hard]) - np.nanmean(x[ok][~hard])) / pooled_sd)
        rho, p = spearmanr(x[ok], rt_rate[ok])
        hard_univariate.append({"fingerprint": nm, "smd_top_decile": smd, "rho_loss": float(rho), "p": float(p)})
    hard_univariate = sorted(hard_univariate, key=lambda x: abs(x["smd_top_decile"]), reverse=True)[:12]

    # Failure manifolds: fixed k in {2,3}, cluster fingerprints plus loss/advantage cube.
    fp_z = standardize(fp, dev_series)
    loss_cube = np.column_stack([rt_rate, best_adv, adv])
    loss_z = standardize(loss_cube, dev_series)
    x_cluster = np.column_stack([fp_z, loss_z])
    idx = np.flatnonzero(dev_series)
    clusters = {}
    for k in (2, 3):
        km = KMeans(n_clusters=k, random_state=0, n_init=20)
        labels_sub = km.fit_predict(x_cluster[idx])
        labels = np.full(len(fp), -1, dtype=np.int16)
        labels[idx] = labels_sub
        sil = float(silhouette_score(x_cluster[idx], labels_sub))
        summaries = []
        for g in range(k):
            m = dev_series & (labels == g)
            summaries.append(
                {
                    "cluster": int(g),
                    "n_series": int(m.sum()),
                    "never_break_share": float((~c.has_break[m]).mean()),
                    "mean_rt600_loss_rate": float(rt_rate[m].mean()),
                    "mean_best_specialist_advantage": float(best_adv[m].mean()),
                    "winner_mode": str(pd.Series(winner[m]).mode().iloc[0]),
                }
            )
        clusters[str(k)] = {
            "silhouette": sil,
            "eta2_rt600_loss_rate": float(eta_squared(rt_rate, labels)),
            "eta2_best_specialist_advantage": float(eta_squared(best_adv, labels)),
            "summaries": summaries,
        }

    # Specialist correlations and sampled pair flow.
    rr = dev_cell
    tt = c.d.t[rr]
    rank_base = within_t_rank(rt600[rr], tt)
    corr = {}
    pair_flow = {}
    for s in SPECIALISTS:
        sr = within_t_rank(oof[s][rr], tt)
        corr[s] = float(np.corrcoef(rank_base, sr)[0, 1])
        pair_flow[s] = pair_repair_stats(rt600, oof[s], c.d.y, c.d.t, rr, n_pairs_per_t=20, seed=0)
    disagreement = np.std(np.column_stack([within_t_rank(oof[s][rr], tt) for s in SPECIALISTS]), axis=1)
    series_dis = np.bincount(c.d.sidx[rr], weights=disagreement, minlength=len(c.has_break)) / np.maximum(
        np.bincount(c.d.sidx[rr], minlength=len(c.has_break)), 1
    )
    dis_rho, dis_p = spearmanr(series_dis[dev_series], rt_rate[dev_series])

    best_selector = top_selector[0]
    best_perm = top_selector_perm[0]
    gates = {
        "selector_delta": best_selector["delta_vs_rt600"],
        "permutation_selector_delta": best_perm["delta_vs_rt600"],
        "selector_clears_001": best_selector["delta_vs_rt600"] >= 0.0010,
        "argmax_varies": max(x["unique_winners"] for x in qsum) > max(x["unique_winners"] for x in qsum_perm),
        "cluster_eta2_loss_max": max(v["eta2_rt600_loss_rate"] for v in clusters.values()),
        "top_abs_adv_spearman": max(abs(x["rho_with_best_specialist_advantage"]) for x in monotone),
    }
    if gates["selector_delta"] >= 0.0030 and gates["top_abs_adv_spearman"] >= 0.10:
        verdict = "HIGH-PRIORITY"
    elif gates["selector_delta"] >= 0.0020 or (
        gates["selector_delta"] >= 0.0010 and gates["selector_delta"] > gates["permutation_selector_delta"]
    ):
        verdict = "OPEN"
    elif gates["selector_delta"] >= 0.0010 or gates["top_abs_adv_spearman"] >= 0.10:
        verdict = "WEAK"
    else:
        verdict = "CLOSED"

    result = {
        "generated": "2026-08-24",
        "branch": "research/new-avenues-pilots-2026",
        "runtime_s": time.time() - t0,
        "rt600_reproduction": {
            "mean": rt600_mean,
            "per_fold": rt600_per_fold,
            "pooled": rt600_pooled,
            "dominant_cell_auc": rt600_cell,
        },
        "population": {
            "dev_series_with_weight": int(dev_series.sum()),
            "never_break_series_with_weight": int(never.sum()),
            "break_series_with_weight": int(brk.sum()),
            "min_series_weight": MIN_SERIES_WEIGHT,
        },
        "top_selector_oof_headroom": top_selector,
        "top_selector_permutation_control": top_selector_perm,
        "top_argmax_variation_by_fingerprint": top_argmax,
        "top_loss_monotone": top_loss_monotone,
        "top_specialist_advantage_monotone": top_adv_monotone,
        "hard_never_break_top_decile_fingerprints": hard_univariate,
        "clusters": clusters,
        "within_t_rank_corr_vs_rt600": corr,
        "specialist_pair_flow_vs_rt600_dominant_cell": pair_flow,
        "specialist_disagreement": {
            "spearman_series_disagreement_vs_rt600_loss": float(dis_rho),
            "p": float(dis_p),
        },
        "gates": gates,
        "verdict": verdict,
    }
    json_path = OUTDIR / "pilot01_failure_manifolds.json"
    md_path = OUTDIR / "pilot01_failure_manifolds.md"
    json_path.write_text(json.dumps(finite_float(result), indent=2, sort_keys=True) + "\n")

    md = [
        "# PILOT 1 -- SPECIALIST COMPETENCE AND FAILURE MANIFOLDS",
        "",
        f"Generated: 2026-08-24 on `research/new-avenues-pilots-2026`.",
        "",
        "## RT-600 Anchor",
        "",
        f"* Dev mean TS-AUC: `{rt600_mean:.6f}`.",
        f"* Dev pooled TS-AUC: `{rt600_pooled:.6f}`.",
        f"* Dominant-cell AUC: `{rt600_cell:.6f}`.",
        "",
        "## Primary Diagnostic Verdict",
        "",
        f"Verdict: **{verdict}**.",
        "",
        f"Best fold-held history-only selector headroom: `{best_selector['delta_vs_rt600']:+.6f}` "
        f"dominant-cell AUC via `{best_selector['fingerprint']}`.",
        f"Best fold-preserving permutation-control selector headroom: `{best_perm['delta_vs_rt600']:+.6f}`.",
        "",
        "The diagnostic does not allocate an RT ID and does not update `RESULTS.csv`.",
        "",
        "## Specialist Selector Headroom",
        "",
        "| fingerprint | delta vs RT600 | AUC | selected specialists |",
        "|---|---:|---:|---|",
    ]
    for row in top_selector[:8]:
        md.append(
            f"| `{row['fingerprint']}` | {row['delta_vs_rt600']:+.6f} | "
            f"{row['auc']:.6f} | {', '.join(row['unique_selected_specialists'])} |"
        )
    md += [
        "",
        "## Strongest Monotone Fingerprint Signals",
        "",
        "| fingerprint | population | rho with RT600 loss | rho with best specialist advantage |",
        "|---|---|---:|---:|",
    ]
    for row in top_adv_monotone[:10]:
        md.append(
            f"| `{row['fingerprint']}` | {row['population']} | "
            f"{row['rho_with_rt600_loss_rate']:+.4f} | "
            f"{row['rho_with_best_specialist_advantage']:+.4f} |"
        )
    md += [
        "",
        "## Hard Never-Break Strata",
        "",
        "| fingerprint | top-decile SMD | rho with loss |",
        "|---|---:|---:|",
    ]
    for row in hard_univariate[:10]:
        md.append(f"| `{row['fingerprint']}` | {row['smd_top_decile']:+.4f} | {row['rho_loss']:+.4f} |")
    md += [
        "",
        "## Failure-Manifold Clusters",
        "",
        "| k | silhouette | eta2 loss | eta2 best advantage | cluster summaries |",
        "|---:|---:|---:|---:|---|",
    ]
    for k, row in clusters.items():
        summary = "; ".join(
            f"c{s['cluster']}: n={s['n_series']}, nb={s['never_break_share']:.2f}, "
            f"loss={s['mean_rt600_loss_rate']:.3f}, adv={s['mean_best_specialist_advantage']:+.4f}, "
            f"mode={s['winner_mode']}"
            for s in row["summaries"]
        )
        md.append(
            f"| {k} | {row['silhouette']:.4f} | {row['eta2_rt600_loss_rate']:.4f} | "
            f"{row['eta2_best_specialist_advantage']:.4f} | {summary} |"
        )
    md += [
        "",
        "## Specialist Pair Flow Vs RT-600, Dominant Cell",
        "",
        "| specialist | corr vs RT600 | repairs | damage | net |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in SPECIALISTS:
        pf = pair_flow[s]
        md.append(f"| `{s}` | {corr[s]:+.4f} | {pf['repairs']} | {pf['damage']} | {pf['net_pair_lift']} |")
    md += [
        "",
        "## Interpretation",
        "",
        "History-only fingerprints do carry weak monotone information about where RT-600 fails, "
        "but the fold-held selector headroom is read against a same-procedure permutation control. "
        "This pilot opens a future gate only if the real selector headroom and specialist-advantage "
        "monotonicity clear the preregistered thresholds.",
        "",
        f"Runtime: `{time.time() - t0:.1f}s`.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(json.dumps(finite_float({"verdict": verdict, "best_selector": best_selector, "best_perm": best_perm}), indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
