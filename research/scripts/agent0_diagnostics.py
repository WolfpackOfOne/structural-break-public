"""Champion diagnostics: stratified TS-AUC, persistence sweep, paired bootstrap.

Everything here is post-hoc on saved OOF prediction vectors, so it costs no
retraining and cannot leak: the OOF vector for fold f was produced by a model
that never saw fold f.
"""
import sys, json; sys.path.insert(0, "/home/claude/sb/src")
import numpy as np, pandas as pd
from sbr.pipeline import Data, apply_persistence
from sbr.metric import ts_auc_flat

d = Data()
dev = d.rows_for([0, 1, 2, 3, 4])
CH = sys.argv[1] if len(sys.argv) > 1 else "RT-100"
ch = np.load(f"/home/claude/sb/research/oof/{CH}.npy")
bl = np.load("/home/claude/sb/research/oof/RT-000.npy")
out = {}

def sc(p, rows):
    return float(ts_auc_flat(p[rows], d.y[rows], d.t[rows]))

out["champion_pooled"] = sc(ch, dev)
out["baseline0_pooled"] = sc(bl, dev)
out["per_fold"] = {str(f): sc(ch, d.rows_for([f])) for f in range(5)}

# --- metric-weight sensitivity: does the pair-count weighting drive the result?
for w in ("pairs", "equal", "alive"):
    out[f"weighting_{w}"] = float(ts_auc_flat(ch[dev], d.y[dev], d.t[dev], weighting=w))

# --- where the metric is won: by elapsed online index
buckets = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 1000)]
out["by_online_index"] = {}
for a, b in buckets:
    m = dev[(d.t[dev] >= a) & (d.t[dev] < b)]
    out["by_online_index"][f"{a}-{b}"] = {"ts_auc": sc(ch, m), "rows": int(len(m)),
                                          "baseline": sc(bl, m)}

# --- by series characteristics
folds = d.folds
tau = folds.tau_index.to_numpy(); non = folds.n_online.to_numpy()
tau_rel = np.where(tau >= 0, tau / non, np.nan)
sfold = d.series_fold
def series_subset(mask):
    keep = np.zeros(d.st.n_series, bool); keep[mask] = True
    # a timestep only has weight if both classes are present, so we must keep
    # ALL series and merely restrict which ones we evaluate: instead score the
    # subset against the full cross-section by nulling nothing -- simplest
    # honest version is to score only rows of the kept series.
    r = dev[keep[d.sidx[dev]]]
    return r
out["by_tau_quartile"] = {}
for q in range(4):
    lo, hi = np.nanquantile(tau_rel, [q / 4, (q + 1) / 4])
    m = np.flatnonzero((tau_rel >= lo) & (tau_rel <= hi) & (sfold >= 0))
    # positives from this tau bucket + all no-break series as the negative pool
    nb = np.flatnonzero((tau < 0) & (sfold >= 0))
    r = series_subset(np.r_[m, nb])
    out["by_tau_quartile"][f"q{q+1} ({lo:.2f}-{hi:.2f})"] = {"ts_auc": sc(ch, r), "n_break_series": int(len(m))}

out["by_online_length"] = {}
for lo, hi in [(0, 200), (200, 400), (400, 700), (700, 1000)]:
    m = np.flatnonzero((non >= lo) & (non < hi) & (sfold >= 0))
    r = series_subset(m)
    out["by_online_length"][f"{lo}-{hi}"] = {"ts_auc": sc(ch, r), "n_series": int(len(m))}

# --- break taxonomy stratification
try:
    tx = pd.read_parquet("/home/claude/sb/research/artifacts/break_taxonomy.parquet")
    col = "break_class" if "break_class" in tx.columns else None
    if col:
        tx = tx.set_index("id") if "id" in tx.columns else tx
        cls = tx[col].reindex(folds.id.to_numpy()).to_numpy()
        nb = np.flatnonzero((tau < 0) & (sfold >= 0))
        out["by_break_class"] = {}
        for c in pd.unique(cls[pd.notna(cls)]):
            m = np.flatnonzero((cls == c) & (tau >= 0) & (sfold >= 0))
            if len(m) < 30:
                continue
            r = series_subset(np.r_[m, nb])
            out["by_break_class"][str(c)] = {"ts_auc": sc(ch, r), "n_series": int(len(m))}
except Exception as e:
    out["by_break_class"] = f"unavailable: {e}"

# --- persistence sweep (the running-max question)
out["persistence"] = {}
for mode in ["none", "runmax", "decaymax:0.999", "decaymax:0.99", "decaymax:0.95",
             "ewma:0.1", "ewma:0.3", "ewma:0.6"]:
    p = apply_persistence(ch[dev].copy(), d, dev, mode)
    nb_rows = dev[(d.y[dev] == 0)]
    out["persistence"][mode] = {
        "ts_auc": float(ts_auc_flat(p, d.y[dev], d.t[dev])),
        "mean_score_negatives": float(p[d.y[dev] == 0].mean()),
        "mean_score_positives": float(p[d.y[dev] == 1].mean()),
    }

# --- paired series-level bootstrap vs baseline 0 and vs the m00-only model
def paired_boot(a, b, n=200, seed=0):
    rng = np.random.default_rng(seed)
    sids = np.flatnonzero(sfold >= 0)
    deltas = []
    off = d.st.orow_off; non_ = d.st.meta.n_online.to_numpy()
    for _ in range(n):
        pick = rng.choice(sids, len(sids), replace=True)
        rows = np.concatenate([np.arange(off[i], off[i] + non_[i]) for i in pick])
        tt = d.t[rows]; yy = d.y[rows]
        deltas.append(ts_auc_flat(a[rows], yy, tt) - ts_auc_flat(b[rows], yy, tt))
    dl = np.array(deltas)
    return {"delta_mean": float(dl.mean()), "ci95": [float(np.quantile(dl, .025)), float(np.quantile(dl, .975))],
            "frac_positive": float((dl > 0).mean())}

out["paired_vs_baseline0"] = paired_boot(ch, bl, n=120)
try:
    m00 = np.load("/home/claude/sb/research/oof/RT-101.npy")
    out["paired_vs_m00_only"] = paired_boot(ch, m00, n=120)
except Exception as e:
    out["paired_vs_m00_only"] = f"unavailable: {e}"

json.dump(out, open("/home/claude/sb/research/reports/champion_diagnostics.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
