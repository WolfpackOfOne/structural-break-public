"""MISSION 1: settle the persistence question.

One fixed strong base model (lgb_all, trained once on screen folds 1-4) predicts
screen fold 0.  Every persistence rule is then a pure post-transform of that one
score vector, so all comparisons are exactly paired.
"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/sb/scripts")
import agent13_common as C
from sbr.metric import ts_auc_flat

d = C.data()
rows = np.load(f"{C.ART}/rows_outer_va.npy")
P = np.load(f"{C.ART}/preds_lgb_all.npz")["outer"]
y = d.y[rows]; t = d.t[rows]; sid = d.sidx[rows]
tau = d.st.meta.tau_index.to_numpy()[sid]
has_break = tau >= 0
b, e = C.series_bounds(rows, d)

# ------------------------------------------------------------------ transforms
def per_series(p, fn):
    out = np.empty_like(p)
    for i, j in zip(b, e):
        out[i:j] = fn(p[i:j])
    return out

def f_runmax(v):
    return np.maximum.accumulate(v)

def f_decaymax(g):
    def f(v):
        m = -np.inf; r = np.empty_like(v)
        for i, x in enumerate(v):
            m = x if x > m * g else m * g
            r[i] = m
        return r
    return f

def f_ewma(a):
    def f(v):
        m = v[0]; r = np.empty_like(v)
        for i, x in enumerate(v):
            m = (1 - a) * m + a * x; r[i] = m
        return r
    return f

def f_hyst(hi, lo, boost):
    """Latch on when p>hi, release when p<lo; latched rows get +boost (rank tier)."""
    def f(v):
        z = 0; r = np.empty_like(v)
        for i, x in enumerate(v):
            if z == 0 and x > hi:
                z = 1
            elif z == 1 and x < lo:
                z = 0
            r[i] = x + boost * z
        return r
    return f

def f_tot(thr, w, K):
    """Time-over-threshold: additive credit for how long p has stayed above thr."""
    def f(v):
        k = 0; r = np.empty_like(v)
        for i, x in enumerate(v):
            k = k + 1 if x > thr else 0
            r[i] = x + w * min(1.0, k / K)
        return r
    return f

def f_tot_frac(thr, w):
    def f(v):
        c = np.cumsum(v > thr) / np.arange(1, len(v) + 1)
        return v + w * c
    return f

def f_hmm(h, lam, prior=0.2):
    """Absorbing-state forward filter.  p_t is read as P(break|x_t); the implied
    likelihood ratio is tempered by lam to fight the massive serial dependence
    between consecutive feature vectors."""
    def f(v):
        q = np.clip(v.astype(np.float64), 1e-6, 1 - 1e-6)
        lr = ((q / (1 - q)) * ((1 - prior) / prior)) ** lam
        a = 0.0; r = np.empty(len(v), np.float64)
        for i in range(len(v)):
            ap = a + (1 - a) * h
            num = ap * lr[i]
            a = num / (num + (1 - ap))
            r[i] = a
        return r.astype(v.dtype)
    return f

TRANSFORMS = [("raw", None), ("runmax", f_runmax)]
for g in (0.999, 0.995, 0.99, 0.98, 0.95, 0.90):
    TRANSFORMS.append((f"decaymax:{g}", f_decaymax(g)))
for a in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
    TRANSFORMS.append((f"ewma:{a}", f_ewma(a)))
for hi, lo, bo in ((0.6, 0.3, 1.0), (0.8, 0.4, 1.0), (0.8, 0.4, 0.2), (0.9, 0.5, 1.0)):
    TRANSFORMS.append((f"hyst:{hi}/{lo}/+{bo}", f_hyst(hi, lo, bo)))
for thr, w, K in ((0.5, 1.0, 10), (0.5, 0.3, 20), (0.7, 1.0, 10), (0.7, 0.3, 30)):
    TRANSFORMS.append((f"tot:{thr}/w{w}/K{K}", f_tot(thr, w, K)))
for thr, w in ((0.5, 0.5), (0.7, 0.5)):
    TRANSFORMS.append((f"totfrac:{thr}/w{w}", f_tot_frac(thr, w)))
for h in (1 / 500.0, 1 / 200.0):
    for lam in (0.03, 0.1, 0.25, 0.5, 1.0):
        TRANSFORMS.append((f"hmm:h{h:.4f}/lam{lam}", f_hmm(h, lam)))

# ------------------------------------------------------------------ evaluation
BUCKETS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 320), (320, 1000)]

def report(p):
    r = {"overall": ts_auc_flat(p, y, t)}
    for lo, hi in BUCKETS:
        m = (t >= lo) & (t < hi)
        if m.sum() and y[m].sum() and (~y[m].astype(bool)).sum():
            r[f"t{lo}-{hi}"] = ts_auc_flat(p[m], y[m], t[m])
        else:
            r[f"t{lo}-{hi}"] = np.nan
    # positives vs NO-BREAK-series negatives  (ratchet damage lives here)
    m = (y == 1) | (~has_break)
    r["vs_nobreak"] = ts_auc_flat(p[m], y[m], t[m])
    # positives vs PRE-break negatives of breaking series
    m = (y == 1) | (has_break & (y == 0))
    r["vs_prebreak"] = ts_auc_flat(p[m], y[m], t[m])
    # ratchet index: mean rise of the score over the life of a NO-BREAK series
    nb = ~has_break
    rise = []
    for i, j in zip(b, e):
        if not nb[i]:
            continue
        v = p[i:j]
        if len(v) >= 20:
            rise.append(float(v[-10:].mean() - v[:10].mean()))
    r["nobreak_drift"] = float(np.mean(rise))
    return r

out = {}
for name, fn in TRANSFORMS:
    p = P.astype(np.float64) if fn is None else per_series(P.astype(np.float64), fn)
    out[name] = report(p)
    print(f"{name:24s} " + "  ".join(f"{k}={v:.5f}" for k, v in out[name].items()), flush=True)

json.dump(out, open(f"{C.ART}/persistence_fold0.json", "w"), indent=1)
