"""Agent 13 shared helpers: screen-store base-model zoo + leakage-safe nested folds.

Nothing here touches src/sbr. All models train on the SCREEN store
(cache/store_screen, 2500 series, 5 folds of 500).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
from sbr.pipeline import Data, load_features, _stack           # noqa: E402
from sbr.metric import ts_auc_flat                              # noqa: E402

ART = "/home/claude/sb/research/artifacts/agent13"
ALL_MODULES = ["m00_core", "m01_seq", "m02_dist", "m03_dyn", "m04_resid", "m05_ctx"]

_D = None
_MATS = None
_NAMES = None


def data():
    global _D, _MATS, _NAMES
    if _D is None:
        _D = Data(screen=True)
        _MATS, _NAMES = load_features(ALL_MODULES, screen=True)
    return _D


def col_index(modules):
    """Column positions (into the concatenated 440-col space) for a module subset."""
    data()
    off, out = 0, []
    for m, A in zip(ALL_MODULES, _MATS):
        if m in modules:
            out += list(range(off, off + A.shape[1]))
        off += A.shape[1]
    return np.array(out)


def X(rows, cols):
    data()
    return _stack(_MATS, _NAMES, rows, cols)


def sample_rows(rows, cap, seed):
    if len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, cap, replace=False))


# ---------------------------------------------------------------- model zoo
LGB_BASE = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
                num_threads=2, verbose=-1, max_bin=127)

MODELS = {
    # --- feature-block LightGBMs (identical algorithm, different information) ---
    "lgb_m00":  dict(algo="lgb", modules=["m00_core"]),
    "lgb_m01":  dict(algo="lgb", modules=["m01_seq"]),
    "lgb_m02":  dict(algo="lgb", modules=["m02_dist"]),
    "lgb_m03":  dict(algo="lgb", modules=["m03_dyn"]),
    "lgb_m04":  dict(algo="lgb", modules=["m04_resid"]),
    "lgb_m05":  dict(algo="lgb", modules=["m05_ctx"]),
    "lgb_all":  dict(algo="lgb", modules=ALL_MODULES),
    # --- same features, genuinely different algorithm / capacity ---
    "lgb_deep": dict(algo="lgb", modules=ALL_MODULES,
                     params=dict(num_leaves=255, min_data_in_leaf=40, learning_rate=0.04,
                                 feature_fraction=0.35, bagging_fraction=0.6, lambda_l2=20.0),
                     n_estimators=200),
    "lgb_stump": dict(algo="lgb", modules=ALL_MODULES,
                      params=dict(num_leaves=10, min_data_in_leaf=500, learning_rate=0.10,
                                  feature_fraction=0.9, lambda_l2=1.0), n_estimators=300),
    "et_all":   dict(algo="et", modules=["m00_core","m03_dyn"], cap=90_000),
    "logit_rank": dict(algo="logit", modules=["m00_core", "m03_dyn", "m04_resid"], cap=150_000),
}


def fit_predict(name, tr_rows, va_rows, seed=0, n_estimators=250, cap=120_000):
    """Train model `name` on tr_rows, return float32 predictions on va_rows."""
    spec = MODELS[name]
    cols = col_index(spec["modules"])
    cap = min(cap, spec.get("cap", cap))
    tr = sample_rows(tr_rows, cap, seed)
    d = data()
    t0 = time.time()
    algo = spec["algo"]
    if algo == "lgb":
        import lightgbm as lgb
        p = dict(LGB_BASE); p.update(spec.get("params", {})); p["seed"] = seed
        nr = spec.get("n_estimators", n_estimators)
        Xtr = X(tr, cols)
        ds = lgb.Dataset(Xtr, label=d.y[tr])
        bst = lgb.train(p, ds, num_boost_round=nr)
        del Xtr, ds
        pred = np.empty(len(va_rows), np.float32)
        for s in range(0, len(va_rows), 200_000):
            pred[s:s + 200_000] = bst.predict(X(va_rows[s:s + 200_000], cols)).astype(np.float32)
    elif algo == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        Xtr = X(tr, cols)
        np.clip(Xtr, -1e9, 1e9, out=Xtr)
        m = ExtraTreesClassifier(n_estimators=60, min_samples_leaf=150, max_features=0.2,
                                 n_jobs=2, random_state=seed, bootstrap=False)
        m.fit(Xtr, d.y[tr]); del Xtr
        pred = np.empty(len(va_rows), np.float32)
        for s in range(0, len(va_rows), 150_000):
            Xv = X(va_rows[s:s + 150_000], cols); np.clip(Xv, -1e9, 1e9, out=Xv)
            pred[s:s + 150_000] = m.predict_proba(Xv)[:, 1].astype(np.float32)
    elif algo == "logit":
        from sklearn.linear_model import LogisticRegression
        Xtr = X(tr, cols).astype(np.float64)
        np.clip(Xtr, -1e9, 1e9, out=Xtr)
        # per-column quantile grid fitted on TRAIN rows only -> rank/gauss transform
        qs = np.linspace(0, 1, 65)
        knots = np.nanquantile(Xtr, qs, axis=0)          # (65, k)
        med = np.nanmedian(Xtr, axis=0)
        def tf(A):
            out = np.empty_like(A)
            for j in range(A.shape[1]):
                kj = knots[:, j]
                good = np.isfinite(kj)
                col = A[:, j].copy()
                bad = ~np.isfinite(col)
                col[bad] = med[j] if np.isfinite(med[j]) else 0.0
                if good.sum() < 3 or np.nanstd(kj) == 0:
                    out[:, j] = 0.0
                    continue
                kk = np.maximum.accumulate(np.nan_to_num(kj, nan=-1e18))
                u = np.interp(col, kk, qs)
                out[:, j] = u - 0.5
            return out
        Ztr = tf(Xtr); del Xtr
        m = LogisticRegression(C=0.05, max_iter=150, solver="lbfgs")
        m.fit(Ztr, d.y[tr]); del Ztr
        pred = np.empty(len(va_rows), np.float32)
        for s in range(0, len(va_rows), 150_000):
            Xv = X(va_rows[s:s + 150_000], cols).astype(np.float64)
            np.clip(Xv, -1e9, 1e9, out=Xv)
            pred[s:s + 150_000] = m.predict_proba(tf(Xv))[:, 1].astype(np.float32)
    else:
        raise ValueError(algo)
    return pred, round(time.time() - t0, 1)


# ---------------------------------------------------------------- evaluation
def tsauc(pred, rows, d=None):
    d = d or data()
    return float(ts_auc_flat(pred, d.y[rows], d.t[rows]))


def series_bounds(rows, d=None):
    """(starts, ends) of contiguous per-series blocks inside `rows` (rows sorted)."""
    d = d or data()
    s = d.sidx[rows]
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    e = np.r_[b[1:], len(s)]
    return b, e
