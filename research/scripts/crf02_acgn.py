#!/usr/bin/env python3
"""CRF-02 -- ACGN, amortized conditional generative null.  TORCH PROCESS.

Implements the frozen protocol in
research/reports/causal_representation_frontier/CRF02_EXECUTION_PREREG.md
(committed at Stage A, 9a3d3c7, before this file existed), which implements
research/reports/causal_representation_frontier/CRF_PROGRAM_PREREG.md sections
0 and 2.  Nothing here may be tuned after a score is seen.

  RT-1237  candidate    learned null + frozen signals + pairwise ranking head
  RT-1238  fixed_null   AR(5) + history residual ECDF through identical
                        statistics and an identical head (isolates the
                        LEARNED vs FIXED conditional null)
  RT-1239  deranged     h_i permuted across series within fold (isolates
                        useful conditioning vs series memorisation)

THIS FILE IMPORTS NO LIGHTGBM.  Integration lives in the separate no-torch
process research/scripts/crf01_integrate.py.  No KMP_DUPLICATE_LIB_OK.

FORBIDDEN AND NOT REACHABLE: tau, post-break age, future observations,
n_online, the final online length, the 500-column bank, RT600 or any
specialist score, `elapsed` or any function of t, any cross-sectional
quantity, fold -1, X_test.reduced.

The generative null NEVER sees a label, an online row, or a validation-fold
series' history.  Tau appears only in the row label y[t] = 1[t >= tau] used by
the ranking head, which PROTOCOL.md permits.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave6_neural_lib import CausalTCN, set_determinism, sha_state_dict
import crf01_nncsr as K1
from sbr.pipeline import Data

# ------------------------------------------------------------ frozen constants
FOLDS = (0, 1, 2, 3, 4)
HWIN = 1024                 # history window, ~4x the measured receptive field 253
HIDDEN = 32
BOTTLENECK = 8
DROPOUT = 0.1
BATCH_SERIES = 32
PRETRAIN_EPOCHS = 10        # compute-budget choice, fixed in advance (prereg 4.1)
HEAD_EPOCHS = 20
LR = 3e-3
WD = 1e-2
CLIP_GRAD = 1.0
SEED = 0
PAIR_SEED = 2026082610
DERANGE_SEED = 2026082611
M_NEG = 8
FEAT_CLIP = 20.0

LEVELS = np.array(
    [0.01, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(17)] + [0.95, 0.99],
    dtype=np.float64)
assert len(LEVELS) == 21 and np.all(np.diff(LEVELS) > 0), LEVELS

SIGNAL_NAMES = ("pit", "neglog", "ad", "surp", "lshift")
FEATURE_NAMES = SIGNAL_NAMES + tuple(f"peak_{s}" for s in SIGNAL_NAMES)
N_FEAT = 10

CLIPS = {"pit": (0.005, 0.995), "neglog": (-5.0, 20.0), "ad": (-5.0, 20.0),
         "surp": (-10.0, 10.0), "lshift": (0.0, 20.0)}

ARMS = {"candidate":  dict(exp_id="RT-1237", null="learned", derange=False, feats="all"),
        "fixed_null": dict(exp_id="RT-1238", null="fixed",   derange=False, feats="all"),
        "deranged":   dict(exp_id="RT-1239", null="learned", derange=True,  feats="all"),
        "shared8_diag": dict(exp_id=None,    null="learned", derange=False, feats="shared8")}

CACHE = os.environ.get("CRF02_CACHE", f"{ROOT}/cache/crf02")


# ================================================================ history state
class SeriesNull:
    """Per-series history-only constants.  Legal at t=0, trivially fold-pure."""

    __slots__ = ("mu", "sd", "hwin", "phi", "sig", "ecdf_resid", "last_hist")

    def __init__(self, hist: np.ndarray):
        h = np.asarray(hist, dtype=np.float64)
        self.mu = float(h.mean()) if len(h) else 0.0
        sd = float(h.std(ddof=1)) if len(h) > 1 else 1.0
        self.sd = max(sd, 1e-9) if np.isfinite(sd) else 1.0
        z = (h - self.mu) / self.sd
        self.hwin = z[-HWIN:].astype(np.float32)
        self.last_hist = float(z[-1]) if len(z) else 0.0
        # fixed-null path (C1): AR(5) + history residual ECDF, CRF-01 channels 2-3
        self.phi = K1._yule_walker(z, K1.AR_ORDER)
        e = K1._ar_resid_hist(z, self.phi)
        s = float(e.std(ddof=1)) if len(e) > 1 else 1.0
        self.sig = max(s, 1e-9) if np.isfinite(s) else 1.0
        self.ecdf_resid = K1._ecdf_knots(e / self.sig)


def z_online(null: SeriesNull, online: np.ndarray) -> np.ndarray:
    return ((np.asarray(online, dtype=np.float64) - null.mu) / null.sd).astype(np.float32)


def shift_input(z_on: np.ndarray, null: SeriesNull) -> np.ndarray:
    """Position t carries z_{t-1}; position 0 carries the LAST HISTORY POINT.

    This is what makes the head at t a function of the STRICT past: a causal TCN
    at t would otherwise see z_t, the very value the null is pricing.
    """
    out = np.empty_like(z_on)
    out[0] = null.last_hist
    if len(z_on) > 1:
        out[1:] = z_on[:-1]
    return out


# ==================================================================== the model
class ACGN:
    """Shared body + 8-d history bottleneck + 21 monotone quantile knots."""

    @staticmethod
    def build(seed, device):
        import torch
        from torch import nn
        torch.manual_seed(seed)
        body = CausalTCN.build(HIDDEN, DROPOUT, seed, device, n_in=1)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = body
                self.bottleneck = nn.Linear(HIDDEN, BOTTLENECK)
                self.head = nn.Conv1d(HIDDEN + BOTTLENECK, len(LEVELS), 1)
                self.register_buffer("levels", torch.tensor(LEVELS, dtype=torch.float32))

            def encode(self, zh, mask):                    # (B,1,T),(B,T) -> (B,32)
                f = self.body_features(zh)                 # (B,32,T)
                m = mask.unsqueeze(1)
                return (f * m).sum(-1) / m.sum(-1).clamp(min=1.0)

            def body_features(self, z):                    # (B,1,T) -> (B,32,T)
                x = z
                for b in self.body.blocks:
                    x = b(x)
                return x

            def h_of(self, p):                             # (B,32) -> (B,8)
                return self.bottleneck(p)

            def knots(self, z_shift, h):                   # (B,1,T),(B,8) -> (B,21,T)
                f = self.body_features(z_shift)
                hh = h.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                raw = self.head(torch.cat([f, hh], dim=1))
                base = raw[:, :1]
                inc = nn.functional.softplus(raw[:, 1:])
                return torch.cat([base, base + torch.cumsum(inc, dim=1)], dim=1)

            def forward(self, z_hist, hmask, z_shift):
                p = self.encode(z_hist, hmask)
                return self.knots(z_shift, self.h_of(p)), p

        return Net().to(device)


def pinball(q, target, mask, levels):
    """mean over levels and VALID positions.  q (B,21,T), target (B,T), mask (B,T)."""
    import torch
    d = target.unsqueeze(1) - q
    lv = levels.view(1, -1, 1)
    loss = torch.maximum(lv * d, (lv - 1.0) * d)
    m = mask.unsqueeze(1)
    return (loss * m).sum() / (m.sum() * q.shape[1])


def build_ranking_head(seed, device, n_in=N_FEAT):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(n_in, 32), nn.GELU(), nn.Linear(32, 1)).to(device)


# ================================================== signals from predicted knots
def signals_from_knots(Q: np.ndarray, z: np.ndarray, neglog_base: float,
                       body_feat: np.ndarray | None, p_i: np.ndarray | None
                       ) -> np.ndarray:
    """(n,21) knots + (n,) realised z -> (n,10) frozen features.  CRF02 prereg 5.

    Strictly causal: every accumulator is a running mean or running max over
    s <= t, and Q[t] already depends only on the strict past and the history.
    """
    n = len(z)
    lv = LEVELS
    F = np.zeros((n, N_FEAT), dtype=np.float64)
    if n == 0:
        return F.astype(np.float32)
    Q = np.maximum.accumulate(np.asarray(Q, dtype=np.float64), axis=1)  # monotone

    # 1. pit -- linear interpolation across the knots, clamped outside
    idx = (Q < np.asarray(z, dtype=np.float64)[:, None]).sum(axis=1)
    lo = np.clip(idx - 1, 0, len(lv) - 2)
    hi = lo + 1
    ql, qh = Q[np.arange(n), lo], Q[np.arange(n), hi]
    w = np.where(qh - ql > 1e-12, (z - ql) / np.maximum(qh - ql, 1e-12), 0.0)
    u = lv[lo] + np.clip(w, 0.0, 1.0) * (lv[hi] - lv[lo])
    u = np.where(z <= Q[:, 0], lv[0], u)
    u = np.where(z >= Q[:, -1], lv[-1], u)
    u = np.clip(u, CLIPS["pit"][0], CLIPS["pit"][1])

    # 2. neglog -- -log of the piecewise-uniform density in the containing bin
    dens = (lv[hi] - lv[lo]) / np.maximum(qh - ql, 1e-6)
    neglog = np.clip(-np.log(np.maximum(dens, 1e-8)), *CLIPS["neglog"])

    # 3. ad -- running Anderson-Darling-weighted uniformity discrepancy, mean 0
    k = np.arange(1, n + 1, dtype=np.float64)
    ad_term = -np.log(u) - np.log(1.0 - u)
    ad = np.clip(np.cumsum(ad_term) / k - 2.0, *CLIPS["ad"])

    # 4. surp -- running mean neglog minus the model's own history baseline
    surp = np.clip(np.cumsum(neglog) / k - neglog_base, *CLIPS["surp"])

    # 5. lshift -- running-mean latent state vs the history-pooled value
    if body_feat is None:
        lshift = np.zeros(n)
    else:
        run = np.cumsum(body_feat, axis=0) / k[:, None]
        lshift = np.clip(
            np.sqrt(((run - p_i[None, :]) ** 2).sum(1) / HIDDEN), *CLIPS["lshift"])

    F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4] = u, neglog, ad, surp, lshift
    F[:, 5] = np.maximum.accumulate(np.abs(u - 0.5))
    for j, v in enumerate((neglog, ad, surp, lshift), start=6):
        F[:, j] = np.maximum.accumulate(v)
    F[~np.isfinite(F)] = 0.0
    return F.astype(np.float32)


def fixed_null_knots(null: SeriesNull, z_on: np.ndarray) -> np.ndarray:
    """C1: AR(5) mean + history residual ECDF quantiles.  CRF-01 channels 2-3.

    Same 21 levels, same downstream statistics, same ranking head -- only the
    conditional null differs.  Strictly causal: the AR mean at t uses z[<t] and
    the historical tail, never z_t.
    """
    ux, up = null.ecdf_resid
    qres = np.interp(LEVELS, up, ux) if len(ux) > 1 else np.zeros(len(LEVELS))
    zs = shift_input(z_on, null).astype(np.float64)
    p = len(null.phi)
    warm = null.hwin[-p:].astype(np.float64) if len(null.hwin) >= p else np.zeros(p)
    pad = np.concatenate([warm, np.asarray(z_on, dtype=np.float64)])
    m = len(pad)
    X = np.column_stack([pad[p - k - 1: m - k - 1] for k in range(p)])
    mean = X @ null.phi                                   # strict-past AR(5) mean
    return (mean[:, None] + null.sig * qres[None, :]).astype(np.float64), zs


# ============================================================ per-series caches
def build_nulls(d, log):
    """Per-series history constants, built once and shared by every arm."""
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    nulls = []
    for i in range(d.st.n_series):
        nulls.append(SeriesNull(d.st.hist(i)))
        if (i + 1) % 2000 == 0:
            log(f"    nulls {i + 1}/{d.st.n_series}  {time.time() - t0:.0f}s")
    log(f"  per-series history nulls built in {time.time() - t0:.0f}s")
    return nulls


def _pack_hist(nulls, ids):
    T = max(len(nulls[s].hwin) for s in ids)
    Z = np.zeros((len(ids), 1, T), dtype=np.float32)
    M = np.zeros((len(ids), T), dtype=np.float32)
    for k, s in enumerate(ids):
        w = nulls[s].hwin
        Z[k, 0, :len(w)] = w
        M[k, :len(w)] = 1.0
    return Z, M


# ================================================================= pretraining
def pretrain_null(fold, d, nulls, log, seed=SEED):
    """One outer fold's generative null.  TRAINING-FOLD HISTORIES ONLY.

    PURITY: the fit set is exactly FOLDS \\ {fold} at the SERIES level, asserted
    here.  No online row, no label, and no validation-fold history enters this
    function.  CRF_PROGRAM_PREREG 2.6: it may NOT be fitted once globally and
    reused across outer folds -- that is the Wave-7 nested-teacher pattern.
    """
    import torch
    from torch import nn
    device = set_determinism(seed)
    tr = np.flatnonzero(np.isin(d.series_fold, [f for f in FOLDS if f != fold]))
    va = np.flatnonzero(d.series_fold == fold)
    lb = np.flatnonzero(d.series_fold == -1)
    assert len(np.intersect1d(tr, va)) == 0, "PURITY VIOLATION: null fit set n val"
    assert len(np.intersect1d(tr, lb)) == 0, "PURITY VIOLATION: null fit set n lockbox"
    assert set(d.series_fold[tr]) == {f for f in FOLDS if f != fold}, \
        "PURITY VIOLATION: null fit set is not exactly FOLDS \\ {fold}"

    torch.manual_seed(seed * 1000 + fold)
    net = ACGN.build(seed * 1000 + fold, device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    hlen = np.array([len(nulls[s].hwin) for s in range(d.st.n_series)], dtype=np.int64)
    n_steps = PRETRAIN_EPOCHS * int(np.ceil(len(tr) / BATCH_SERIES))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    rng = np.random.default_rng(seed * 1000 + fold)
    t0, hist = time.time(), []
    net.train()
    for ep in range(PRETRAIN_EPOCHS):
        tot, nb = 0.0, 0
        for ids in K1._batches(tr, hlen, rng, BATCH_SERIES):
            Z, M = _pack_hist(nulls, ids)
            zt = torch.from_numpy(Z)
            mt = torch.from_numpy(M)
            # the null is trained to price its OWN history one step ahead:
            # target z_t, input shifted right by one, zero-padded at position 0
            tgt = zt[:, 0, :]
            shift = torch.zeros_like(zt)
            shift[:, :, 1:] = zt[:, :, :-1]
            msk = mt.clone(); msk[:, 0] = 0.0
            opt.zero_grad(set_to_none=True)
            p = net.encode(zt, mt)
            q = net.knots(shift, net.h_of(p))
            loss = pinball(q, tgt, msk, net.levels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist.append(tot / max(nb, 1))
        log(f"    null fold {fold} epoch {ep+1}/{PRETRAIN_EPOCHS}  pinball "
            f"{hist[-1]:.6f}  ({time.time()-t0:.0f}s)")
    net.eval()
    for prm in net.parameters():          # FROZEN before any signal is computed
        prm.requires_grad_(False)
    return net, {"fit_series": int(len(tr)), "val_series": int(len(va)),
                 "pinball_history": hist, "state_sha256": sha_state_dict(net.state_dict()),
                 "n_parameters": int(sum(p.numel() for p in net.parameters())),
                 "pretrain_runtime_s": round(time.time() - t0, 1)}


# =========================================================== signal generation
def generate_signals(net, fold, d, nulls, log, derange=False, fixed=False, seed=SEED):
    """(n_rows, 10) frozen features for every dev series.  Generative model FROZEN.

    `h_i` for a VALIDATION series is a forward pass through the training-fold-
    fitted encoder over that series' own history -- never a fit (prereg 3).
    """
    import torch
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    F = np.zeros((len(d.y), N_FEAT), dtype=np.float32)
    dev = np.flatnonzero(d.series_fold >= 0)
    t0 = time.time()

    # ---- pooled history states p_i and h_i, per series, forward pass only
    P = np.zeros((d.st.n_series, HIDDEN), dtype=np.float32)
    if not fixed:
        with torch.no_grad():
            for i in range(0, len(dev), BATCH_SERIES):
                ids = dev[i:i + BATCH_SERIES]
                Z, M = _pack_hist(nulls, ids)
                P[ids] = net.encode(torch.from_numpy(Z), torch.from_numpy(M)).numpy()
        log(f"  pooled history states in {time.time()-t0:.0f}s")

    # ---- C2 derangement: permute h_i ACROSS SERIES WITHIN FOLD GROUP
    src = np.arange(d.st.n_series)
    if derange:
        for f in FOLDS:
            g = np.flatnonzero(d.series_fold == f)
            r = np.random.default_rng(DERANGE_SEED + f)
            perm = r.permutation(len(g))
            fixed_pts = np.flatnonzero(perm == np.arange(len(g)))
            for j in fixed_pts:                       # make it a true derangement
                k = (j + 1) % len(g)
                perm[j], perm[k] = perm[k], perm[j]
            src[g] = g[perm]

    # ---- baseline neglog on each series' OWN history window, history only
    with torch.no_grad():
        for i in range(0, len(dev), BATCH_SERIES):
            ids = dev[i:i + BATCH_SERIES]
            if not fixed:
                Z, M = _pack_hist(nulls, ids)
                zt = torch.from_numpy(Z)
                shift = torch.zeros_like(zt); shift[:, :, 1:] = zt[:, :, :-1]
                h = net.h_of(torch.from_numpy(P[src[ids]]))
                Qh = net.knots(shift, h).numpy()
            for k, s in enumerate(ids):
                nl = nulls[s]
                o = d.st.online(int(s))
                zo = z_online(nl, o)
                if fixed:
                    Qw, _ = fixed_null_knots(nl, nl.hwin.astype(np.float64))
                    base = float(np.mean(signals_from_knots(
                        Qw, nl.hwin.astype(np.float64), 0.0, None, None)[:, 1]))
                    Qo, _ = fixed_null_knots(nl, zo)
                    F[int(off[s]):int(off[s]) + len(o)] = signals_from_knots(
                        Qo, zo.astype(np.float64), base, None, None)
                else:
                    w = len(nl.hwin)
                    base = float(np.mean(signals_from_knots(
                        Qh[k, :, :w].T, nl.hwin.astype(np.float64), 0.0, None, None)[1:, 1]))
                    zs = shift_input(zo, nl)
                    zst = torch.from_numpy(zs[None, None, :])
                    hk = net.h_of(torch.from_numpy(P[src[s]][None, :]))
                    Qo = net.knots(zst, hk).numpy()[0].T
                    bf = net.body_features(zst).numpy()[0].T
                    F[int(off[s]):int(off[s]) + len(o)] = signals_from_knots(
                        Qo, zo.astype(np.float64), base, bf, P[s])
            if (i // BATCH_SERIES) % 40 == 0:
                log(f"    signals {i}/{len(dev)}  {time.time()-t0:.0f}s")
    log(f"  signals generated in {time.time()-t0:.0f}s")
    return F, {"derange": bool(derange), "fixed": bool(fixed),
               "signal_runtime_s": round(time.time() - t0, 1)}


# ============================================================== ranking head
class FeatureStandardiser:
    """Median/IQR from TRAINING-FOLD ROWS ONLY, frozen, then applied everywhere.

    This is the one genuinely fitted global object in CRF-02 (prereg 6), so
    CRF_PROGRAM_PREREG 0.5's training-fold-only requirement is NOT vacuous here
    and is asserted rather than assumed.
    """

    __slots__ = ("med", "iqr", "fit_rows")

    def fit(self, F, rows):
        X = np.asarray(F[rows], dtype=np.float64)
        q = np.nanquantile(X, [0.25, 0.5, 0.75], axis=0)
        self.med = q[1].astype(np.float32)
        iqr = (q[2] - q[0]).astype(np.float32)
        iqr[~np.isfinite(iqr) | (iqr <= 1e-12)] = 1.0
        self.med[~np.isfinite(self.med)] = 0.0
        self.iqr = iqr
        self.fit_rows = int(len(rows))
        return self

    def transform(self, F, rows=None):
        X = np.asarray(F if rows is None else F[rows], dtype=np.float32)
        Z = (X - self.med) / self.iqr
        Z[~np.isfinite(Z)] = 0.0
        return np.clip(Z, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)


def _pack_feat(Fz, y, off, n_on, ids, cols):
    T = int(n_on[ids].max())
    X = np.zeros((len(ids), len(cols), T), dtype=np.float32)
    Y = np.zeros((len(ids), T), dtype=np.float32)
    M = np.zeros((len(ids), T), dtype=np.float32)
    for k, s in enumerate(ids):
        a, n = int(off[s]), int(n_on[s])
        X[k, :, :n] = Fz[a:a + n][:, cols].T
        Y[k, :n] = y[a:a + n]
        M[k, :n] = 1.0
    return X, Y, M


def train_ranking_head(arm, fold, d, F, log, gen_params=None, seed=SEED):
    """Same-`t` pairwise logistic head over the frozen signals.  CRF-01's objective.

    The generative model is already frozen; `gen_params` is checked after a
    backward pass to prove the label never rewrites the null (prereg gate P6).
    """
    import torch
    from torch import nn
    cfg = ARMS[arm]
    cols = list(range(N_FEAT)) if cfg["feats"] == "all" else [0, 1, 2, 3, 5, 6, 7, 8]
    device = set_determinism(seed)
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    y = d.y.astype(np.float32)

    tr = np.flatnonzero(np.isin(d.series_fold, [f for f in FOLDS if f != fold]))
    va = np.flatnonzero(d.series_fold == fold)
    assert len(np.intersect1d(tr, va)) == 0, "PURITY VIOLATION: head train n val"
    tr_rows = np.concatenate([np.arange(int(off[s]), int(off[s]) + int(n_on[s]))
                              for s in tr])
    std = FeatureStandardiser().fit(F, tr_rows)
    Fz = std.transform(F)

    torch.manual_seed(seed * 1000 + fold)
    net = build_ranking_head(seed * 1000 + fold, device, n_in=len(cols))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    n_steps = HEAD_EPOCHS * int(np.ceil(len(tr) / BATCH_SERIES))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    batch_rng = np.random.default_rng(seed * 1000 + fold)
    pair_rng = np.random.default_rng(PAIR_SEED + fold)
    t0, hist, checked = time.time(), [], False
    net.train()
    for ep in range(HEAD_EPOCHS):
        tot, nb, pairs = 0.0, 0, 0
        for ids in K1._batches(tr, n_on, batch_rng, BATCH_SERIES):
            X, Y, M = _pack_feat(Fz, y, off, n_on, ids, cols)
            xb = torch.from_numpy(X).permute(0, 2, 1)      # (B,T,C)
            opt.zero_grad(set_to_none=True)
            logits = net(xb).squeeze(-1)                    # (B,T)
            s = K1.sample_pairs(Y, M, pair_rng)
            if s is None:
                loss = logits.sum() * 0.0
            else:
                pb, pt, nbi, nt, _ = s
                sp = logits[torch.from_numpy(pb), torch.from_numpy(pt)]
                sn = logits[torch.from_numpy(nbi), torch.from_numpy(nt)]
                loss = nn.functional.softplus(-(sp - sn)).mean()
                pairs += len(pb)
            loss.backward()
            if not checked and gen_params is not None:
                bad = [n for n, p in gen_params
                       if p.grad is not None and float(p.grad.abs().sum()) > 0.0]
                assert not bad, f"ISOLATION VIOLATION: label reached the null: {bad[:5]}"
                checked = True
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist.append(tot / max(nb, 1))
        log(f"    {arm} head fold {fold} epoch {ep+1}/{HEAD_EPOCHS}  loss "
            f"{hist[-1]:.6f}  pairs {pairs}  ({time.time()-t0:.0f}s)")

    net.eval()
    rows, vals = [], []
    with torch.no_grad():
        for i in range(0, len(va), BATCH_SERIES):
            ids = va[i:i + BATCH_SERIES]
            X, _, _ = _pack_feat(Fz, y, off, n_on, ids, cols)
            p = net(torch.from_numpy(X).permute(0, 2, 1)).squeeze(-1).numpy()
            for k, s in enumerate(ids):
                a, n = int(off[s]), int(n_on[s])
                rows.append(np.arange(a, a + n, dtype=np.int64))
                vals.append(p[k, :n].astype(np.float32))
    return {
        "arm": arm, "exp_id": cfg["exp_id"], "fold": int(fold),
        "rows": np.concatenate(rows), "scores": np.concatenate(vals),
        "feature_cols": [FEATURE_NAMES[c] for c in cols],
        "standardiser": {"median": std.med.tolist(), "iqr": std.iqr.tolist(),
                         "fit_rows": std.fit_rows},
        "loss_history": hist, "isolation_checked": bool(checked),
        "head_state_sha256": sha_state_dict(net.state_dict()),
        "runtime_s": round(time.time() - t0, 1),
    }


def emit(res, log):
    outdir = f"{CACHE}/scores"
    os.makedirs(outdir, exist_ok=True)
    tag = f"{res['exp_id'] or res['arm']}_fold{res['fold']}"
    np.save(f"{outdir}/{tag}.rows.npy", res["rows"])
    np.save(f"{outdir}/{tag}.scores.npy", res["scores"])
    json.dump({k: v for k, v in res.items() if k not in ("rows", "scores")} |
              {"n_rows": int(len(res["rows"]))}, open(f"{outdir}/{tag}.json", "w"), indent=1)
    log(f"  emitted {outdir}/{tag}.*  ({len(res['rows'])} rows)")


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arms", default="candidate,fixed_null,deranged,shared8_diag")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    logf = open(f"{ROOT}/logs/crf02.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    log(f"=== CRF-02 ACGN fold {a.fold} === {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(json.dumps(K1.env_report_notorch()))
    d = Data()
    nulls = build_nulls(d, log)
    meta = {"fold": a.fold, "levels": LEVELS.tolist(), "hwin": HWIN,
            "pretrain_epochs": PRETRAIN_EPOCHS, "head_epochs": HEAD_EPOCHS,
            "features": list(FEATURE_NAMES), "arms": {}}

    log(f"--- pretraining the fold-{a.fold} generative null (training-fold histories only) ---")
    net, pm = pretrain_null(a.fold, d, nulls, log, seed=a.seed)
    meta["null"] = pm
    log(f"  null frozen, state {pm['state_sha256'][:12]}, {pm['pretrain_runtime_s']}s")
    gen_params = [(n, p) for n, p in net.named_parameters()]

    sig_cache = {}
    for arm in a.arms.split(","):
        cfg = ARMS[arm]
        key = (cfg["null"], cfg["derange"])
        if key not in sig_cache:
            log(f"--- signals: null={cfg['null']} derange={cfg['derange']} ---")
            sig_cache[key] = generate_signals(
                net, a.fold, d, nulls, log,
                derange=cfg["derange"], fixed=(cfg["null"] == "fixed"), seed=a.seed)
        F, sm = sig_cache[key]
        log(f"--- {arm} ({cfg['exp_id']}) ranking head, fold {a.fold} ---")
        res = train_ranking_head(arm, a.fold, d, F, log, gen_params=gen_params, seed=a.seed)
        emit(res, log)
        meta["arms"][arm] = {k: v for k, v in res.items()
                             if k not in ("rows", "scores")} | sm
        json.dump(meta, open(f"{CACHE}/fold{a.fold}_meta.json", "w"), indent=1)
    log("CRF-02 fold complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
