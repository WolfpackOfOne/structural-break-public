#!/usr/bin/env python3
"""CRF-01 -- NNCSR, null-normalized causal sequence ranker.  TORCH PROCESS.

Implements the frozen protocol in
research/reports/causal_representation_frontier/CRF01_EXECUTION_PREREG.md
(committed at Stage A, before this file existed), which in turn implements
research/reports/causal_representation_frontier/CRF_PROGRAM_PREREG.md sections
0 and 1.  Nothing here may be tuned after a score is seen; a change to any
frozen quantity creates a new RT id and a new execution preregistration.

  RT-1234  candidate     8 null-normalised channels + RT-970 TCN shell,
                         same-t pairwise logistic, m_neg = 8
  RT-1235  bce_control   byte-identical arm, masked rowwise BCE (isolates
                         the objective)
  RT-1236  shuffle_ctrl  identical arm and objective, online channels
                         temporally shuffled within each series (conditional)

THIS FILE IMPORTS NO LIGHTGBM.  torch and lightgbm segfault sharing a process
on macOS/arm64 (Wave-6 finding); ensemble integration lives in the separate
no-torch process research/scripts/crf01_integrate.py.  KMP_DUPLICATE_LIB_OK
and every equivalent hack are forbidden.

FORBIDDEN AND NOT REACHABLE FROM THIS FILE
    tau, post-break age, future observations, n_online, the final online
    length, boundary-conditioned availability, oracle features, the 500-column
    bank, RT600 or any specialist / seed-clone score, any first- or
    second-sweep candidate score, `elapsed` or any monotone function of t, any
    cross-sectional quantity, fold -1, X_test.reduced, folds_final10k.

Tau appears in exactly one place and it is legal: building the row label
y[t] = 1[t >= tau], which is the supervised target PROTOCOL.md permits.  It
never enters a channel, a normalisation constant, a mask, a loss weight or a
hidden state.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave6_neural_lib import CausalTCN, env_report, set_determinism, sha_state_dict

from sbr.pipeline import Data

# --------------------------------------------------------------- frozen constants
N_CHANNELS = 8
CHANNEL_NAMES = ("pit", "inn", "inn_pit", "abs_inn_pit",
                 "vol_norm", "surp", "exceed", "lag1_pit")
AR_ORDER = 5
N_KNOTS = 256
EWMA_HALFLIFE = 32.0
VOL_FLOOR_FRAC = 0.25
EXCEED_SLOPE = 4.0
CLIP_PIT = 4.0
CLIP_VOL = 8.0
CLIP_SURP = 12.0
CLIP_LAG1 = 16.0

FOLDS = (0, 1, 2, 3, 4)
HIDDEN = 32
DROPOUT = 0.1
BATCH_SERIES = 32
EPOCHS = 20
LR = 3e-3
WD = 1e-2
CLIP_GRAD = 1.0
M_NEG = 8
SEED = 0
PAIR_SEED = 2026082600
SHUFFLE_SEED = 2026082601
MAX_FOLD_SECONDS = 3 * 3600

ARMS = {
    "candidate":   dict(exp_id="RT-1234", objective="pairwise_t", shuffle=False),
    "bce_control": dict(exp_id="RT-1235", objective="bce",        shuffle=False),
    "shuffle_ctrl": dict(exp_id="RT-1236", objective="pairwise_t", shuffle=True),
}

CACHE = os.environ.get("CRF01_CACHE", f"{ROOT}/cache/crf01")

FORBIDDEN_TOKENS = ("tau", "cut", "boundary", "break_at", "changepoint_true",
                    "has_break", "n_online", "n_hist", "final_row",
                    "eligible", "availability", "pad", "elapsed", "rt600",
                    "age", "rank")


def assert_no_forbidden_channel_names(names) -> None:
    bad = [n for n in names if any(tok in n.lower() for tok in FORBIDDEN_TOKENS)]
    assert not bad, f"forbidden name(s) reachable by CRF-01: {bad}"


# =============================================================== channel builder
def _ecdf_knots(v: np.ndarray, n_knots: int = N_KNOTS):
    """256-knot ECDF with (r - 0.5)/n plotting positions.  CRF01 prereg section 4.

    Duplicate abscissae collapse to their MAXIMUM probability so the knot
    sequence is strictly increasing and np.interp is well defined.
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return np.zeros(1), np.full(1, 0.5)
    s = np.sort(v)
    idx = (np.arange(n) if n <= n_knots
           else np.rint(np.linspace(0.0, n - 1.0, n_knots)).astype(np.int64))
    x = s[idx]
    p = (idx.astype(np.float64) + 0.5) / n
    ux, inv = np.unique(x, return_inverse=True)
    up = np.zeros(len(ux), dtype=np.float64)
    np.maximum.at(up, inv, p)
    return ux, up


def _ecdf_eval(knots, x) -> np.ndarray:
    ux, up = knots
    x = np.asarray(x, dtype=np.float64)
    if len(ux) == 1:
        return np.full(x.shape, up[0], dtype=np.float64)
    return np.interp(x, ux, up)


def _ndtri(p: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri
    return ndtri(np.asarray(p, dtype=np.float64))


def _yule_walker(z: np.ndarray, p: int = AR_ORDER) -> np.ndarray:
    """Yule-Walker AR(p) on the history only.  Biased autocovariances, ridge 1e-10*r0.

    Guard `n < 10p + 10` matches sbr.transforms._fit_ar, which is OLS; this is
    deliberately Yule-Walker as CRF_PROGRAM_PREREG section 1.4 specifies.
    """
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if p <= 0 or n < 10 * p + 10:
        return np.zeros(p)
    r = np.array([float(np.dot(z[:n - k], z[k:])) / n for k in range(p + 1)])
    if not np.isfinite(r).all() or r[0] <= 0.0:
        return np.zeros(p)
    i = np.arange(p)
    T = r[np.abs(i[:, None] - i[None, :])] + np.eye(p) * (1e-10 * r[0])
    try:
        phi = np.linalg.solve(T, r[1:])
    except np.linalg.LinAlgError:
        return np.zeros(p)
    return phi if np.isfinite(phi).all() else np.zeros(p)


def _ar_resid_hist(z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    n = len(z)
    if p == 0 or n <= p:
        return z.copy()
    X = np.column_stack([z[p - k - 1: n - k - 1] for k in range(p)])
    return z[p:] - X @ phi


def _ar_filter_causal(z_on: np.ndarray, phi: np.ndarray, warm: np.ndarray) -> np.ndarray:
    """Online AR residuals with the HISTORICAL TAIL as the lag source for t < p.

    Row t uses z_on[:t] and `warm` only -- the sbr.transforms.ar_filter_causal
    convention, which is what makes rows t < 5 legal and prefix-invariant.
    """
    p = len(phi)
    if p == 0:
        return z_on.copy()
    pad = np.concatenate([warm[-p:], z_on]) if len(warm) >= p else \
          np.concatenate([np.zeros(p), z_on])
    m = len(pad)
    X = np.column_stack([pad[p - k - 1: m - k - 1] for k in range(p)])
    return z_on - X @ phi


class HistoryNull:
    """Every per-series constant CRF-01 needs, fitted on that series' history ONLY.

    Legal at inference: the history is complete at t = 0.  Trivially fold-pure:
    it never touches another series, another fold, a label, or an online row.
    """

    __slots__ = ("mu", "sd", "phi", "sig", "ecdf_x", "ecdf_inn", "ecdf_absinn",
                 "mad", "q90", "n_hist")

    def __init__(self, hist: np.ndarray):
        h = np.asarray(hist, dtype=np.float64)
        self.n_hist = len(h)
        self.mu = float(h.mean()) if len(h) else 0.0
        sd = float(h.std(ddof=1)) if len(h) > 1 else 1.0
        self.sd = max(sd, 1e-9) if np.isfinite(sd) else 1.0
        z = (h - self.mu) / self.sd
        self.phi = _yule_walker(z, AR_ORDER)
        e = _ar_resid_hist(z, self.phi)
        s = float(e.std(ddof=1)) if len(e) > 1 else 1.0
        self.sig = max(s, 1e-9) if np.isfinite(s) else 1.0
        inn_h = e / self.sig
        self.ecdf_x = _ecdf_knots(h)
        self.ecdf_inn = _ecdf_knots(inn_h)
        self.ecdf_absinn = _ecdf_knots(np.abs(inn_h))
        med = float(np.median(inn_h)) if len(inn_h) else 0.0
        mad = 1.4826 * float(np.median(np.abs(inn_h - med))) if len(inn_h) else 1.0
        self.mad = max(mad, 1e-9) if np.isfinite(mad) else 1.0
        self.q90 = float(np.quantile(np.abs(inn_h), 0.90)) if len(inn_h) else 1.0
        if not np.isfinite(self.q90):
            self.q90 = 1.0

    def warm_tail(self, hist: np.ndarray) -> np.ndarray:
        """The standardised tail of the history, the AR lag source for t < p."""
        h = np.asarray(hist[-AR_ORDER:], dtype=np.float64)
        return (h - self.mu) / self.sd


def _lagged_ewma(absinn: np.ndarray, seed: float, half_life: float) -> np.ndarray:
    """L[t] = E[t-1] with E[-1] = seed and E[t] = E[t-1] + a(|inn_t| - E[t-1]).

    STRICTLY LAGGED: inn_t never influences its own denominator.  Explicit
    left-to-right loop, so bitwise prefix invariance is structural.
    """
    a = 1.0 - 0.5 ** (1.0 / half_life)
    n = len(absinn)
    out = np.empty(n, dtype=np.float64)
    acc = float(seed)
    for i in range(n):
        out[i] = acc
        acc = acc + a * (absinn[i] - acc)
    return out


def causal_channels(hist: np.ndarray, online: np.ndarray, null: HistoryNull | None = None
                    ) -> np.ndarray:
    """The eight frozen null-normalised channels.  (n_online, 8) float32.

    Row t uses `hist` and `online[:t+1]` and NOTHING ELSE.  No global scale is
    fitted anywhere: every clip bound is a constant written in the execution
    preregistration and every per-series constant is a function of `hist` alone.
    """
    h = np.asarray(hist, dtype=np.float64)
    x = np.asarray(online, dtype=np.float64)
    n = len(x)
    if null is None:
        null = HistoryNull(h)
    C = np.empty((n, N_CHANNELS), dtype=np.float64)
    if n == 0:
        return C.astype(np.float32)

    # 1. pit -- history ECDF -> normal score
    pit = np.clip(_ndtri(_ecdf_eval(null.ecdf_x, x)), -CLIP_PIT, CLIP_PIT)

    # 2. inn -- AR(5) one-step innovation, sigma_H-normalised, history-tail lags
    z = (x - null.mu) / null.sd
    warm = null.warm_tail(h)
    inn = _ar_filter_causal(z, null.phi, warm) / null.sig
    absinn = np.abs(inn)

    # 3/4. residual ECDF -> normal score, and its magnitude
    inn_pit = np.clip(_ndtri(_ecdf_eval(null.ecdf_inn, inn)), -CLIP_PIT, CLIP_PIT)

    # 5. vol_norm -- strictly lagged half-life-32 EWMA, floored at 0.25*mad_H
    lag_ewma = _lagged_ewma(absinn, null.mad, EWMA_HALFLIFE)
    den = np.maximum(lag_ewma, VOL_FLOOR_FRAC * null.mad)
    vol = np.clip(inn / den, -CLIP_VOL, CLIP_VOL)

    # 6. surp -- -log tail probability of |inn| under the history's own |inn| ECDF
    g = _ecdf_eval(null.ecdf_absinn, absinn)
    surp = np.clip(-np.log(1.0 - g + 1.0 / (null.n_hist + 1.0)), 0.0, CLIP_SURP)

    # 7. exceed -- soft exceedance against the history's own 90th percentile
    ex = 1.0 / (1.0 + np.exp(-np.clip(
        EXCEED_SLOPE * (absinn - null.q90) / null.mad, -60.0, 60.0)))

    # 8. lag1_pit -- pit_t * pit_{t-1}, pit_{-1} = 0
    lag1 = np.empty(n, dtype=np.float64)
    lag1[0] = 0.0
    if n > 1:
        lag1[1:] = pit[1:] * pit[:-1]
    lag1 = np.clip(lag1, -CLIP_LAG1, CLIP_LAG1)

    C[:, 0] = pit
    C[:, 1] = inn
    C[:, 2] = inn_pit
    C[:, 3] = np.abs(inn_pit)
    C[:, 4] = vol
    C[:, 5] = surp
    C[:, 6] = ex
    C[:, 7] = lag1
    C[~np.isfinite(C)] = 0.0
    return C.astype(np.float32)


# ------------------------------------------------------------------ channel cache
def build_channels(d, log):
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/channels.npy"
    meta = f"{CACHE}/channels.json"
    n_rows = len(d.y)
    if os.path.exists(meta) and os.path.exists(path):
        m = json.load(open(meta))
        if m.get("n_rows") == n_rows and m.get("channels") == list(CHANNEL_NAMES):
            log(f"  channel cache hit: {path}")
            return np.load(path, mmap_mode="r")
    log(f"  building {n_rows} x {N_CHANNELS} null-normalised causal channels ...")
    out = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float32,
                                    shape=(n_rows, N_CHANNELS))
    t0 = time.time()
    off = d.st.orow_off
    for i in range(d.st.n_series):
        h, o, _ = d.st.series(i)
        a = int(off[i])
        out[a:a + len(o)] = causal_channels(h, o)
        if (i + 1) % 1000 == 0:
            log(f"    {i + 1}/{d.st.n_series}  {time.time() - t0:.0f}s")
    out.flush(); del out
    os.replace(path + ".tmp", path)
    json.dump({"n_rows": n_rows, "channels": list(CHANNEL_NAMES),
               "built_s": round(time.time() - t0, 1)}, open(meta, "w"))
    log(f"  channels built in {time.time() - t0:.0f}s")
    return np.load(path, mmap_mode="r")


# ------------------------------------------------------------------- batching
def _batches(series_ids, lens, rng, batch=BATCH_SERIES):
    """Length-bucketed batches, chunk order shuffled.  wave6_n2_tcn._batches verbatim.

    Legal and not a covert length channel: the network has no BatchNorm and no
    time-axis normalisation, so a series' output cannot depend on its
    neighbours -- asserted by the batch-composition gate, not assumed.
    """
    order = np.argsort(lens[series_ids], kind="stable")
    ordered = np.asarray(series_ids)[order]
    chunks = [ordered[i:i + batch] for i in range(0, len(ordered), batch)]
    rng.shuffle(chunks)
    return chunks


def _series_perm(sid: int, n: int) -> np.ndarray:
    """C2's per-series time permutation.  Same permutation for all eight channels."""
    return np.random.default_rng(SHUFFLE_SEED + int(sid)).permutation(n)


def _pack(C, y, off, n_on, ids, shuffle=False):
    """(B, 8, T) channels, (B, T) labels, (B, T) validity mask.

    Column index IS the online index t, because the store lays every series out
    as t = 0..n-1 contiguous.  Padded positions carry mask 0, contribute zero
    loss and are never scored.  When `shuffle`, the ONLINE CHANNELS are permuted
    within each series and the label stays at its original t (C2).
    """
    T = int(n_on[ids].max())
    B = len(ids)
    X = np.zeros((B, N_CHANNELS, T), dtype=np.float32)
    Y = np.zeros((B, T), dtype=np.float32)
    M = np.zeros((B, T), dtype=np.float32)
    for k, s in enumerate(ids):
        a, n = int(off[s]), int(n_on[s])
        blk = np.asarray(C[a:a + n])
        if shuffle:
            blk = blk[_series_perm(s, n)]
        X[k, :, :n] = blk.T
        Y[k, :n] = y[a:a + n]
        M[k, :n] = 1.0
    return X, Y, M


# -------------------------------------------------------------- pairwise sampler
def sample_pairs(Y, M, rng, m_neg=M_NEG):
    """Same-t (positive, negative) index pairs inside one packed batch.

    Groups are the online index t.  A timestep contributes only if it holds
    >= 1 positive AND >= m_neg negatives among the batch's series at that t --
    the minimum group occupancy of CRF_PROGRAM_PREREG section 1.6.  Negatives
    are drawn uniformly WITH REPLACEMENT, the sbr.pipeline._make_pairwise_t
    convention, resampled on every optimiser step.
    """
    T = Y.shape[1]
    pb, pt, nb, nt = [], [], [], []
    n_groups = 0
    for t in range(T):
        m = M[:, t] > 0
        if not m.any():
            continue
        yt = Y[:, t]
        pos = np.flatnonzero(m & (yt == 1))
        neg = np.flatnonzero(m & (yt == 0))
        if len(pos) == 0 or len(neg) < m_neg:
            continue
        n_groups += 1
        j = neg[rng.integers(0, len(neg), size=(len(pos), m_neg))]
        pb.append(np.repeat(pos, m_neg))
        nb.append(j.ravel())
        k = len(pos) * m_neg
        pt.append(np.full(k, t, dtype=np.int64))
        nt.append(np.full(k, t, dtype=np.int64))
    if not pb:
        return None
    return (np.concatenate(pb), np.concatenate(pt),
            np.concatenate(nb), np.concatenate(nt), n_groups)


# ----------------------------------------------------------------------- training
def train_fold(arm: str, fold: int, d, C, log, seed: int = SEED):
    """One outer fold.  Returns the raw logit score on that fold's validation rows.

    PURITY: `tr_series` is exactly the series of folds != fold, asserted here and
    again at emission time.  No validation series, and no lockbox series, is ever
    packed into a training batch.
    """
    import torch
    from torch import nn

    cfg = ARMS[arm]
    device = set_determinism(seed)
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    y = d.y.astype(np.float32)

    tr_series = np.flatnonzero(np.isin(d.series_fold, [f for f in FOLDS if f != fold]))
    va_series = np.flatnonzero(d.series_fold == fold)
    lb_series = np.flatnonzero(d.series_fold == -1)
    assert len(np.intersect1d(tr_series, va_series)) == 0, "PURITY VIOLATION: train n val"
    assert len(np.intersect1d(tr_series, lb_series)) == 0, "PURITY VIOLATION: train n lockbox"
    assert set(d.series_fold[tr_series]) == {f for f in FOLDS if f != fold}, \
        "PURITY VIOLATION: training series are not exactly FOLDS \\ {fold}"

    torch.manual_seed(seed * 1000 + fold)
    net = CausalTCN.build(HIDDEN, DROPOUT, seed * 1000 + fold, device, n_in=N_CHANNELS)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    n_steps = EPOCHS * int(np.ceil(len(tr_series) / BATCH_SERIES))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    batch_rng = np.random.default_rng(seed * 1000 + fold)
    pair_rng = np.random.default_rng(PAIR_SEED + fold)

    t0 = time.time()
    hist_loss, occupancy, empty_steps = [], [], 0
    net.train()
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        groups, pairs = 0, 0
        for ids in _batches(tr_series, n_on, batch_rng, BATCH_SERIES):
            X, Y, M = _pack(C, y, off, n_on, ids, shuffle=cfg["shuffle"])
            xb = torch.from_numpy(X)
            mb = torch.from_numpy(M)
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            if cfg["objective"] == "bce":
                l = nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(Y), reduction="none")
                loss = (l * mb).sum() / mb.sum()
            else:
                s = sample_pairs(Y, M, pair_rng)
                if s is None:
                    empty_steps += 1
                    loss = logits.sum() * 0.0
                else:
                    pbi, pti, nbi, nti, ng = s
                    sp = logits[torch.from_numpy(pbi), torch.from_numpy(pti)]
                    sn = logits[torch.from_numpy(nbi), torch.from_numpy(nti)]
                    loss = nn.functional.softplus(-(sp - sn)).mean()
                    groups += ng
                    pairs += len(pbi)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist_loss.append(tot / max(nb, 1))
        occupancy.append({"epoch": ep + 1, "contributing_groups": int(groups),
                          "pairs": int(pairs), "steps": int(nb)})
        el = time.time() - t0
        log(f"    {arm} fold {fold} epoch {ep+1}/{EPOCHS}  loss {hist_loss[-1]:.6f}"
            f"  groups {groups}  pairs {pairs}  ({el:.0f}s)")
        if el > MAX_FOLD_SECONDS:
            log(f"  {arm} fold {fold} EXCEEDED the 3h fold budget at epoch {ep+1}; "
                f"reporting infeasible rather than shrinking")
            return None

    net.eval()
    val_scores = {}
    with torch.no_grad():
        for i in range(0, len(va_series), BATCH_SERIES):
            ids = va_series[i:i + BATCH_SERIES]
            X, _, _ = _pack(C, y, off, n_on, ids, shuffle=cfg["shuffle"])
            p = net(torch.from_numpy(X)).numpy()
            for k, s in enumerate(ids):
                val_scores[int(s)] = p[k, :int(n_on[s])].astype(np.float32)

    rows, vals = [], []
    for s in va_series:
        a, n = int(off[s]), int(n_on[s])
        rows.append(np.arange(a, a + n, dtype=np.int64))
        vals.append(val_scores[int(s)])
    rows = np.concatenate(rows); vals = np.concatenate(vals)
    return {
        "arm": arm, "exp_id": cfg["exp_id"], "fold": int(fold),
        "rows": rows, "scores": vals,
        "state_sha256": sha_state_dict(net.state_dict()),
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "n_train_series": int(len(tr_series)), "n_val_series": int(len(va_series)),
        "loss_history": hist_loss, "occupancy": occupancy,
        "empty_pair_steps": int(empty_steps),
        "runtime_s": round(time.time() - t0, 1),
    }


def emit(res, log):
    outdir = f"{CACHE}/scores"
    os.makedirs(outdir, exist_ok=True)
    tag = f"{res['exp_id']}_fold{res['fold']}"
    np.save(f"{outdir}/{tag}.rows.npy", res["rows"])
    np.save(f"{outdir}/{tag}.scores.npy", res["scores"])
    meta = {k: v for k, v in res.items() if k not in ("rows", "scores")}
    meta["n_rows"] = int(len(res["rows"]))
    meta["env"] = env_report_notorch()
    json.dump(meta, open(f"{outdir}/{tag}.json", "w"), indent=1)
    log(f"  emitted {outdir}/{tag}.*  ({meta['n_rows']} rows, "
        f"state {res['state_sha256'][:12]})")
    return f"{outdir}/{tag}"


def env_report_notorch() -> dict:
    """env_report() without the lightgbm import -- this process must never load it."""
    import platform, torch, scipy
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__,
        "scipy": scipy.__version__, "torch": torch.__version__, "device": "cpu",
        "torch_threads": 6, "cpu_count": os.cpu_count(),
        "deterministic_algorithms": True, "lightgbm_imported": "lightgbm" in sys.modules,
    }


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-channels", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    logf = open(f"{ROOT}/logs/crf01.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    assert_no_forbidden_channel_names(CHANNEL_NAMES)
    log(f"=== CRF-01 NNCSR === {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(json.dumps(env_report_notorch()))
    d = Data()
    C = build_channels(d, log)
    if a.build_channels and a.arm is None:
        log("channel build complete")
        return 0
    assert a.arm is not None and a.fold is not None, "--arm and --fold are required"
    log(f"--- {a.arm} ({ARMS[a.arm]['exp_id']}) fold {a.fold} ---")
    res = train_fold(a.arm, a.fold, d, C, log, seed=a.seed)
    if res is None:
        log("INFEASIBLE_AT_THIS_BUDGET")
        return 2
    emit(res, log)
    log(f"  {a.arm} fold {a.fold} done in {res['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
