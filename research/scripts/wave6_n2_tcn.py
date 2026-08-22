"""W6-N2 -- TRACK B: can a learned CAUSAL TEMPORAL REPRESENTATION add information
the 500 handcrafted columns do not?

RT-970 (hidden 32) and RT-971 (hidden 64), exactly as pre-registered in
research/WAVE6_NEURAL_PREREG.md section 2.1 at 17d01b8.  Kernel 3, dilations
1/2/4/8/16/32, six residual blocks, GELU, dropout 0.1, weight-normed causal
convolutions, per-timestep linear head.  AdamW lr 3e-3, weight decay 1e-2,
batch 32 SERIES, 20 epochs, gradient clip 1.0, BCE.  None of it is tuneable
after a score is seen.

INPUT: the ten pre-registered legal channels (WAVE6_NEURAL_PREREG section 3),
computed from `hist` and `online[:t+1]` only.  No hand-engineered column, no
tau, no future observation, no final length.

BATCHING: series are length-bucketed for throughput.  That is legal here and
not a covert length channel, because the network contains no BatchNorm and no
time-axis normalisation, so a series' output cannot depend on its neighbours --
asserted, not assumed, by
tests/test_neural_causality.py::test_gate5_batch_composition_does_not_change_a_prediction.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave6_neural_lib import (CHANNEL_NAMES, CausalTCN, N_CHANNELS, causal_channels,
                              env_report, set_determinism, sha_state_dict)

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, append_result, git_sha

FOLDS = (0, 1, 2, 3, 4)
BATCH_SERIES = 32          # prereg 2.1
EPOCHS = 20                # prereg 2.1
LR = 3e-3                  # prereg 2.1
WD = 1e-2                  # prereg 2.1
DROPOUT = 0.1              # prereg 2.1
CLIP = 1.0                 # prereg 2.1
MAX_FOLD_SECONDS = 3 * 3600   # prereg section 9: report infeasible, do not shrink

ARMS = {"RT-970": dict(hidden=32), "RT-971": dict(hidden=64)}

HYP = ("A causal dilated TCN reading ten legal causal channels learns break "
       "MORPHOLOGY -- transient shock, sustained shift, recovery, repeated "
       "excursion, dependence-regime change, multi-scale persistence -- that "
       "the 500 handcrafted columns encode only incompletely. W6-E2R showed "
       "our bank beats a generic oracle bank by +0.0235 series AUC, i.e. "
       "representation quality is a live lever; this asks whether a LEARNED "
       "representation is a better one still.")
FALS = ("fails to clear +0.0030 TS-AUC over the strongest matched control on "
        ">=4/5 folds with paired bootstrap support, AND fails to add ensemble "
        "value over the seven-specialist set beyond a matched seed clone")

CHAN_CACHE = os.environ.get("W6_CHAN", f"{ROOT}/cache/w6chan")


def build_channels(d, log):
    """(n_online_rows, 10) float32 in the store's series-major row order."""
    os.makedirs(CHAN_CACHE, exist_ok=True)
    path = f"{CHAN_CACHE}/channels.npy"
    meta = f"{CHAN_CACHE}/channels.json"
    n_rows = len(d.y)
    if os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("n_rows") == n_rows and m.get("channels") == list(CHANNEL_NAMES):
            log(f"  channel cache hit: {path}")
            return np.load(path, mmap_mode="r")
    log(f"  building {n_rows} x {N_CHANNELS} causal channels ...")
    out = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float32,
                                    shape=(n_rows, N_CHANNELS))
    t0 = time.time()
    off = d.st.orow_off
    for i in range(d.st.n_series):
        h, o, _ = d.st.series(i)
        a = int(off[i])
        out[a:a + len(o)] = causal_channels(h, o)
        if (i + 1) % 2000 == 0:
            log(f"    {i + 1}/{d.st.n_series}  {time.time() - t0:.0f}s")
    out.flush(); del out
    os.replace(path + ".tmp", path)
    json.dump({"n_rows": n_rows, "channels": list(CHANNEL_NAMES)}, open(meta, "w"))
    log(f"  channels built in {time.time() - t0:.0f}s")
    return np.load(path, mmap_mode="r")


def _batches(series_ids, lens, rng, batch):
    """Length-bucketed batches, order shuffled.  Legal: see the module docstring."""
    order = np.argsort(lens[series_ids], kind="stable")
    ordered = np.asarray(series_ids)[order]
    chunks = [ordered[i:i + batch] for i in range(0, len(ordered), batch)]
    rng.shuffle(chunks)
    return chunks


def _pack(C, y, off, n_on, ids):
    T = int(n_on[ids].max())
    B = len(ids)
    X = np.zeros((B, N_CHANNELS, T), dtype=np.float32)
    Y = np.zeros((B, T), dtype=np.float32)
    M = np.zeros((B, T), dtype=np.float32)
    for k, s in enumerate(ids):
        a, n = int(off[s]), int(n_on[s])
        X[k, :, :n] = np.asarray(C[a:a + n]).T
        Y[k, :n] = y[a:a + n]
        M[k, :n] = 1.0
    return X, Y, M


def run_arm(exp_id, hidden, seed, d, C, log):
    import torch
    from torch import nn
    device = set_determinism(seed)
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    y = d.y.astype(np.float32)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per_fold, history, shas = [], {}, {}
    t0 = time.time()

    for f in FOLDS:
        ft = time.time()
        tr_series = np.flatnonzero(np.isin(d.series_fold, [x for x in FOLDS if x != f]))
        va_series = np.flatnonzero(d.series_fold == f)
        torch.manual_seed(seed * 1000 + f)
        net = CausalTCN.build(hidden, DROPOUT, seed * 1000 + f, device)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        n_steps = EPOCHS * int(np.ceil(len(tr_series) / BATCH_SERIES))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        rng = np.random.default_rng(seed * 1000 + f)

        hist = []
        net.train()
        for ep in range(EPOCHS):
            tot, nb = 0.0, 0
            for ids in _batches(tr_series, n_on, rng, BATCH_SERIES):
                X, Y, M = _pack(C, y, off, n_on, ids)
                xb = torch.from_numpy(X); yb = torch.from_numpy(Y); mb = torch.from_numpy(M)
                opt.zero_grad(set_to_none=True)
                logits = net(xb)
                l = nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none")
                loss = (l * mb).sum() / mb.sum()      # uniform over ROWS, like the control
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP)
                opt.step(); sched.step()
                tot += float(loss.detach()); nb += 1
            hist.append(tot / max(nb, 1))
            el = time.time() - ft
            log(f"    {exp_id} fold {f} epoch {ep + 1}/{EPOCHS}  bce {hist[-1]:.5f}  ({el:.0f}s)")
            if el > MAX_FOLD_SECONDS:
                log(f"  {exp_id} fold {f} EXCEEDED the pre-registered 3h fold budget "
                    f"at epoch {ep + 1}; reporting infeasible rather than shrinking")
                return None

        net.eval()
        with torch.no_grad():
            for i in range(0, len(va_series), BATCH_SERIES):
                ids = va_series[i:i + BATCH_SERIES]
                X, _, _ = _pack(C, y, off, n_on, ids)
                p = torch.sigmoid(net(torch.from_numpy(X))).numpy()
                for k, s in enumerate(ids):
                    a, n = int(off[s]), int(n_on[s])
                    oof[a:a + n] = p[k, :n]
        va_rows = d.rows_for([f])
        s = float(ts_auc_flat(oof[va_rows], d.y[va_rows], d.t[va_rows]))
        per_fold.append(s)
        history[str(f)] = hist
        shas[str(f)] = sha_state_dict(net.state_dict())
        log(f"  {exp_id} fold {f}: TS-AUC {s:.5f}  ({len(tr_series)} train series, "
            f"{len(va_series)} valid series, {time.time() - ft:.0f}s)")

    dev_rows = d.rows_for(list(FOLDS))
    pooled = float(ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows]))
    outdir = os.environ.get("SBR_OOF", f"{ROOT}/research/oof")
    np.save(f"{outdir}/{exp_id}.npy", oof)
    n_par = sum(p.numel() for p in CausalTCN.build(hidden, DROPOUT, 0, device).parameters())
    res = {
        "experiment_id": exp_id, "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(), "agent": "claude-wave6", "hypothesis": HYP,
        "falsification_condition": FALS,
        "feature_set": "causal_channels:" + "/".join(CHANNEL_NAMES),
        "n_features": N_CHANNELS, "model": f"torch_tcn_h{hidden}", "objective": "bce",
        "folds": ",".join(map(str, FOLDS)), "random_seed": seed,
        "train_series": 6400, "train_rows": int(len(dev_rows) * 0.8),
        "mean_oof_ts_auc": float(np.mean(per_fold)), "pooled_oof_ts_auc": pooled,
        "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in per_fold),
        "fold_std": float(np.std(per_fold)), "persistence": "none",
        "sample_mode": "all_rows", "training_runtime_s": round(time.time() - t0, 1),
        "causal_verified": "gates1-5@tests/test_neural_causality.py",
        "test_reduced_touched": "no", "lockbox_touched": "no", "status": "recorded",
        "notes": (f"W6-N2 track B: pre-registered causal dilated TCN, hidden {hidden}, "
                  f"kernel 3, dilations 1/2/4/8/16/32, 6 residual blocks, dropout "
                  f"{DROPOUT}, AdamW lr {LR} wd {WD}, cosine, batch {BATCH_SERIES} "
                  f"series, {EPOCHS} epochs, masked BCE uniform over rows. Ten legal "
                  f"causal channels; no handcrafted column, no tau, no n_online."),
        "protocol": "full",
    }
    append_result(res)
    return res, per_fold, history, shas, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="RT-970,RT-971")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logf = open(f"{ROOT}/logs/w6n2_detail.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    env = env_report()
    log("=== W6-N2 TRACK B -- causal dilated TCN on ten legal channels ===")
    log(json.dumps(env, indent=1))
    d = Data()
    C = build_channels(d, log)
    out = {"env": env, "git_sha": git_sha(), "channels": list(CHANNEL_NAMES), "arms": {}}
    for exp_id in a.arms.split(","):
        exp_id = exp_id.strip()
        assert exp_id in ARMS, f"{exp_id} is not a pre-registered arm; ARMS={list(ARMS)}"
        cfg = ARMS[exp_id]
        log(f"\n--- {exp_id}  {cfg} ---")
        r = run_arm(exp_id, cfg["hidden"], a.seed, d, C, log)
        if r is None:
            out["arms"][exp_id] = {"config": cfg, "status": "INFEASIBLE_AT_THIS_BUDGET"}
        else:
            res, per_fold, hist, shas, n_par = r
            out["arms"][exp_id] = {
                "config": cfg, "result": res, "per_fold": per_fold,
                "training_history_bce": hist, "model_sha256_per_fold": shas,
                "n_parameters": int(n_par)}
            log(f"  {exp_id}: mean {res['mean_oof_ts_auc']:.5f}  pooled "
                f"{res['pooled_oof_ts_auc']:.5f}  folds {res['per_fold_ts_auc']}")
        json.dump(out, open(f"{ROOT}/research/reports/wave6_n2_tcn.json", "w"), indent=1)
    log("\nW6-N2 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
