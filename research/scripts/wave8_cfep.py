"""M4 -- Causal Future-Embedding Pretraining (CFEP). Fold-0 pilot.

Pre-registered in research/WAVE8_FUTURE_AWARE_PREREG.md section 5.4. Reuses
Wave 6's already-validated causal-TCN shell (RT-970/971:
research/scripts/wave6_neural_lib.py CausalTCN, ten legal causal channels,
kernel 3, dilations 1/2/4/8/16/32, six residual blocks) with one change: a
16-dim embedding bottleneck before the head, so CFEP-B (BCE) and CFEP-C
(future-predictive) are a matched-capacity ablation of the OBJECTIVE only.

Torch training runs in THIS process; the downstream LightGBM step is invoked
as a SEPARATE subprocess (PROTOCOL.md / commit 1068b96: torch and LightGBM
segfault sharing a process).

CFEP-C's target reuses SST's already-cached true h=200 structural targets
(research/oof/wave8_sst/Y_h200.npy / valid_h200.npy) -- masked MSE, uniform
over ELIGIBLE rows only (never an availability feature).

Usage:
    python wave8_cfep.py --train-embedding --head bce    # RT-1041 embedding
    python wave8_cfep.py --train-embedding --head future  # RT-1042 embedding
    python wave8_cfep.py --extract --head bce
    python wave8_cfep.py --extract --head future
    python wave8_classify.py --candidate cfep_bce   # separate subprocess (LightGBM)
    python wave8_classify.py --candidate cfep_future
    python wave8_cfep.py --analyze
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave6_neural_lib import N_CHANNELS, causal_channels, env_report, set_determinism, sha_state_dict
from wave6_n2_tcn import build_channels, _batches, CHAN_CACHE
from sbr.pipeline import Data, git_sha
from wave5_lib import FOLDS, OOFDIR, REPORTS

OUTER_F = 0
HIDDEN = 32
EMB_DIM = 16
KERNEL = 3
DILATIONS = (1, 2, 4, 8, 16, 32)
BATCH_SERIES = 32
EPOCHS = 20
LR = 3e-3
WD = 1e-2
DROPOUT = 0.1
CLIP = 1.0
CACHE = f"{OOFDIR}/wave8_cfep"
os.makedirs(CACHE, exist_ok=True)


def build_net(device, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class CausalConv(nn.Module):
        def __init__(self, cin, cout, d):
            super().__init__()
            self.pad = (KERNEL - 1) * d
            self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(cin, cout, KERNEL, dilation=d))

        def forward(self, x):
            return self.conv(nn.functional.pad(x, (self.pad, 0)))

    class Block(nn.Module):
        def __init__(self, cin, cout, d):
            super().__init__()
            self.c1 = CausalConv(cin, cout, d)
            self.c2 = CausalConv(cout, cout, d)
            self.act = nn.GELU()
            self.drop = nn.Dropout(DROPOUT)
            self.down = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            h = self.drop(self.act(self.c1(x)))
            h = self.drop(self.act(self.c2(h)))
            return self.act(h + self.down(x))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [Block(N_CHANNELS if i == 0 else HIDDEN, HIDDEN, d) for i, d in enumerate(DILATIONS)])
            self.emb_head = nn.Conv1d(HIDDEN, EMB_DIM, 1)   # the 16-dim bottleneck, shared by both objectives

        def embed(self, x):                # (B, C, T) -> (B, EMB_DIM, T)
            for b in self.blocks:
                x = b(x)
            return self.emb_head(x)

    return Net().to(device)


def _pack_multi(C, off, n_on, ids, extra=None, extra_dim=0):
    T = int(n_on[ids].max())
    B = len(ids)
    X = np.zeros((B, N_CHANNELS, T), dtype=np.float32)
    M = np.zeros((B, T), dtype=np.float32)
    Y = np.zeros((B, T, max(extra_dim, 1)), dtype=np.float32)
    YM = np.zeros((B, T), dtype=np.float32)
    for k, s in enumerate(ids):
        a, n = int(off[s]), int(n_on[s])
        X[k, :, :n] = np.asarray(C[a:a + n]).T
        M[k, :n] = 1.0
        if extra is not None:
            ev, em = extra
            Y[k, :n, :extra_dim] = ev[a:a + n]
            YM[k, :n] = em[a:a + n]
    return X, M, Y, YM


def train(head, seed=0):
    import torch
    from torch import nn
    assert head in ("bce", "future")
    device = set_determinism(seed)
    d = Data()
    C = build_channels(d, print)
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    y = d.y.astype(np.float32)

    if head == "future":
        Yh200 = np.load(f"{OOFDIR}/wave8_sst/Y_h200.npy").astype(np.float32)   # (n,8)
        valid = np.load(f"{OOFDIR}/wave8_sst/valid_h200.npy").astype(np.float32)
        # robust per-channel standardization, fit on OUTER-TRAIN eligible rows only --
        # channel 0 (cz100_cur) ranges to +-6000 unstandardized; MSE against that with
        # AdamW lr=3e-3 (tuned for BCE) diverges to NaN within the first few batches
        # (observed: all 20 epochs NaN on the unstandardized target). Median/IQR, not
        # mean/std, because these are the same heavy-tailed structural channels m00-m07
        # already document as outlier-prone.
        tr_row_mask = np.isin(d.series_fold[d.sidx], [g for g in FOLDS if g != OUTER_F])
        fit_mask = tr_row_mask & (valid > 0.5)
        med = np.nanmedian(np.where(fit_mask[:, None], Yh200, np.nan), axis=0)
        q75 = np.nanpercentile(np.where(fit_mask[:, None], Yh200, np.nan), 75, axis=0)
        q25 = np.nanpercentile(np.where(fit_mask[:, None], Yh200, np.nan), 25, axis=0)
        iqr = q75 - q25
        iqr[~np.isfinite(iqr) | (iqr <= 1e-6)] = 1.0
        med[~np.isfinite(med)] = 0.0
        Yh200 = np.clip((Yh200 - med) / iqr, -20.0, 20.0).astype(np.float32)
        # NaN * 0 is NaN, not 0 -- the mask (YM) alone does not stop an ineligible
        # row's NaN target from poisoning the whole batch loss once packed into a
        # dense tensor; replace with a finite placeholder now, mask does the rest.
        Yh200 = np.nan_to_num(Yh200, nan=0.0)
        with open(f"{CACHE}/target_standardization.json", "w") as f:
            json.dump({"median": med.tolist(), "iqr": iqr.tolist()}, f, indent=2)
        extra, extra_dim = (Yh200, valid), 8
    else:
        extra, extra_dim = (y[:, None], np.ones_like(y)), 1

    net = build_net(device, seed)
    task_head = nn.Conv1d(EMB_DIM, extra_dim, 1).to(device)
    params = list(net.parameters()) + list(task_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    tr_series = np.flatnonzero(np.isin(d.series_fold, [g for g in FOLDS if g != OUTER_F]))
    n_steps = EPOCHS * int(np.ceil(len(tr_series) / BATCH_SERIES))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    hist = []
    net.train(); task_head.train()
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for ids in _batches(tr_series, n_on, rng, BATCH_SERIES):
            X, M, Y, YM = _pack_multi(C, off, n_on, ids, extra, extra_dim)
            xb = torch.from_numpy(X)
            emb = net.embed(xb)                       # (B, EMB_DIM, T)
            pred = task_head(emb)                      # (B, extra_dim, T)
            yb = torch.from_numpy(Y).permute(0, 2, 1)   # (B, extra_dim, T)
            mb = torch.from_numpy(M) * torch.from_numpy(YM)   # (B, T): row present AND target eligible
            mb3 = mb.unsqueeze(1).expand_as(pred)
            opt.zero_grad(set_to_none=True)
            if head == "bce":
                l = nn.functional.binary_cross_entropy_with_logits(pred, yb, reduction="none")
            else:
                l = (pred - yb) ** 2
            denom = mb3.sum().clamp_min(1.0)
            loss = (l * mb3).sum() / denom
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        hist.append(tot / max(nb, 1))
        print(f"  CFEP-{head} epoch {ep+1}/{EPOCHS}  loss {hist[-1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    sd = {"net": net.state_dict(), "task_head": task_head.state_dict()}
    torch.save(sd, f"{CACHE}/model_{head}.pt")
    with open(f"{CACHE}/history_{head}.json", "w") as f:
        json.dump({"loss_history": hist, "runtime_s": time.time() - t0,
                   "sha": sha_state_dict(net.state_dict())}, f, indent=2)
    print(f"CFEP-{head} embedding trained, {time.time()-t0:.1f}s -> {CACHE}/model_{head}.pt")


def extract(head, seed=0):
    import torch
    device = set_determinism(seed)
    d = Data()
    C = build_channels(d, print)
    off = d.st.orow_off
    n_on = d.st.meta.n_online.to_numpy().astype(np.int64)
    net = build_net(device, seed)
    sd = torch.load(f"{CACHE}/model_{head}.pt", map_location=device)
    net.load_state_dict(sd["net"])
    net.eval()
    Z = np.zeros((len(d.y), EMB_DIM), dtype=np.float32)
    all_series = np.arange(d.st.n_series)
    with torch.no_grad():
        for i in range(0, len(all_series), BATCH_SERIES):
            ids = all_series[i:i + BATCH_SERIES]
            T = int(n_on[ids].max())
            X = np.zeros((len(ids), N_CHANNELS, T), dtype=np.float32)
            for k, s in enumerate(ids):
                a, n = int(off[s]), int(n_on[s])
                X[k, :, :n] = np.asarray(C[a:a + n]).T
            emb = net.embed(torch.from_numpy(X)).numpy()   # (B, EMB_DIM, T)
            for k, s in enumerate(ids):
                a, n = int(off[s]), int(n_on[s])
                Z[a:a + n] = emb[k, :, :n].T
    np.save(f"{CACHE}/embedding_{head}.npy", Z)
    print(f"CFEP-{head} embedding extracted for all {len(d.y)} rows -> {CACHE}/embedding_{head}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-embedding", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--head", choices=["bce", "future"])
    args = ap.parse_args()
    if args.train_embedding:
        train(args.head)
    elif args.extract:
        extract(args.head)
    else:
        raise SystemExit("pass --train-embedding --head bce|future, or --extract --head bce|future")


if __name__ == "__main__":
    main()
