"""W6-N1 -- TRACK A: does a different LEARNER extract more from the SAME 500 features?

RT-960 (dropout 0.1, weight decay 1e-2) and RT-961 (dropout 0.3, weight decay
1e-1), exactly as pre-registered in research/WAVE6_NEURAL_PREREG.md section 2.1
at 17d01b8, before any neural number existed.  Nothing here may be tuned after a
score is seen.

The comparison is IDENTICAL-INFORMATION by construction: the same seven feature
modules, the same canonical folds, the same 1,000,000 uniformly sampled training
rows in the same order that `sbr.pipeline.run` draws them for the CHAMP
protocol, the same row labels, the same official scorer.  The only thing that
changes is the learner.

    control  RT-300   CHAMP-protocol LightGBM on those exact rows   0.61605
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time

import numpy as np

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave6_neural_lib import (FoldStandardiser, PROD_MODULES, build_mlp,
                              champ_fold_rows, env_report, set_determinism,
                              sha_state_dict)

from sbr.metric import ts_auc_flat
from sbr.pipeline import Data, FEAT, _stack, append_result, git_sha, load_features

FOLDS = (0, 1, 2, 3, 4)
MAX_TRAIN_ROWS = 1_000_000          # the CHAMP protocol's, unchanged
BATCH = 4096                        # prereg 2.1
EPOCHS = 12                         # prereg 2.1
LR = 1e-3                           # prereg 2.1
CLIP = 1.0                          # prereg 2.1
PRED_BLOCK = 250_000

#: the two pre-registered arms.  TWO, and no others.
ARMS = {
    "RT-960": dict(dropout=0.1, weight_decay=1e-2),
    "RT-961": dict(dropout=0.3, weight_decay=1e-1),
}

HYP = ("A different learner on the IDENTICAL 500 causal columns and the "
       "identical CHAMP training rows extracts information gradient-boosted "
       "trees leave behind. W6-E2R showed the same 500 columns gain +0.0259 "
       "series AUC purely from being refit to the right question, which makes "
       "learner capacity a first-class hypothesis rather than a formality.")
FALS = ("fails to clear +0.0030 TS-AUC over the strongest matched control on "
        ">=4/5 folds with paired bootstrap support, AND fails to add ensemble "
        "value over the seven-specialist set beyond a matched seed clone")


def run_arm(exp_id, dropout, weight_decay, seed, d, mats, names, sizes, folds_rows,
            outdir, log):
    import torch
    from torch import nn
    device = set_determinism(seed)
    oof = np.full(len(d.y), np.nan, dtype=np.float32)
    per_fold, history, shas = [], {}, {}
    t0 = time.time()

    for f in FOLDS:
        tr_rows, va_rows = folds_rows[f]
        ft = time.time()
        Xtr_raw = _stack(mats, names, tr_rows, np.arange(len(names)))
        std = FoldStandardiser(sizes).fit(Xtr_raw)      # GATE 4: training rows only
        Xtr = std.transform(Xtr_raw)
        del Xtr_raw
        ytr = d.y[tr_rows].astype(np.float32)

        torch.manual_seed(seed * 1000 + f)
        net = build_mlp(Xtr.shape[1], dropout, seed * 1000 + f, device)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=weight_decay)
        n_steps = EPOCHS * (len(Xtr) // BATCH)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        lossf = nn.BCEWithLogitsLoss()
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(ytr)
        gen = torch.Generator().manual_seed(seed * 1000 + f)

        net.train()
        hist = []
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xt), generator=gen)
            tot, nb = 0.0, 0
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad(set_to_none=True)
                out = net(Xt[idx])[:, 0]
                loss = lossf(out, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP)
                opt.step()
                sched.step()
                tot += float(loss.detach()); nb += 1
            hist.append(tot / max(nb, 1))
            log(f"    {exp_id} fold {f} epoch {ep + 1}/{EPOCHS}  bce {hist[-1]:.5f}"
                f"  ({time.time() - ft:.0f}s)")
        del Xt, Xtr

        net.eval()
        with torch.no_grad():
            for a in range(0, len(va_rows), PRED_BLOCK):
                blk = va_rows[a:a + PRED_BLOCK]
                Xv = std.transform(_stack(mats, names, blk, np.arange(len(names))))
                p = torch.sigmoid(net(torch.from_numpy(Xv))[:, 0]).numpy()
                oof[blk] = p.astype(np.float32)
                del Xv
        s = float(ts_auc_flat(oof[va_rows], d.y[va_rows], d.t[va_rows]))
        per_fold.append(s)
        history[str(f)] = hist
        shas[str(f)] = sha_state_dict(net.state_dict())
        log(f"  {exp_id} fold {f}: TS-AUC {s:.5f}  "
            f"({len(tr_rows)} train rows, {len(va_rows)} valid rows, {time.time() - ft:.0f}s)")

    dev_rows = d.rows_for(list(FOLDS))
    pooled = float(ts_auc_flat(oof[dev_rows], d.y[dev_rows], d.t[dev_rows]))
    np.save(f"{outdir}/{exp_id}.npy", oof)
    n_par = sum(p.numel() for p in build_mlp(len(names) + len(sizes), dropout, 0,
                                             device).parameters())
    res = {
        "experiment_id": exp_id, "date": time.strftime("%Y-%m-%d %H:%M"),
        "git_sha": git_sha(), "agent": "claude-wave6", "hypothesis": HYP,
        "falsification_condition": FALS, "feature_set": ",".join(PROD_MODULES),
        "n_features": len(names) + len(sizes), "model": "torch_mlp",
        "objective": "bce", "folds": ",".join(map(str, FOLDS)), "random_seed": seed,
        "train_series": 8000, "train_rows": MAX_TRAIN_ROWS,
        "mean_oof_ts_auc": float(np.mean(per_fold)), "pooled_oof_ts_auc": pooled,
        "per_fold_ts_auc": ";".join(f"{x:.5f}" for x in per_fold),
        "fold_std": float(np.std(per_fold)), "persistence": "none",
        "sample_mode": "uniform", "training_runtime_s": round(time.time() - t0, 1),
        "causal_verified": "gates1-5@tests/test_neural_causality.py",
        "test_reduced_touched": "no", "lockbox_touched": "no", "status": "recorded",
        "notes": (f"W6-N1 track A: pre-registered MLP 500(+7 group indicators)"
                  f"->256->128->32->1, GELU/LayerNorm, dropout {dropout}, AdamW "
                  f"lr {LR} wd {weight_decay}, cosine, batch {BATCH}, {EPOCHS} "
                  f"epochs, BCE. IDENTICAL rows/features/folds/scorer to RT-300."),
        "protocol": "full",
    }
    append_result(res)
    return res, per_fold, history, shas, n_par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="RT-960,RT-961")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    outdir = os.environ.get("SBR_OOF", f"{ROOT}/research/oof")
    os.makedirs(outdir, exist_ok=True)
    logf = open(f"{ROOT}/logs/w6n1_detail.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    env = env_report()
    log("=== W6-N1 TRACK A -- MLP on the identical 500 causal columns ===")
    log(json.dumps(env, indent=1))

    d = Data()
    mats, names = load_features(PROD_MODULES)
    sizes = [len(json.load(open(f"{FEAT}/{m}.cols.json"))["cols"]) for m in PROD_MODULES]
    assert sum(sizes) == len(names) == 500, (sizes, len(names))
    folds_rows = champ_fold_rows(d, FOLDS, MAX_TRAIN_ROWS, seed=a.seed)
    log(f"rows: " + ", ".join(f"f{f} {len(folds_rows[f][0])}tr/{len(folds_rows[f][1])}va"
                              for f in FOLDS))

    out = {"env": env, "git_sha": git_sha(), "arms": {}}
    for exp_id in a.arms.split(","):
        exp_id = exp_id.strip()
        assert exp_id in ARMS, f"{exp_id} is not a pre-registered arm; ARMS={list(ARMS)}"
        cfg = ARMS[exp_id]
        log(f"\n--- {exp_id}  {cfg} ---")
        res, per_fold, hist, shas, n_par = run_arm(
            exp_id, cfg["dropout"], cfg["weight_decay"], a.seed, d, mats, names,
            sizes, folds_rows, outdir, log)
        out["arms"][exp_id] = {
            "config": cfg, "result": res, "per_fold": per_fold,
            "training_history_bce": hist, "model_sha256_per_fold": shas,
            "n_parameters": int(n_par),
        }
        log(f"  {exp_id}: mean {res['mean_oof_ts_auc']:.5f}  pooled "
            f"{res['pooled_oof_ts_auc']:.5f}  folds {res['per_fold_ts_auc']}")
        json.dump(out, open(f"{ROOT}/research/reports/wave6_n1_mlp.json", "w"), indent=1)
    log("\nW6-N1 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
