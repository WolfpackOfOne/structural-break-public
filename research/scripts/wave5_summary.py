"""Assemble the wave-5 executive table from the ledger and the report JSONs.

Reads only what the runs actually wrote, so the table in the final document
cannot drift from the numbers on disk.
"""
from __future__ import annotations

import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
REPORTS = f"{ROOT}/research/reports"

LABEL = {
    "RT-730": "W5-E7  + m11_focus (exact max over tau)",
    "RT-740": "W5-E2  + m10_persist (outlier vs bulk scale)",
    "RT-750": "W5-E4/5/6  + m12_rdep (residual dist / CUSUMSQ / dep LR)",
    "RT-760": "W5-E10  + all three blocks",
    "RT-700": "W5-E9a  pairwise_w (n_neg-weighted pairs)",
    "RT-701": "W5-E9b  pairwise_h (weighted squared hinge)",
    "RT-702": "W5-E9   CONTROL incumbent pairwise_t via the hook",
    "RT-710": "W5-E3   CONTROL wbinary, uniform weights",
    "RT-711": "W5-E3   hard-negative reweighting",
    "RT-712": "W5-E3   hard-negative oversampling",
    "RT-731": "W5-E7   CHAMP protocol + m11_focus",
    "RT-751": "W5-E4/5/6  CHAMP protocol + m12_rdep",
}
#: each candidate's matched control, and what kind of comparison it is.
#: NOTE the "CHAMP standalone stream" rows compare STREAM to STREAM.  That is
#: NOT the binding ensemble comparison -- (S + C) - (S + N) -- which lives in
#: research/reports/wave5_staged_binding.log and is a much smaller number,
#: because an eighth member carries 1/8 of the blend weight.
CONTROL = {
    "RT-730": ("RT-301", "ABL feature block"),
    "RT-740": ("RT-301", "ABL feature block"),
    "RT-750": ("RT-301", "ABL feature block"),
    "RT-760": ("RT-301", "ABL feature block"),
    "RT-700": ("RT-702", "objective"),
    "RT-701": ("RT-702", "objective"),
    "RT-711": ("RT-710", "curriculum"),
    "RT-712": ("RT-710", "curriculum"),
    "RT-731": ("RT-401", "CHAMP standalone stream"),
    "RT-751": ("RT-401", "CHAMP standalone stream"),
}


def pf(s):
    return [float(x) for x in str(s).split(";")]


def main():
    d = pd.read_csv(f"{ROOT}/research/RESULTS.csv").drop_duplicates("experiment_id", keep="last")
    d = d.set_index("experiment_id")
    rows = []
    for e, (ctl, kind) in CONTROL.items():
        if e not in d.index or ctl not in d.index:
            rows.append({"id": e, "what": LABEL.get(e, e), "control": ctl, "kind": kind,
                         "score": None, "delta": None, "folds": None})
            continue
        a, b = pf(d.loc[e, "per_fold_ts_auc"]), pf(d.loc[ctl, "per_fold_ts_auc"])
        dd = [x - y for x, y in zip(a, b)]
        rows.append({"id": e, "what": LABEL.get(e, e), "control": ctl, "kind": kind,
                     "score": float(d.loc[e, "mean_oof_ts_auc"]),
                     "ctl_score": float(d.loc[ctl, "mean_oof_ts_auc"]),
                     "delta": float(np.mean(dd)),
                     "folds": f"{sum(x>0 for x in dd)}/5",
                     "runtime_s": float(d.loc[e, "training_runtime_s"])})
    T = pd.DataFrame(rows)
    print("=== WAVE-5 RUNS AGAINST THEIR MATCHED CONTROLS ===\n")
    for _, r in T.iterrows():
        if r["score"] is None or (isinstance(r["score"], float) and np.isnan(r["score"])):
            print(f"  {r['id']}  {r['what'][:52]:52s}  NOT RUN")
        else:
            print(f"  {r['id']}  {r['what'][:52]:52s}  {r['score']:.5f}  "
                  f"vs {r['control']} {r['delta']:+.5f}  {r['folds']}")
    T.to_csv(f"{REPORTS}/wave5_executive.csv", index=False)

    # the binding seed-clone comparisons, from whichever report files exist
    print("\n=== BINDING SEED-CLONE COMPARISONS ===\n")
    p = f"{REPORTS}/wave5_abl_blocks.json"
    if os.path.exists(p):
        j = json.load(open(p))
        bar = j["seed_clone_blend"]
        print(f"  ABL two-model blend bar (control + seed clone): "
              f"{bar['mean']:.5f}  (+{bar['delta']:.5f} over control)")
        for e, a in j["arms"].items():
            bl = a["blend"]
            print(f"    {e}: blend {bl['mean']:.5f}  vs seed clone "
                  f"{bl['delta_vs_seed_clone_blend']:+.5f}  "
                  f"({bl['n_folds_better_than_seed_blend']}/5)  -> {a['verdict']}")
    for f in sorted(glob.glob(f"{REPORTS}/wave5_W5-*.json")):
        j = json.load(open(f))
        v = j["verdict"]
        print(f"  {j['experiment']} ({j['candidate']}): "
              f"(S+C)-(S+N) = {v['delta_vs_seed_control']:+.5f} on {v['n_folds_better']}/5"
              f"  -> {v['outcome']}")
    print(f"\nwrote {REPORTS}/wave5_executive.csv")


if __name__ == "__main__":
    main()
