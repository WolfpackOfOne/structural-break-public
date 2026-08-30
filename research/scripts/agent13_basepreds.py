"""Train the base-model zoo with a leakage-safe nested design on the SCREEN store.

OUTER: valid = screen fold 0 (screen protocol).  Base models trained on folds 1-4.
INNER: the 4 training folds are split into two inner blocks A={1,2}, B={3,4}.
       For each base model we fit on A and predict B, and fit on B and predict A.
       Every inner prediction therefore comes from a model that never saw that
       series -> those rows are legal stacker-training input.
Because the row cap (300k) binds in both cases, inner and outer models see the
same NUMBER of training rows (300k) -- only the number of distinct series differs
(1000 vs 2000), so inner/outer prediction scales stay comparable.

Outputs research/artifacts/agent13/preds_<model>.npz with arrays
  outer  (len rows fold 0)   inner (len rows folds 1-4)
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, "/home/claude/sb/scripts")
import agent13_common as C

CAP = 120_000

ORDER = ["lgb_all", "lgb_m03", "lgb_m04", "lgb_m00", "lgb_m01", "lgb_m02",
         "et_all", "logit_rank", "lgb_deep", "lgb_stump"]

def main(only=None):
    d = C.data()
    outer_tr = d.rows_for([1, 2, 3, 4])
    outer_va = d.rows_for([0])
    A = d.rows_for([1, 2]); B = d.rows_for([3, 4])
    np.save(f"{C.ART}/rows_outer_va.npy", outer_va)
    np.save(f"{C.ART}/rows_innerA.npy", A)
    log = []
    for name in (only or ORDER):
        out_path = f"{C.ART}/preds_{name}.npz"
        if os.path.exists(out_path):
            print("skip", name, flush=True); continue
        t0 = time.time()
        print("start", name, flush=True)
        po, rt1 = C.fit_predict(name, outer_tr, outer_va, seed=0, cap=CAP)
        s_out = C.tsauc(po, outer_va)
        pa, rt2 = C.fit_predict(name, B, A, seed=0, cap=CAP)   # fit B, predict A
        s_in = C.tsauc(pa, A)
        np.savez(out_path, outer=po, innerA=pa)
        rec = dict(model=name, outer_tsauc=round(s_out, 5), innerA_tsauc=round(s_in, 5),
                   secs=round(time.time() - t0, 1), fits=[rt1, rt2])
        log.append(rec)
        print(json.dumps(rec), flush=True)
    with open(f"{C.ART}/basepreds_log.jsonl", "a") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main(sys.argv[1:] or None)
