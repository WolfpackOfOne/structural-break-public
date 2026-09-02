import importlib.util, os, sys
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, sbr.pipeline as PL
spec = importlib.util.spec_from_file_location("w7", f"{ROOT}/research/scripts/wave7_teacher_nested.py")
W = importlib.util.module_from_spec(spec); sys.modules["w7"] = W; spec.loader.exec_module(W)

d = PL.Data()
print("canonical fold rows:", [len(d.rows_for([f])) for f in W.FOLDS], f"total {len(d.y):,}")
print("canonical _qpath(0,1):", os.path.basename(W._qpath(0, 1)))

W.PART_SUFFIX = W._suffix_for("alt1")
with W.alt_folds("alt1"):
    da = PL.Data()
    rows = [len(da.rows_for([f])) for f in W.FOLDS]
    print("\nalt1 fold rows     :", rows, f"total {len(da.y):,}")
    print("alt1 dev rows      :", sum(rows))
    same = all(np.array_equal(d.rows_for([f]), da.rows_for([f])) for f in W.FOLDS)
    print("alt1 folds identical to canonical:", same, "(must be False)")
    p = W._qpath(0, 1)
    print("alt1 _qpath(0,1)   :", os.path.basename(p))
    print("alt1 target exists :", os.path.exists(p), "(must be False -- guard won't trip)")
    print("canonical vector still there:",
          os.path.exists(f"{ROOT}/research/oof/nested_Q_outer0_inner1.npy"))
