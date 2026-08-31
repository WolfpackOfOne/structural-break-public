"""Does the refactored wave7_teacher_nested produce bitwise-identical teachers?

Loads the pre-edit version from git alongside the edited one and compares:
  A. old train_inner_teacher   vs  new train_inner_teacher      (0,1)
  B. new cmd pair path         vs  old train_inner_teacher      (0,1) and (1,0)

Reduced scale (fewer rows/rounds) so the check is cheap; the code paths under
test are identical to the full-scale ones.
"""
import importlib.util, os, sys
ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
S = os.path.dirname(os.path.abspath(__file__))
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/research/scripts")
import numpy as np, sbr.pipeline as PL

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

NEW = load("w7_new", f"{ROOT}/research/scripts/wave7_teacher_nested.py")
OLD = load("w7_old", f"{S}/w7_old.py")

ROWS, ROUNDS = 120_000, 20
for M in (NEW, OLD):
    p = dict(M.ARM_B_PARAMS); p["n_estimators"] = ROUNDS
    M.ARM_B_PARAMS = p
    M.MAX_TRAIN_ROWS = ROWS

d = PL.Data(); mats, names = PL.load_features(NEW.FULL)
keep = np.arange(len(names)); final_of_row = NEW.last_row_lookup(d)[d.sidx]
args = (d, mats, names, keep, final_of_row)

print("A. old vs new train_inner_teacher, (outer=0, inner=1)", flush=True)
rows_o, pred_o = OLD.train_inner_teacher(0, 1, *args)
rows_n, pred_n = NEW.train_inner_teacher(0, 1, *args)
print(f"   rows equal      : {np.array_equal(rows_o, rows_n)}  ({len(rows_o):,})")
print(f"   preds bitwise eq: {np.array_equal(pred_o, pred_n)}  max|diff| {np.abs(pred_o-pred_n).max():.3e}")

print("\nB. pair booster serves BOTH roles, vs old path for each", flush=True)
bst, tf = NEW._fit_pair_booster(0, 1, *args)
print(f"   pair train_folds: {tf}")
r_g, p_g = NEW._predict_teacher_on(bst, 1, *args)      # (outer=0, inner=1)
r_f, p_f = NEW._predict_teacher_on(bst, 0, *args)      # (outer=1, inner=0)
print(f"   vs old (0,1): rows {np.array_equal(r_g, rows_o)}  preds {np.array_equal(p_g, pred_o)}  "
      f"max|diff| {np.abs(p_g-pred_o).max():.3e}")
rows_o2, pred_o2 = OLD.train_inner_teacher(1, 0, *args)
print(f"   vs old (1,0): rows {np.array_equal(r_f, rows_o2)}  preds {np.array_equal(p_f, pred_o2)}  "
      f"max|diff| {np.abs(p_f-pred_o2).max():.3e}")

ok = (np.array_equal(pred_o, pred_n) and np.array_equal(p_g, pred_o)
      and np.array_equal(p_f, pred_o2))
print(f"\nVERDICT: {'REFACTOR IS BITWISE FAITHFUL' if ok else 'MISMATCH -- DO NOT USE'}")
sys.exit(0 if ok else 1)
