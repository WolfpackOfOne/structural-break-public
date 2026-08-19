"""Execute a built submission notebook's code cells in an ISOLATED directory and
run the resulting train()/infer() through the Crunch-contract harness.

This is the closest thing to `crunch test` that can run without network access:
it proves the notebook is self-contained (no import of this repo), that the
payload verifies, that train() installs a loadable model, and that infer()
satisfies the streaming contract on real series.
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--nb", default="/home/claude/sb/submissions/rt100_streaming.ipynb")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--fold", type=int, default=0)
a = ap.parse_args()

# --- pull the series BEFORE we sandbox sys.path -----------------------------
sys.path.insert(0, "/home/claude/sb/src")
sys.path.insert(0, "/home/claude/sb/research/scripts")
from sbr.store import load_store
from sbr.pipeline import Data
from local_runner import check_all

st = load_store(); d = Data()
idx = np.flatnonzero(d.series_fold == a.fold)
pick = np.random.default_rng(0).choice(idx, min(a.n, len(idx)), replace=False)
series = [(st.series(int(i))[0], st.series(int(i))[1]) for i in pick]
labels = [st.labels(int(i)) for i in pick]

# --- execute the notebook in a scratch dir with THIS repo removed from path --
nb = json.load(open(a.nb))
code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
work = tempfile.mkdtemp(prefix="sbrnb_")
os.chdir(work)
clean_path = [p for p in sys.path if "/home/claude/sb" not in p]
saved_path, saved_mods = list(sys.path), dict(sys.modules)
for m in [m for m in list(sys.modules) if m == "sbr" or m.startswith("sbr.")]:
    del sys.modules[m]
sys.path[:] = clean_path

g = {"__name__": "__main__"}
t0 = time.time()
for i, src in enumerate(code):
    exec(compile(src, f"<cell {i}>", "exec"), g)
print(f"notebook cells executed in {time.time()-t0:.1f}s from {work}")
assert g["sbr" if False else "ProductionModel"], "entry cell did not import the model"
import sbr as _s
assert "/home/claude/sb" not in os.path.dirname(_s.__file__), \
    f"notebook imported THIS repo, not its payload: {_s.__file__}"
print("payload isolation OK:", os.path.dirname(_s.__file__))

mdir = os.path.join(work, "model_out")
os.makedirs(mdir, exist_ok=True)
g["train"]([], mdir)
print("train() installed:", sorted(os.listdir(mdir)))

res = check_all(g["infer"], series, mdir, labels=labels)
res["notebook"] = a.nb
res["n_series"] = len(series)
json.dump(res, open("/home/claude/sb/research/reports/submission_notebook_test.json", "w"), indent=2)
print("PASS" if res["PASS"] else "FAIL")
