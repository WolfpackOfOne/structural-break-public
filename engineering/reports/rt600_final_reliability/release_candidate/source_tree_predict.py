"""Same 67-series prediction dump, but run from the SOURCE TREE instead of the
packaged artifact.  Comparing this against the packaged dump isolates whether
packaging changed behaviour."""
import os, sys, json
import numpy as np
WT=os.environ["WT"]; SP=os.environ["SP"]
sys.path.insert(0, f"{WT}/src"); sys.path.insert(0, f"{WT}/research/scripts")
from sbr.store import load_store
from sbr.production.model import ProductionModel
from local_runner import run_infer
DIVERGENT=[2575,4746,5002,5111,7395,8581]
st=load_store()
n_on=st.meta.n_online.to_numpy()
pool=np.setdiff1d(np.flatnonzero(n_on>=50), np.asarray(DIVERGENT))
extra=np.random.default_rng(7).choice(pool, 67-len(DIVERGENT), replace=False)
idx=DIVERGENT+sorted(int(x) for x in extra)
mdl=os.environ["SBR_MODEL_DIR"]
_M=ProductionModel.load(mdl)
def infer(datasets, model_directory_path):
    yield
    for xh, xo in datasets:
        _M.start_series(np.asarray(xh, dtype=np.float64))
        for p in xo:
            yield _M.step(float(p))
series=[(st.series(i)[0], st.series(i)[1]) for i in idx]
out=run_infer(infer, series, mdl)
np.savez_compressed(f"{SP}/srctree_preds.npz", ids=np.array(idx),
                    lens=np.array([len(s) for s in out]),
                    scores=np.concatenate([np.asarray(s,dtype=np.float64) for s in out]))
print("wrote srctree_preds.npz", sum(len(s) for s in out), "predictions")
