import os, sys, json
import numpy as np
sys.path.insert(0,"src")
os.environ.setdefault("SBR_STORE", os.path.join(os.getcwd(),"cache","store"))
from sbr.store import load_store
from sbr.features.base import REGISTRY, load_all, make_ctx
from sbr.stream.engine import PRODUCTION_MODULES, StreamEngine
load_all(); st=load_store(); MODS=list(PRODUCTION_MODULES)

# short-to-medium series so the sweep finishes; boundary prefixes preserved
n_on = st.meta.n_online.to_numpy()
cand = np.flatnonzero((n_on >= 200) & (n_on <= 320))
sample = [int(x) for x in cand[:14]]
PREFIX=[1,2,3,5,10,16,20,32,50,64,100,128,200,256]
pib=0; pic=0; det=[]
for si in sample:
    h,o,_=st.series(si); n=len(o)
    _e=StreamEngine().fit_historical(h)
    full=np.vstack([_e.step(x) for x in o])
    for L in PREFIX:
        if L>n: continue
        e=StreamEngine().fit_historical(h)
        pre=np.vstack([e.step(x) for x in o[:L]])
        eq=(pre==full[:L])|(np.isnan(pre)&np.isnan(full[:L]))
        pic+=int(eq.size); b=int((~eq).sum()); pib+=b
        if b: det.append({"series":si,"prefix":L,"n_bad":b,"mode":"stream_truncated"})
    for L in (10,50,200):
        if L>n: continue
        ct=make_ctx(h,o[:L])
        Bt=np.hstack([np.asarray(REGISTRY[m].fn(ct)[1]) for m in MODS])
        eq=(Bt==full[:L])|(np.isnan(Bt)&np.isnan(full[:L]))
        pic+=int(eq.size); b=int((~eq).sum()); pib+=b
        if b: det.append({"series":si,"prefix":L,"n_bad":b,"mode":"batch_recomputed_on_truncated_online"})
    print(f"  {si}: cumulative {pic:,} cells, {pib} mismatches", flush=True)
print(f"\nPREFIX INVARIANCE: {len(sample)} series, prefixes {PREFIX}")
print(f"  {pic:,} cells checked, {pib} mismatches -> {'PASS' if pib==0 else 'FAIL'}")

# series independence at feature level
tgt=sample[:8]; ref={}
for si in tgt:
    h,o,_=st.series(si); _e=StreamEngine().fit_historical(h); ref[si]=np.vstack([_e.step(x) for x in o])
others=[int(x) for x in cand[20:24]]
bad=0; cells=0
for si in reversed(tgt):
    for oj in others:
        hh,oo,_=st.series(oj); e=StreamEngine().fit_historical(hh)
        for x in oo[:40]: e.step(x)
    h,o,_=st.series(si)
    _e=StreamEngine().fit_historical(h)
    got=np.vstack([_e.step(x) for x in o])
    eq=(got==ref[si])|(np.isnan(got)&np.isnan(ref[si]))
    cells+=int(eq.size); bad+=int((~eq).sum())
print(f"\nSERIES INDEPENDENCE (features): {cells:,} cells, {bad} mismatches -> {'PASS' if bad==0 else 'FAIL'}")
json.dump({"prefix_invariance":{"n_series":len(sample),"prefixes":PREFIX,"cells":pic,
           "mismatches":pib,"detail":det},
           "series_independence_features":{"cells":cells,"mismatches":bad,"n_series":len(tgt)}},
          open(os.environ["OUT"]+"/PREFIX_INDEPENDENCE.json","w"), indent=2)
print("\nwrote PREFIX_INDEPENDENCE.json")
