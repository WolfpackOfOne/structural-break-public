"""D4: causal excursion-duration / excursion-mass state variables. Descriptive."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from sbr.store import load_store
from sbr.transforms import HistParams, ar_filter_causal, _ar_resid

st = load_store(f"{ROOT}/cache/store")
S = len(st.meta)
WS = (32, 64, 128)
CH = ("loc", "scale", "res")
COLS = []
for ch in CH:
    for w in WS:
        COLS += [f"{ch}{w}_run90", f"{ch}{w}_run99", f"{ch}{w}_mass90", f"{ch}{w}_maxrun90"]
n_rows = int(st.meta.n_online.sum())
OUTM = np.full((n_rows, len(COLS)), np.nan, dtype=np.float32)

def rollmean(c, w):
    return (c[w:] - c[:-w]) / w

t0 = time.time(); pos = 0
for i in range(S):
    h = st.hist(i); o = st.online(i); n = len(o)
    hp = HistParams(h, ar_order=2)
    zh = (h - hp.mu)/hp.sd
    zo = (o - hp.mu)/hp.sd
    eh = np.concatenate([np.zeros(2), _ar_resid(zh, hp.ar_coef)])/hp.ar_sigma
    eo = ar_filter_causal(zo, hp.ar_coef, zh)/hp.ar_sigma
    streams = {"loc": (zh, zo), "scale": (zh*zh, zo*zo), "res": (eh*eh, eo*eo)}
    j = 0
    for ch in CH:
        sh, so = streams[ch]
        ch_ = np.concatenate([[0.0], np.cumsum(sh)])
        co_ = np.concatenate([[0.0], np.cumsum(so)])
        for w in WS:
            if w >= len(sh)//2 or w > n:
                j += 4; continue
            nul = rollmean(ch_, w)                      # history-only null
            q90, q99 = np.quantile(np.abs(nul - np.median(nul)), [0.90, 0.99])
            med = np.median(nul)
            on = np.full(n, np.nan)
            on[w-1:] = rollmean(co_, w)
            dev = np.abs(on - med)
            hot90 = dev > q90; hot99 = dev > q99
            hot90[:w-1] = False; hot99[:w-1] = False
            # causal run length: consecutive True up to and including t
            def runlen(mask):
                r = np.zeros(n); acc = 0
                for k in range(n):
                    acc = acc + 1 if mask[k] else 0
                    r[k] = acc
                return r
            r90 = runlen(hot90); r99 = runlen(hot99)
            # excursion mass: cumulative (dev-q90)+ since the last re-entry
            excess = np.where(hot90, dev - q90, 0.0)
            mass = np.zeros(n); acc = 0.0
            for k in range(n):
                acc = acc + excess[k] if hot90[k] else 0.0
                mass[k] = acc
            OUTM[pos:pos+n, j]   = r90
            OUTM[pos:pos+n, j+1] = r99
            OUTM[pos:pos+n, j+2] = mass
            OUTM[pos:pos+n, j+3] = np.maximum.accumulate(r90)
            j += 4
    pos += n
    if i % 1000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
np.save(f"{OUT}/d4_exc.npy", OUTM)
np.save(f"{OUT}/d4_cols.npy", np.array(COLS))
print("done", time.time()-t0)
