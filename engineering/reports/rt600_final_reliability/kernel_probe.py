import os, sys, math, json
import numpy as np
sys.path.insert(0, "src")
os.environ.setdefault("SBR_STORE", os.path.join(os.getcwd(), "cache", "store"))

from sbr.store import load_store
from sbr.features import m07_bayes as M
from sbr.stream.s_m07_bayes import _BocpdStream
from sbr.features.base import make_ctx

st = load_store()
SER = int(os.environ.get("SER", "5111"))
h, o, tau = st.series(SER)
ctx = make_ctx(h, o)

# rebuild g / gh exactly as m07_bayes.build does
hp = ctx.hp
zh = (np.asarray(ctx.hist, dtype=np.float64) - hp.mu) / hp.sd
zo = np.asarray(ctx.tr["mean"], dtype=np.float64)
coef = M._fit_ar(zh, M.AR_ORDER)
eh = M._ar_resid(zh, coef)
if eh.shape[0] < 200:
    eh = zh; coef = np.zeros(M.AR_ORDER)
sig = max(float(eh.std(ddof=1)), 1e-9)
eh = eh / sig
eo = M.ar_filter_causal(zo, coef, zh) / sig
srt = np.sort(eh); H = srt.shape[0]
def _ns(x):
    lo = np.searchsorted(srt, x, side="left"); hi = np.searchsorted(srt, x, side="right")
    return np.clip(M.ndtri((0.5*(lo+hi)+0.5)/(H+1.0)), -M.GCLIP, M.GCLIP)
g = _ns(eo); gh = _ns(eh)

lbh, lb1 = math.log(M.BO_HAZ), math.log1p(-M.BO_HAZ)
v_g = max(float(gh.var()), 1e-6)
be0 = (M.BO_AL0 - 1.0) * v_g
mu0 = float(gh.mean())
ghh = gh[-M.BO_HIST_MAX:] if gh.shape[0] > M.BO_HIST_MAX else gh

def run_stream(x):
    bo = _BocpdStream(mu0, be0, M.BO_KAP0, M.BO_AL0, lbh, lb1, M.R_MAX)
    return np.array([bo.step(v) for v in np.asarray(x).tolist()], dtype=np.float64)

pyfunc = M._bocpd.py_func if hasattr(M._bocpd, "py_func") else None
print("numba active:", M._HAVE_NUMBA, "| py_func available:", pyfunc is not None)

report = {"series": SER, "n_hist": int(len(h)), "n_online": int(len(o)), "tau": int(tau)}
for label, x in (("ONLINE g", g), ("HIST ghh", ghh)):
    jit = M._bocpd(np.ascontiguousarray(x), mu0, be0, M.BO_KAP0, M.BO_AL0, lbh, lb1, M.R_MAX)
    stm = run_stream(x)
    py  = pyfunc(np.ascontiguousarray(x), mu0, be0, M.BO_KAP0, M.BO_AL0, lbh, lb1, M.R_MAX) if pyfunc else None
    def cmp(a, b, na, nb):
        d = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        n = int(d.sum())
        first = None
        if n:
            r, c = np.nonzero(d)
            first = dict(t=int(r[0]), col=int(c[0]), a=float(a[r[0], c[0]]), b=float(b[r[0], c[0]]),
                         absdiff=abs(float(a[r[0],c[0]])-float(b[r[0],c[0]])))
        print(f"  {label}: {na} vs {nb}: {n}/{a.size} cells differ" + (f"  first t={first['t']} col={first['col']} adiff={first['absdiff']:.3e}" if first else ""))
        return dict(n_diff=n, size=int(a.size), first=first)
    print(f"{label}: shape {jit.shape}")
    report[label] = {
        "jit_vs_stream": cmp(jit, stm, "jit_bocpd", "BocpdStream"),
    }
    if py is not None:
        report[label]["pyfunc_vs_stream"] = cmp(py, stm, "py_bocpd", "BocpdStream")
        report[label]["jit_vs_pyfunc"]    = cmp(jit, py,  "jit_bocpd", "py_bocpd")

json.dump(report, open(os.environ.get("OUT","/tmp")+f"/kernel_probe_{SER}.json","w"), indent=2, default=str)
