"""D2: historical-DGP fingerprint bank (history only -> legal at t=0)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import numpy as np
from sbr.store import load_store

st = load_store(f"{ROOT}/cache/store")
S = len(st.meta)

def rollmean(x, w):
    cx = np.concatenate([[0.0], np.cumsum(x)])
    return (cx[w:] - cx[:-w]) / w

def ar_coefs(z, p):
    if len(z) < 10*p+10: return np.zeros(p)
    X = np.column_stack([z[p-k-1:len(z)-k-1] for k in range(p)]); yv = z[p:]
    return np.linalg.solve(X.T@X + 1e-6*np.eye(p)*len(yv), X.T@yv)

def perm_entropy(z, m=3):
    n = len(z) - m + 1
    if n < 10: return np.nan
    W = np.column_stack([z[i:i+n] for i in range(m)])
    ranks = np.argsort(np.argsort(W, axis=1), axis=1)
    code = ranks[:,0]*9 + ranks[:,1]*3 + ranks[:,2]
    cnt = np.bincount(code, minlength=27).astype(float); cnt = cnt[cnt>0]/n
    return float(-(cnt*np.log(cnt)).sum()/np.log(6))

NAMES = ["n_hist","kurt","skew","hill","ar1","ar2","ar3","ar4","ar5","ar_sum",
         "acf1_sq","acf1_abs","vr10","vr50","spec_slope","perm_ent","turn_rate",
         "max_absz64","exc_max_run64","exc_n64","max_logvr128","q_ratio","zerocross"]
F = np.full((S, len(NAMES)), np.nan)
t0 = time.time()
for i in range(S):
    h = st.hist(i)
    n = len(h)
    z = (h - h.mean())/max(h.std(ddof=1),1e-9)
    d = z - z.mean()
    v = d @ d / n
    k = float(((d**4).mean())/ (v*v))
    sk = float(((d**3).mean())/ (v**1.5))
    a = np.sort(np.abs(z))[::-1]; m = max(int(0.025*n), 20)
    hill = float(m / np.log(a[:m]/a[m]).sum()) if a[m] > 0 else np.nan
    ar = ar_coefs(z, 5)
    zs = z*z; zs = zs - zs.mean()
    acf1sq = float((zs[1:]@zs[:-1])/(zs@zs))
    za = np.abs(z); za = za - za.mean()
    acf1abs = float((za[1:]@za[:-1])/(za@za))
    def vr(q):
        m2 = rollmean(z, q)*q
        return float(m2.var()/(q*z.var()))
    v10, v50 = vr(10), vr(50)
    # spectral slope over the lowest 25% of frequencies (Welch-free periodogram)
    nn = 1 << int(np.floor(np.log2(n)))
    P = np.abs(np.fft.rfft(z[:nn]))**2
    fq = np.arange(1, len(P))
    sel = fq <= max(len(fq)//4, 8)
    sl = np.polyfit(np.log(fq[sel]), np.log(np.maximum(P[1:][sel],1e-30)), 1)[0]
    pe = perm_entropy(z[:min(n,4000)])
    dz = np.diff(z); turn = float(np.mean(dz[1:]*dz[:-1] < 0))
    zc = float(np.mean(z[1:]*z[:-1] < 0))
    r64 = rollmean(z, 64); sd64 = r64.std()
    hot = np.abs(r64) > 2*sd64
    if hot.any():
        ch = np.diff(np.r_[0, hot.view(np.int8), 0])
        starts = np.flatnonzero(ch == 1); ends = np.flatnonzero(ch == -1)
        run = int((ends-starts).max()); nev = len(starts)
    else:
        run, nev = 0, 0
    maxz64 = float(np.abs(r64).max()/max(sd64,1e-12))
    lv = np.log(np.maximum(rollmean(z*z, 128), 1e-12))
    maxlvr = float(np.abs(lv - np.median(lv)).max())
    qs = np.quantile(z, [0.01,0.25,0.75,0.99])
    qr = float((qs[3]-qs[0])/max(qs[2]-qs[1],1e-9))
    F[i] = [n,k,sk,hill,ar[0],ar[1],ar[2],ar[3],ar[4],ar.sum(),acf1sq,acf1abs,
            v10,v50,sl,pe,turn,maxz64,run,nev,maxlvr,qr,zc]
    if i % 1000 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)
np.savez(f"{OUT}/d2_fingerprint.npz", F=F, names=np.array(NAMES))
print("done", time.time()-t0)
