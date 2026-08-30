import math, numpy as np, struct
from numba import njit

def bits(x): return struct.pack('>d', x).hex()

@njit(cache=True, fastmath=False)
def n_log(x): return math.log(x)
@njit(cache=True, fastmath=False)
def n_exp(x): return math.exp(x)
@njit(cache=True, fastmath=False)
def n_log1p(x): return math.log1p(x)
@njit(cache=True, fastmath=False)
def n_lgamma(x): return math.lgamma(x)
@njit(cache=True, fastmath=False)
def n_fma(a,b,c): return a*b + c
@njit(cache=True, fastmath=False)
def n_expr(d, nu, s2, ct, hnu):
    return ct - 0.5*math.log(s2) - hnu*math.log1p(d*d/(nu*s2))

rng = np.random.default_rng(0)
N = 200000
tests = {
 "log":   (n_log,   math.log,   np.abs(rng.standard_normal(N))*10 + 1e-6),
 "exp":   (n_exp,   math.exp,   rng.uniform(-50, 5, N)),
 "log1p": (n_log1p, math.log1p, np.abs(rng.standard_normal(N))*3),
 "lgamma":(n_lgamma,math.lgamma,np.abs(rng.standard_normal(N))*40 + 0.5),
}
for name,(nf, pf, xs) in tests.items():
    diff = 0; ex = None
    for x in xs.tolist():
        a = nf(x); b = pf(x)
        if a != b:
            diff += 1
            if ex is None: ex = (x, a, b)
    print(f"{name:7s}: {diff}/{N} differ" + (f"   e.g. x={ex[0]!r} numba={bits(ex[1])} cpython={bits(ex[2])} adiff={abs(ex[1]-ex[2]):.3e}" if ex else ""))

# FMA contraction probe
diff = 0; ex=None
A=rng.standard_normal(N); B=rng.standard_normal(N); C=rng.standard_normal(N)
for a,b,c in zip(A.tolist(),B.tolist(),C.tolist()):
    x = n_fma(a,b,c); y = a*b+c
    if x != y:
        diff += 1
        if ex is None: ex=(a,b,c,x,y)
print(f"a*b+c  : {diff}/{N} differ" + (f"   adiff={abs(ex[3]-ex[4]):.3e}" if ex else ""))
