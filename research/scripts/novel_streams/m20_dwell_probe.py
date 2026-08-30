"""Reference StreamingMechanism -- the Pilot 1 skeleton, VERIFIED PREFIX-INVARIANT.

This is a three-column probe, not the pilot. Pilot 1 (research/NEW_AVENUES_2026.md
section L) adds the matched-length run NULL (IM2), the contiguity ratio, the growth
exponent, and the second channel/window grid. Start from this file so the causal
contract is satisfied before a single number exists.

    export SBR_ROOT=<worktree>; export PYTHONPATH=$SBR_ROOT/research/scripts
    python research/scripts/novel_streams/m20_dwell_probe.py

WHY THE NaN GUARD IS WRITTEN THIS WAY (it is not cosmetic).
The first version of this file initialised `run` and `mass` to ZERO and returned an
all-NaN block when n_online < W. `harness.verify()` rejected it on the first call
(series 7588, prefix 3): the full build emitted 0.0 for rows < W-1 while the truncated
build emitted NaN for the same rows. Not a look-ahead, but exactly the class of
inconsistency that makes a batch/stream parity test fail later. Rows before the window
has filled must be NaN -- never zero, never back-filled from history (PROTOCOL.md section 3).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import StreamingMechanism, verify, build, Ctx, rt600_blend, diagnostic_pack, pair_flow_by_cell
from sbr.transforms import HistParams, ar_filter_causal, _ar_resid


class DwellProbe(StreamingMechanism):
    """Contiguous excursion dwell of the AR(2)-residual scale, outside the per-series
    historical q90 band. SIGNAL: after a persistent break the excursion never terminates,
    so the run grows with elapsed time. FALSE SIGNAL: a heavy-tailed stable series throws
    long-but-finite excursions (measured: 27.8% of never-break series reach length 50).
    DISAMBIGUATOR: the running max normalised by its matched-length historical run null
    (IM2, not in this probe) and the growth exponent d log(maxrun)/d log t."""

    name = "m20_dwell_probe"
    cols = ["res64_run90", "res64_maxrun90", "res64_mass90"]
    W = 64
    Q = 0.90

    def fit_history(self, hist):
        return HistParams(np.asarray(hist, dtype=np.float64), ar_order=2)

    def emit(self, hist, online):
        hp = self.fit_history(hist)
        n, W = len(online), self.W
        out = np.full((n, len(self.cols)), np.nan, dtype=np.float32)
        zh = (np.asarray(hist, np.float64) - hp.mu) / hp.sd
        zo = (np.asarray(online, np.float64) - hp.mu) / hp.sd
        eh = np.concatenate([np.zeros(2), _ar_resid(zh, hp.ar_coef)]) / hp.ar_sigma
        eo = ar_filter_causal(zo, hp.ar_coef, zh) / hp.ar_sigma
        sh, so = eh * eh, eo * eo
        if len(sh) < 2 * W or n < W:
            return out
        c = np.concatenate([[0.0], np.cumsum(sh)])
        nul = (c[W:] - c[:-W]) / W                      # history-only null band
        med = float(np.median(nul))
        q = float(np.quantile(np.abs(nul - med), self.Q))
        co = np.concatenate([[0.0], np.cumsum(so)])
        on = np.full(n, np.nan); on[W - 1:] = (co[W:] - co[:-W]) / W
        dev = np.abs(on - med)
        hot = np.zeros(n, bool); hot[W - 1:] = dev[W - 1:] > q
        run = np.full(n, np.nan); mass = np.full(n, np.nan)
        a, m = 0, 0.0
        for k in range(W - 1, n):
            if hot[k]:
                a += 1; m += dev[k] - q
            else:
                a = 0; m = 0.0
            run[k] = a; mass[k] = m
        out[:, 0] = run
        out[W - 1:, 1] = np.maximum.accumulate(run[W - 1:])
        out[:, 2] = mass
        return out


if __name__ == "__main__":
    m = DwellProbe()
    ok, msg = verify(m)
    print(f"prefix invariance (atol=0.0): {ok}  {msg}")
    if not ok:
        raise SystemExit(1)
    A, cols = build(m)
    c = Ctx(); base = rt600_blend(c)
    for j, nm in enumerate(cols):
        cand = np.nan_to_num(np.asarray(A[:, j], np.float64), nan=0.0)
        p = diagnostic_pack(c, cand, base, label=nm)
        print(f"{nm:>18s}  cell {p['dominant_cell']['candidate']:.5f} "
              f"(rt600 {p['dominant_cell']['rt600']:.5f})  "
              f"vs-neverbreak {p['mature_vs_neverbreak']['candidate']:.5f}  "
              f"rho {p['within_t_rank_corr_rt600']:+.3f}")
