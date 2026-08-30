"""tau_hat quality diagnostics for m06_loc (LOCALISATION CHALLENGER).

Reads the built screen-store feature cache and the store labels and reports:
  * distribution of (tau_hat - tau) on BREAK series at fixed elapsed horizons
  * how often the localiser is inside +-20% / +-50% of the true elapsed
  * the spurious-confidence rate on NO-BREAK series (q >= 2 == p < 0.01)
Descriptive only -- no model, no split, nothing appended to RESULTS.csv.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "/home/claude/sb/src")
from sbr.store import load_store  # noqa: E402

FEAT = "/home/claude/sb/cache/features_screen/m06_loc"
CHANNELS = ["h_rsq", "h_rlab", "h_rl1", "p_rsq", "p_rlab", "p_rl1"]
HORIZONS = (20, 40, 80, 160, 320)


def main():
    st = load_store("/home/claude/sb/cache/store_screen")
    cols = json.load(open(FEAT + ".cols.json"))["cols"]
    A = np.load(FEAT + ".npy", mmap_mode="r")
    ci = {c: i for i, c in enumerate(cols)}
    off = st.orow_off
    n_on = st.meta.n_online.to_numpy()
    tau = st.meta.tau_index.to_numpy()

    print(f"# tau_hat quality — screen store, {len(n_on)} series "
          f"({int((tau >= 0).sum())} break / {int((tau < 0).sum())} no-break)\n")

    # ---------------- break series: (tau_hat - tau) at fixed elapsed ----------
    print("## A. Localisation error (tau_hat - tau) on BREAK series, at elapsed h since tau")
    print("`m_hat` = estimated elapsed = 2**lm ; tau_hat = t+1-m_hat ; error = tau_hat - tau.")
    print("`|err|<=.2h` = fraction whose estimated elapsed is within +-20% of the true elapsed.\n")
    hdr = "| channel | h | n | med err | IQR err | med |err| | |err|<=.2h | |err|<=.5h | med m_hat | true h |"
    print(hdr)
    print("|" + "---|" * 10)
    rows = {}
    for ch in CHANNELS:
        jlm = ci[f"loc_{ch}_lm"]
        for h in HORIZONS:
            errs, mh = [], []
            for i in np.flatnonzero(tau >= 0):
                t = int(tau[i]) + h
                if t >= int(n_on[i]):
                    continue
                lm = A[int(off[i]) + t, jlm]
                if not np.isfinite(lm):
                    continue
                m = 2.0 ** lm
                errs.append((t + 1 - m) - tau[i])
                mh.append(m)
            if len(errs) < 30:
                continue
            e = np.asarray(errs, float)
            mh = np.asarray(mh, float)
            q1, q3 = np.percentile(e, [25, 75])
            rows[(ch, h)] = (len(e), np.median(e), q3 - q1, np.median(np.abs(e)),
                             float(np.mean(np.abs(e) <= 0.2 * h)),
                             float(np.mean(np.abs(e) <= 0.5 * h)), np.median(mh))
            r = rows[(ch, h)]
            print(f"| {ch} | {h} | {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | "
                  f"{r[4]:.3f} | {r[5]:.3f} | {r[6]:.0f} | {h} |")

    # a random-grid baseline: what would |err| be if m_hat were drawn from the
    # grid uniformly at random (i.e. no localisation information at all)?
    print("\n### baseline: m_hat drawn uniformly from the available grid (no information)")
    from sbr.features.m06_loc import M_H
    rng = np.random.default_rng(0)
    print("| h | n | med |err| (random grid) | |err|<=.2h | |err|<=.5h |")
    print("|---|---|---|---|---|")
    for h in HORIZONS:
        errs = []
        for i in np.flatnonzero(tau >= 0):
            t = int(tau[i]) + h
            if t >= int(n_on[i]):
                continue
            avail = M_H[M_H <= t + 1]
            if len(avail) == 0:
                continue
            m = float(rng.choice(avail))
            errs.append((t + 1 - m) - tau[i])
        if len(errs) < 30:
            continue
        e = np.abs(np.asarray(errs, float))
        print(f"| {h} | {len(e)} | {np.median(e):.1f} | {np.mean(e <= 0.2*h):.3f} | {np.mean(e <= 0.5*h):.3f} |")

    # ---------------- confidence: q on break vs no-break ---------------------
    print("\n## B. Is the localiser CONFIDENT when it should be?")
    print("`q` = -log10 upper-tail p of the maximised score vs the historical null of the "
          "max over the same number of candidates (capped at 3.0). q>=2 means p<0.01.\n")
    print("| channel | h | n_brk | mean q (break) | P(q>=2) break | n_nb | mean q (no-break) | "
          "P(q>=2) no-break | AUC(q) |")
    print("|" + "---|" * 9)
    # no-break series get a matched evaluation index: the same absolute t
    # distribution as the break cohort at that horizon (draw t uniformly from
    # the same empirical distribution).
    rng = np.random.default_rng(20260818)
    for ch in CHANNELS:
        jq = ci[f"loc_{ch}_q"]
        for h in HORIZONS:
            qb, qn = [], []
            ts = []
            for i in np.flatnonzero(tau >= 0):
                t = int(tau[i]) + h
                if t >= int(n_on[i]):
                    continue
                v = A[int(off[i]) + t, jq]
                if np.isfinite(v):
                    qb.append(v)
                    ts.append(t)
            if len(ts) < 30:
                continue
            ts = np.asarray(ts)
            for i in np.flatnonzero(tau < 0):
                t = int(rng.choice(ts))
                if t >= int(n_on[i]):
                    continue
                v = A[int(off[i]) + t, jq]
                if np.isfinite(v):
                    qn.append(v)
            if len(qn) < 30:
                continue
            qb = np.asarray(qb); qn = np.asarray(qn)
            allv = np.concatenate([qb, qn])
            r = np.argsort(np.argsort(allv)) + 1.0
            n1 = len(qb)
            auc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(qn))
            print(f"| {ch} | {h} | {n1} | {qb.mean():.3f} | {np.mean(qb >= 2):.3f} | {len(qn)} | "
                  f"{qn.mean():.3f} | {np.mean(qn >= 2):.3f} | {auc:.4f} |")

    # ---------------- spurious confidence on no-break series -----------------
    print("\n## C. Spurious confidence on NO-BREAK series (whole online segment)")
    print("Fraction of no-break ROWS with q>=2, and fraction of no-break SERIES that ever fire.\n")
    print("| channel | frac rows q>=2 (no-break) | frac rows q>=2 (break, post-tau) | "
          "frac series ever q>=2 (no-break) | (break) |")
    print("|" + "---|" * 5)
    nb = np.flatnonzero(tau < 0)
    br = np.flatnonzero(tau >= 0)
    for ch in CHANNELS:
        jq = ci[f"loc_{ch}_q"]
        def stats(idx, post_only):
            rows_hot = 0; rows_tot = 0; ser_hot = 0; ser_tot = 0
            for i in idx:
                s = int(off[i]); n = int(n_on[i])
                lo = int(tau[i]) if post_only else 0
                v = np.asarray(A[s + lo:s + n, jq], dtype=np.float64)
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                rows_hot += int((v >= 2).sum()); rows_tot += len(v)
                ser_tot += 1; ser_hot += int((v >= 2).any())
            return rows_hot / max(rows_tot, 1), ser_hot / max(ser_tot, 1)
        rn, sn = stats(nb, False)
        rb, sb = stats(br, True)
        print(f"| {ch} | {rn:.4f} | {rb:.4f} | {sn:.3f} | {sb:.3f} |")


if __name__ == "__main__":
    main()
