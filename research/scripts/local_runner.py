"""A faithful local emulation of the Crunch Real-Time streaming contract.

The official `crunch test` needs api.hub.crunchdao.com, which this container
cannot reach.  This harness enforces the same contract locally, and enforces it
HARDER than the official runner does, so that anything passing here has a very
good chance of passing there:

  * `datasets` is a one-shot iterator -- re-iterating raises
  * each series' `x_online` is a one-shot iterator -- re-iterating raises
  * `x_historical` is handed over as a read-only array; writing to it raises
  * infer() must yield the readiness sentinel first, then exactly one score per
    online observation, in order
  * every score must be a finite float in [0, 1]
  * determinism: the whole run is executed twice and compared exactly
  * parallelism-independence: the run is repeated with the series order shuffled
    and each series' scores must be unchanged (this is what kills any illegal
    cross-series state, which is exactly how the organisers describe it)
  * future-poisoning: a series is replayed with its tail rewritten; the scores
    before the rewrite point must be unchanged

Scoring uses the project's official-metric implementation.
"""
from __future__ import annotations

import time
import numpy as np

from sbr.metric import ts_auc_flat


class OneShot:
    """An iterator that refuses to be iterated twice."""

    def __init__(self, it, what="iterable"):
        self._it = iter(it)
        self._used = False
        self._what = what

    def __iter__(self):
        if self._used:
            raise RuntimeError(f"{self._what} was iterated twice -- illegal under the Crunch contract")
        self._used = True
        return self._it


def _readonly(a):
    b = np.array(a, dtype=np.float64)
    b.setflags(write=False)
    return b


def make_datasets(series):
    """series: list of (hist, online). Returns a legal one-shot `datasets`."""
    def gen():
        for h, o in series:
            yield _readonly(h), OneShot(list(map(float, o)), "x_online")
    return OneShot(gen(), "datasets")


def run_infer(infer_fn, series, model_directory_path):
    """Execute infer() against the contract. Returns list of per-series score arrays."""
    g = infer_fn(make_datasets(series), model_directory_path)
    first = next(g)
    if first is not None:
        raise AssertionError("infer() must yield the readiness sentinel (a bare `yield`) first")
    out = []
    for h, o in series:
        s = np.empty(len(o), dtype=np.float64)
        for t in range(len(o)):
            v = next(g)
            fv = float(v)
            if not np.isfinite(fv):
                raise AssertionError(f"non-finite score at series-relative t={t}: {v!r}")
            if fv < 0.0 or fv > 1.0:
                raise AssertionError(f"score out of [0,1] at t={t}: {fv!r}")
            s[t] = fv
        out.append(s)
    extra = list(g)
    if extra:
        raise AssertionError(f"infer() yielded {len(extra)} scores too many")
    return out


def check_all(infer_fn, series, model_directory_path, labels=None, verbose=True):
    """Run every legality check. Returns a dict of results."""
    res = {}
    t0 = time.time()
    a = run_infer(infer_fn, series, model_directory_path)
    res["runtime_s"] = time.time() - t0
    res["n_series"] = len(series)
    res["n_points"] = int(sum(len(s) for s in a))
    res["ms_per_point"] = 1000.0 * res["runtime_s"] / max(res["n_points"], 1)

    b = run_infer(infer_fn, series, model_directory_path)
    res["deterministic"] = all(np.array_equal(x, y) for x, y in zip(a, b))

    order = np.random.default_rng(0).permutation(len(series))
    shuffled = [series[i] for i in order]
    c = run_infer(infer_fn, shuffled, model_directory_path)
    res["order_independent"] = all(np.array_equal(a[i], c[k]) for k, i in enumerate(order))

    # future poisoning on the longest series
    j = int(np.argmax([len(o) for _, o in series]))
    h, o = series[j]
    cut = max(5, len(o) // 3)
    o2 = np.array(o, dtype=np.float64).copy()
    o2[cut:] = np.random.default_rng(1).standard_normal(len(o) - cut) * 1e6
    d = run_infer(infer_fn, [(h, o2)], model_directory_path)[0]
    res["future_poison_safe"] = bool(np.array_equal(a[j][:cut], d[:cut]))

    res["all_finite"] = bool(all(np.isfinite(s).all() for s in a))
    res["in_range"] = bool(all(((s >= 0) & (s <= 1)).all() for s in a))

    if labels is not None:
        sc = np.concatenate(a)
        la = np.concatenate([np.asarray(l, dtype=np.int8) for l in labels])
        ti = np.concatenate([np.arange(len(s), dtype=np.int64) for s in a])
        # NOT an out-of-sample estimate: the deployed model is fitted on ALL dev
        # folds, so any dev series scored here is IN-SAMPLE. It is also computed
        # on a handful of series, where TS-AUC has sd ~0.06 at n=60 and ~0.14 at
        # n=25. It exists only to catch a catastrophically broken pipeline
        # (a score near 0.5 or below). Never quote it as performance.
        res["ts_auc_diagnostic_in_sample_do_not_quote"] = float(ts_auc_flat(sc, la, ti))

    res["PASS"] = bool(res["deterministic"] and res["order_independent"]
                       and res["future_poison_safe"] and res["all_finite"] and res["in_range"])
    if verbose:
        import json
        print(json.dumps(res, indent=2))
    return res
