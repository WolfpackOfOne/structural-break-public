"""Development-only prediction equivalence between two deployment artifacts.

Reports counts and deltas only.  NO TS-AUC IS COMPUTED, for either side: the
comparison is between two builds of ONE frozen model, not a selection between
candidates, and computing a score would make it one.
"""
import json, sys
import numpy as np

a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
assert (a["ids"] == b["ids"]).all(), "different series sampled"
assert (a["lens"] == b["lens"]).all(), "different online lengths"
sa, sb = a["scores"], b["scores"]
ids, lens = a["ids"], a["lens"]
d = np.abs(sa - sb)
changed = int((sa != sb).sum())

# same-t cross-series pair ordering: the structure TS-AUC scores.
off = np.concatenate([[0], np.cumsum(lens)])
t = np.concatenate([np.arange(n) for n in lens])
ser = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
flips = pairs = 0
for tv in np.unique(t):
    m = t == tv
    if m.sum() < 2:
        continue
    xa, xb = sa[m], sb[m]
    i, j = np.triu_indices(m.sum(), 1)
    ca = np.sign(xa[i] - xa[j])
    cb = np.sign(xb[i] - xb[j])
    pairs += len(i)
    flips += int((ca != cb).sum())

per_series = []
for k, (i0, i1) in enumerate(zip(off[:-1], off[1:])):
    c = int((sa[i0:i1] != sb[i0:i1]).sum())
    if c:
        per_series.append({"series": int(ids[k]), "n_changed": c,
                           "max_abs_delta": float(d[i0:i1].max())})

out = {
    "note": "no TS-AUC computed for either artifact",
    "artifact_a": sys.argv[1], "artifact_b": sys.argv[2],
    "n_series": int(len(ids)),
    "n_predictions_compared": int(sa.size),
    "n_predictions_changed": changed,
    "pct_changed": 100.0 * changed / sa.size,
    "max_abs_prediction_delta": float(d.max()),
    "mean_abs_prediction_delta": float(d.mean()),
    "series_with_any_change": len(per_series),
    "per_series_changed": per_series,
    "same_t_pairs_compared": int(pairs),
    "same_t_pair_order_flips": int(flips),
}
print(json.dumps(out, indent=2))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=2)
