"""Plan section 0.2 -- are the stored alt-partition OOF vectors stale?

Recompute the W4-E2 partition study from the .npy files on disk TODAY, using the
tracked analysis code, and diff against the committed record. Writes to the
scratchpad; never touches the tracked report.
"""
import hashlib, json, os, sys

ROOT = "/path/to/workspace/structural-break-rt1320-promotion-2026"
OUTDIR = os.path.dirname(os.path.abspath(__file__))
os.environ["SBR_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import wave4_partition_analyse as W

# Do NOT clobber the tracked record we are comparing against.
W.OUT = os.path.join(OUTDIR, "reverify_0_2_recomputed.json")

STREAMS = ["RT-300", "RT-401", "RT-402", "RT-403", "RT-404", "RT-405", "RT-406",
           "RT-410", "RT-411", "RT-412", "RT-413", "RT-414", "RT-415"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=== input provenance ===", flush=True)
prov = {}
for part in ("canonical", "alt1", "alt2", "alt3"):
    suf = "" if part == "canonical" else f".{part}"
    for s in STREAMS:
        p = os.path.realpath(f"{ROOT}/research/oof/{s}{suf}.npy")
        prov[f"{s}{suf}"] = {"path": p, "sha256": sha256(p),
                             "mtime": os.path.getmtime(p), "bytes": os.path.getsize(p)}
print(f"hashed {len(prov)} OOF vectors", flush=True)
json.dump(prov, open(os.path.join(OUTDIR, "reverify_0_2_inputs.json"), "w"), indent=2)

print("\n=== recomputing W4-E2 from vectors on disk ===", flush=True)
W.main()

# ---- diff against the committed record -------------------------------------
new = json.load(open(W.OUT))
old = json.load(open(f"{ROOT}/research/reports/ensemble_partition_stability.json"))

print("\n=== DIFF vs committed research/reports/ensemble_partition_stability.json ===")
print(f"{'partition':10s} {'arm':12s} {'committed':>12s} {'recomputed':>12s} {'delta':>12s}")
worst = 0.0
for part in ("canonical", "alt1", "alt2", "alt3"):
    o, n = old["partitions"][part], new["partitions"][part]
    if o.get("status") != "ok" or n.get("status") != "ok":
        print(f"{part:10s} STATUS old={o.get('status')} new={n.get('status')}")
        continue
    for arm in ("single", "seed_clone", "specialist"):
        a, b = o[arm]["mean"], n[arm]["mean"]
        worst = max(worst, abs(a - b))
        print(f"{part:10s} {arm:12s} {a:12.8f} {b:12.8f} {b-a:+12.2e}")
    for k in ("bagging", "specialisation", "total"):
        a, b = o["deltas"][k], n["deltas"][k]
        worst = max(worst, abs(a - b))
        print(f"{part:10s} d_{k:10s} {a:12.8f} {b:12.8f} {b-a:+12.2e}")

print(f"\nworst absolute discrepancy across all levels and deltas: {worst:.3e}")
verdict = ("VECTORS REPRODUCE THE COMMITTED RECORD" if worst < 1e-9
           else "BIT-LEVEL DRIFT (env/numerics)" if worst < 1e-6
           else "MATERIAL DISAGREEMENT -- VECTORS ARE STALE OR CHANGED")
print("VERDICT:", verdict)
json.dump({"worst_abs_discrepancy": worst, "verdict": verdict},
          open(os.path.join(OUTDIR, "reverify_0_2_verdict.json"), "w"), indent=2)
