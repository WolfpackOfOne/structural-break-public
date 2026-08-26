"""Static audit for the CAUSAL REPRESENTATION FRONTIER program.

AUDIT ONLY.  This script trains nothing, scores nothing, allocates no ``RT-xxx``
id, touches no lockbox row and writes no row to ``research/RESULTS.csv``.  It
reads committed artifacts and re-derives the descriptive facts the program's
design rests on, so that a reviewer can check the design against the repository
rather than against prose.

What it checks
--------------
1. ``research/RESULTS.csv`` is APPEND-ONLY with respect to the CRF design task:
   its leading bytes are still byte-identical to the ledger as the design task
   found it, and no ``CRF-`` experiment id has appeared in it.  Rows appended
   afterwards -- including by the CRF execution -- are legitimate and reported.
2. The two audit CSVs parse, are non-empty, and every proposed CRF candidate row
   carries a ``closest_prior`` and a ``collision_severity``.
3. Collision check: every proposed CRF candidate differs from its closest prior on
   at least one *load-bearing* axis (representation or objective), read from
   ``representation_collision_matrix.csv`` rather than asserted.
4. The descriptive signal/redundancy frontier over the prior scored arms, which is
   the quantitative basis for the program's cheap abandon gate.

Nothing here is a candidate performance metric: item 4 recomputes properties of
*already recorded* experiments.

Usage
-----
    python research/scripts/audit_causal_representation_frontier.py
    python research/scripts/audit_causal_representation_frontier.py --json out.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

REPORT_DIR = os.path.join(ROOT, "research", "reports", "causal_representation_frontier")
RESULTS_CSV = os.path.join(ROOT, "research", "RESULTS.csv")

#: sha256 of research/RESULTS.csv at b47b22ad (the exhausted Second Sweep tip),
#: i.e. the ledger as it stood when the CRF DESIGN TASK began.  The design task
#: had to leave it untouched, and did: the file is byte-identical at b219baf
#: (audit), 85d121f (prereg) and afba958 (CRF-01 prereg).
#:
#: WHAT THIS PIN IS AND IS NOT.  It is an assertion about the design task, not
#: about the repository forever.  The CRF *execution* that followed legitimately
#: appended its own experiment rows -- b6f27e3 (CRF-01), 9a5ecc0 and 9c5a352
#: (CRF-02, the second superseding the voided first) -- taking the whole-file
#: hash to 8a0a8832...  Comparing that against this constant was asserting that
#: current HEAD must forever equal a pre-execution hash, which is not the
#: invariant anyone intended and turned this check permanently red.
#:
#: The invariant is instead APPEND-ONLY: the first BASE_RESULTS_BYTES bytes of
#: RESULTS.csv must still be exactly the design-task-era ledger.  That is
#: strictly stronger than the old whole-file check about the thing it was
#: protecting -- it detects any edit to a pre-CRF row -- while permitting the
#: appends the execution was entitled to make.  No historical row and no
#: scientific record is rewritten to satisfy it.
BASE_RESULTS_SHA256 = (
    "5b34c564e69f502c4a54d4ba1b702b400893358073e1897cd82453c83215512c")

#: byte length of research/RESULTS.csv at b47b22ad (246 lines: header + 245 rows).
BASE_RESULTS_BYTES = 155924

#: the design-task commits this pin is a claim about, newest last.
DESIGN_TASK_COMMITS = ("b219baf", "85d121f")

#: Axes on which a proposed candidate must differ from its closest prior for the
#: difference to count as load-bearing rather than cosmetic.
LOAD_BEARING_AXES = (
    "uses_pit",
    "uses_innovations",
    "uses_multistep_residuals",
    "uses_500_features",
    "uses_rt600",
    "learned_temporal_representation",
    "self_supervised",
    "same_t_pairwise_objective",
    "conditional_density_model",
    "per_series_history_conditioning",
)

#: (id, standalone whole-fold TS-AUC, within-t rank corr vs the RT600-family
#: blend, marginal_vs_clone) for every scored arm where all three are recorded.
#: Transcribed from first_sweep_matrix.csv, wave6_neural_analysis*.json and
#: wave7_t2_promotion_final.md.  Descriptive; no new number is produced.
PRIOR_ARMS = (
    ("RT-1200", 0.61961, 0.7746, -0.000207),
    ("RT-1201", 0.58358, 0.3817, +0.000301),
    ("RT-1202", 0.49932, 0.0039, -0.002587),
    ("RT-1204", 0.62944, 0.8860, -0.000236),
    ("RT-1206", 0.62764, 0.8809, -0.000353),
    ("RT-1208", 0.62914, 0.8780, -0.000036),
    ("RT-1210", 0.62612, 0.8779, -0.000522),
    ("RT-1212", 0.62913, 0.8629, +0.000164),
    ("RT-1214", 0.63082, 0.8836, +0.000135),
    ("RT-1215", 0.63133, 0.8859, +0.000226),
    ("RT-1216", 0.63428, 0.8653, +0.000937),
    ("RT-1218", 0.52568, 0.1341, -0.002214),
    ("RT-970", 0.54152, 0.2362, +0.000099),
    ("RT-971", 0.54319, 0.2406, +0.000195),
    ("RT-960", 0.57059, 0.3984, -0.000598),
    ("RT-961", 0.58061, 0.4459, +0.000134),
    ("RT-995", 0.62128, 0.6740, +0.000240),
)


def sha256_of(path: str) -> str:
    """Return the hex sha256 of a file.

    Parameters
    ----------
    path : str
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_results_untouched() -> dict:
    """Assert the design-task-era ledger is intact and carries no CRF row.

    Verifies the append-only invariant described at :data:`BASE_RESULTS_SHA256`:
    the leading ``BASE_RESULTS_BYTES`` bytes of ``research/RESULTS.csv`` still
    hash to the design-task base, so no pre-CRF row has been edited, reordered
    or removed.  Rows appended after the design task -- by the CRF execution or
    by anything later -- are permitted and are reported, not failed.

    Returns
    -------
    dict
        Design-task prefix hash and verdict, current whole-file hash, row
        counts and CRF-row count.
    """
    with open(RESULTS_CSV, "rb") as fh:
        head = fh.read(BASE_RESULTS_BYTES)
    prefix_sha = hashlib.sha256(head).hexdigest()
    observed = sha256_of(RESULTS_CSV)
    size = os.path.getsize(RESULTS_CSV)
    with open(RESULTS_CSV, newline="") as fh:
        ids = [row["experiment_id"] for row in csv.DictReader(fh)]
    crf_rows = [i for i in ids if i.upper().startswith("CRF")]
    base_rows = head.count(b"\n") - 1          # minus the header line
    return {
        "path": os.path.relpath(RESULTS_CSV, ROOT),
        "invariant": "append_only_since_crf_design_task",
        "design_task_commits": list(DESIGN_TASK_COMMITS),
        "expected_prefix_sha256": BASE_RESULTS_SHA256,
        "observed_prefix_sha256": prefix_sha,
        "design_task_prefix_intact": (size >= BASE_RESULTS_BYTES
                                      and prefix_sha == BASE_RESULTS_SHA256),
        "base_bytes": BASE_RESULTS_BYTES,
        "current_bytes": size,
        "current_sha256": observed,
        "n_rows_at_design_task": base_rows,
        "n_rows": len(ids),
        "n_rows_appended_since": len(ids) - base_rows,
        "crf_rows": crf_rows,
        "no_crf_rows": not crf_rows,
    }


def load_csv(name: str) -> list:
    """Read one audit CSV into a list of dicts.

    Parameters
    ----------
    name : str
        File name inside the report directory.

    Returns
    -------
    list
        One dict per data row.
    """
    with open(os.path.join(REPORT_DIR, name), newline="") as fh:
        return list(csv.DictReader(fh))


def check_collisions() -> dict:
    """Verify every proposed CRF candidate differs load-bearingly from its prior.

    Returns
    -------
    dict
        Per-candidate differing axes, closest prior and pass flag.
    """
    matrix = load_csv("representation_collision_matrix.csv")
    by_name = {r["mechanism"]: r for r in matrix}
    proposed = [r for r in matrix if r["status"].startswith("PROPOSED")]

    def prior_row(closest: str):
        """Best-effort lookup of a closest-prior row by name prefix."""
        key = closest.split("(")[0].strip()
        for name, row in by_name.items():
            if name.startswith(key) or key.split("/")[0].strip() in name:
                return name, row
        return None, None

    out, ok = {}, True
    for row in proposed:
        name = row["mechanism"]
        pname, prior = prior_row(row["closest_prior"])
        if prior is None:
            out[name] = {"closest_prior_row_found": False,
                         "closest_prior": row["closest_prior"]}
            # A conditional integration slot legitimately has no single prior row.
            if "conditional" not in row["status"]:
                ok = False
            continue
        differing = [
            ax for ax in LOAD_BEARING_AXES
            if row.get(ax, "").strip().lower() != prior.get(ax, "").strip().lower()
        ]
        passed = bool(differing)
        out[name] = {
            "closest_prior_row_found": True,
            "closest_prior_row": pname,
            "differing_load_bearing_axes": differing,
            "n_differing": len(differing),
            "passes_collision_check": passed,
            "collision_severity": row["collision_severity"],
        }
        ok = ok and (passed or "conditional" in row["status"])
    return {"per_candidate": out, "all_pass": ok}


def frontier() -> dict:
    """Recompute the descriptive signal/redundancy frontier over prior arms.

    Returns
    -------
    dict
        Correlations, an OLS fit of marginal on (standalone, rho), the implied
        standalone needed for +0.0030 at several rho, and the observed best
        standalone at low redundancy.
    """
    import numpy as np

    a = np.array([[r[1], r[2], r[3]] for r in PRIOR_ARMS], dtype=float)
    s, r, m = a[:, 0], a[:, 1], a[:, 2]
    X = np.c_[np.ones(len(s)), s, r]
    beta, *_ = np.linalg.lstsq(X, m, rcond=None)
    pred = X @ beta
    r2 = 1.0 - ((m - pred) ** 2).sum() / ((m - m.mean()) ** 2).sum()
    contour = {
        f"rho={rho:.1f}": round(float((0.0030 - beta[0] - beta[2] * rho) / beta[1]), 4)
        for rho in (0.2, 0.3, 0.4, 0.5, 0.6)
    }
    low = [x for x in PRIOR_ARMS if x[2] <= 0.60]
    return {
        "n_arms": len(PRIOR_ARMS),
        "corr_standalone_rho": round(float(np.corrcoef(s, r)[0, 1]), 4),
        "corr_standalone_marginal": round(float(np.corrcoef(s, m)[0, 1]), 4),
        "corr_rho_marginal": round(float(np.corrcoef(r, m)[0, 1]), 4),
        "ols_coef_const_standalone_rho": [round(float(b), 6) for b in beta],
        "ols_r2": round(float(r2), 4),
        "standalone_needed_for_plus_0.0030": contour,
        "best_standalone_at_rho_le_0.60": max(x[1] for x in low),
        "best_arm_at_rho_le_0.60": max(low, key=lambda x: x[1])[0],
        "rt600_dev_mean_ts_auc": 0.625811,
        "caveat": ("n=17, R2 moderate, rho baselines heterogeneous across waves, "
                   "and the +0.0030 contour is an extrapolation outside the "
                   "observed hull. Quoted as a scale, not a prediction."),
    }


def main() -> int:
    """Run every audit check and print a report.

    Returns
    -------
    int
        0 if all checks pass, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="write the full report to this path")
    args = ap.parse_args()

    report = {
        "results_safety": check_results_untouched(),
        "audit_files": {},
        "collision": check_collisions(),
        "frontier": frontier(),
    }
    for name in ("prior_representation_audit.csv",
                 "representation_collision_matrix.csv",
                 "crf_candidate_priority.csv"):
        rows = load_csv(name)
        report["audit_files"][name] = {"n_rows": len(rows), "non_empty": bool(rows)}

    rs = report["results_safety"]
    print("== RESULTS.csv safety (append-only since the CRF design task) ==")
    print(f"  design-task prefix intact : {rs['design_task_prefix_intact']}  "
          f"({rs['observed_prefix_sha256'][:16]}..., {rs['base_bytes']} B)")
    print(f"  current whole-file sha256 : {rs['current_sha256'][:16]}...")
    print(f"  rows                      : {rs['n_rows']} "
          f"({rs['n_rows_at_design_task']} at the design task, "
          f"+{rs['n_rows_appended_since']} appended since)")
    print(f"  no CRF rows               : {rs['no_crf_rows']}")

    print("\n== audit files ==")
    for name, info in report["audit_files"].items():
        print(f"  {name:42s} {info['n_rows']:3d} rows")

    print("\n== collision check (proposed vs closest prior) ==")
    for name, info in report["collision"]["per_candidate"].items():
        if not info.get("closest_prior_row_found"):
            print(f"  {name:34s} closest prior not a matrix row "
                  f"({info['closest_prior'][:40]})")
            continue
        print(f"  {name:34s} differs on {info['n_differing']} load-bearing axes "
              f"vs {info['closest_prior_row'][:24]} -> "
              f"{'PASS' if info['passes_collision_check'] else 'FAIL'}")
        if info["differing_load_bearing_axes"]:
            print(f"      {', '.join(info['differing_load_bearing_axes'])}")

    f = report["frontier"]
    print("\n== descriptive signal/redundancy frontier (prior scored arms) ==")
    print(f"  corr(standalone, rho)          : {f['corr_standalone_rho']}")
    print(f"  OLS R2                         : {f['ols_r2']}  (n={f['n_arms']})")
    print("  standalone needed for +0.0030  :")
    for k, v in f["standalone_needed_for_plus_0.0030"].items():
        print(f"      {k} -> {v}")
    print(f"  best standalone at rho<=0.60   : {f['best_standalone_at_rho_le_0.60']} "
          f"({f['best_arm_at_rho_le_0.60']})")

    ok = (rs["design_task_prefix_intact"] and rs["no_crf_rows"]
          and report["collision"]["all_pass"]
          and all(v["non_empty"] for v in report["audit_files"].values()))
    print(f"\nAUDIT: {'PASS' if ok else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
