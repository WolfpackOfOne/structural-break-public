"""Phase 1 §1.5 — fold-0 diagnosis across partitions.

The plan (`research/RT1320_PROMOTION_PLAN.md` §1.5) states:

> Fold 0 is negative on both endpoints on canonical. Establish whether that is
> partition-specific noise or a property of the mechanism. If fold 0 is negative
> on multiple partitions, the mechanism has a regime where it does damage, and
> that must be characterised — by online-horizon bucket and by never-break/
> pre-break cut — before promotion is arguable regardless of the mean.

So there are two questions, and the second is conditional on the first:

  Q1  Is fold 0 negative on more than one partition?
  Q2  If so, where does the damage live — which horizon, which negative class?

This **runs alongside and does not gate** (plan §1.5). It reports; it does not
decide. The PASS/KILL/INCONCLUSIVE verdict belongs to §4 and to a human.

Reconstruction is identical to `armc_e2_e1_addition_contract.py` — same loaders,
same calibration, same blend — so the fold-0 numbers here are the same quantity
as the headline endpoint, just restricted to row subsets via the `mask` argument
that `mean_fold_ts_auc` already accepts. Nothing new is fitted.

Usage:
    python research/scripts/rt1320_fold0_diagnosis.py --partitions canonical alt1
    python research/scripts/rt1320_fold0_diagnosis.py          # all four
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import armc_residual_student as A  # noqa: E402

CONTROL_STREAM = "RT-403"
DEFAULT_STUDENT_DIR = REPO / "research" / "reports" / "armc_residual_student_confirm_s20260901"

# Fixed before looking, and anchored on W7-D0's exact-loss cube: the dominant
# remaining-loss region is t >= 200 with post-break age >= 100. The t=200
# boundary is therefore a real edge in this problem, not an arbitrary cut.
HORIZON_BUCKETS = [
    ("t<50", 0, 50),
    ("50<=t<100", 50, 100),
    ("100<=t<200", 100, 200),
    ("200<=t<400", 200, 400),
    ("400<=t<800", 400, 800),
    ("t>=800", 800, 1 << 30),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--partitions", nargs="+",
                   default=["canonical", "alt1", "alt2", "alt3"])
    p.add_argument("--student-dir", type=Path, default=None,
                   help="override; by default each partition uses its own "
                        "research/reports/armc_residual_student[.altK]")
    p.add_argument("--out", type=Path,
                   default=REPO / "research" / "reports" / "rt1320_promotion"
                   / "PHASE1_FOLD0_DIAGNOSIS.json")
    p.add_argument("--fold", type=int, default=0, help="fold under diagnosis")
    return p.parse_args()


def partition_paths(part: str) -> tuple[str, Path]:
    """Return (oof_suffix, folds_path) for a partition name.

    For alternates this is NOT research/folds/folds_altK.parquet -- that file is
    a 2-column [id, fold] reassignment table over the 8,000 dev series, which
    build_row_arrays rejects. wave2_lib.alt_folds materialises the usable
    full-length table (10,000 rows, lockbox at -1) at cache/_folds_altK.parquet
    during any --partition run; that is the one the vectors are aligned to.
    """
    if part == "canonical":
        return "", REPO / "research" / "folds" / "folds.parquet"
    return f".{part}", REPO / "cache" / f"_folds_{part}.parquet"


def student_dir_for(part: str, override: Path | None) -> Path:
    """Student outputs are partition-suffixed on disk.

    armc_residual_student REFUSES to write an alt run into the canonical report
    directory, so alt results live in armc_residual_student.altK. Canonical has
    no stored outer .npy vectors in this worktree -- only committed JSON -- so a
    canonical diagnosis needs those refitted first.
    """
    if override is not None:
        return override
    base = REPO / "research" / "reports"
    return base / ("armc_residual_student" if part == "canonical"
                   else f"armc_residual_student.{part}")


def build_blends(part: str, student_dir: Path):
    """Rebuild E1 and E2 exactly as armc_e2_e1_addition_contract.py does."""
    suffix, folds_path = partition_paths(part)
    if not folds_path.exists():
        return None, f"missing folds file {folds_path}"

    A.OOF_SUFFIX = suffix
    data_root = A.default_data_root()
    oof_dir = data_root / "research" / "oof"
    cat_dir = A.default_catboost_oof_dir() or oof_dir

    cat300_path = cat_dir / f"RT-1255{suffix}.npy"
    cat413_path = cat_dir / f"RT-1254{suffix}.npy"
    if not cat300_path.exists() or not cat413_path.exists():
        # Fall back to the plain OOF dir, which is where alt CatBoost vectors land.
        cat300_path = oof_dir / f"RT-1255{suffix}.npy"
        cat413_path = oof_dir / f"RT-1254{suffix}.npy"
    for q in (cat300_path, cat413_path):
        if not q.exists():
            return None, f"missing CatBoost vector {q.name} — champion lane not computable"

    rows = A.build_row_arrays(folds_path)
    missing_outer = [f for f in range(5)
                     if not (student_dir / f"armc_residual_student_outer{f}.npy").exists()]
    if missing_outer:
        return None, (f"student outer vectors absent in {student_dir.name} "
                      f"(folds {missing_outer}) -- refit needed before diagnosis")
    student_raw, _ = A.merge_student_oof(student_dir, rows, force=False)
    student = A.crossfit_calibrate(student_raw, rows)
    spec = [A.load_calibrated_stream(oof_dir, s, rows) for s in A.SPECIALISTS]
    cat300 = A.crossfit_calibrate(np.load(cat300_path, mmap_mode="r"), rows)
    cat413 = A.crossfit_calibrate(np.load(cat413_path, mmap_mode="r"), rows)
    control = A.load_calibrated_stream(oof_dir, CONTROL_STREAM, rows)

    base = [cat300, spec[1], spec[2], spec[3], cat413, spec[5], spec[6]]
    return {
        "rows": rows,
        "E1": A.blend(base + [control]),
        "E2": A.blend(base + [student]),
    }, None


def delta_on(pack, mask, fold: int) -> float | None:
    """E2-E1 on `fold`, restricted to `mask`. None when the cell is degenerate."""
    e1 = A.mean_fold_ts_auc(pack["E1"], pack["rows"], mask=mask)["per_fold_ts_auc"]
    e2 = A.mean_fold_ts_auc(pack["E2"], pack["rows"], mask=mask)["per_fold_ts_auc"]
    a, b = e1.get(str(fold)), e2.get(str(fold))
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float(b - a)


def diagnose(pack, fold: int) -> dict:
    rows = pack["rows"]
    dev, t = rows["dev"], rows["t"]
    out: dict = {}

    out["overall"] = delta_on(pack, dev, fold)

    # The two negative classes the plan names. These masks keep all positives and
    # vary only which negatives are admitted, so the contrast isolates the class.
    out["by_negative_class"] = {
        "never_break_only": delta_on(pack, rows["dominant_never_break_only"], fold),
        "pre_break_only": delta_on(pack, rows["dominant_pre_break_only"], fold),
        "dominant_cell": delta_on(pack, rows["dominant_cell"], fold),
    }

    buckets = {}
    for name, lo, hi in HORIZON_BUCKETS:
        m = dev & (t >= lo) & (t < hi)
        n = int(m.sum())
        buckets[name] = {"n_rows": n, "delta_E2_minus_E1": delta_on(pack, m, fold) if n else None}
    out["by_horizon_bucket"] = buckets
    return out


def main() -> int:
    args = parse_args()
    fold = args.fold
    results: dict = {}
    skipped: dict = {}

    for part in args.partitions:
        print(f"=== {part} ===", flush=True)
        pack, err = build_blends(part, student_dir_for(part, args.student_dir))
        if pack is None:
            print(f"  SKIP: {err}", flush=True)
            skipped[part] = err
            continue
        results[part] = diagnose(pack, fold)
        print(f"  fold {fold} overall E2-E1 = {results[part]['overall']:+.6f}", flush=True)
        del pack

    # Q1 — the conditional that decides whether Q2 even matters.
    negatives = [p for p, r in results.items()
                 if r["overall"] is not None and r["overall"] < 0]
    verdict = {
        "fold_diagnosed": fold,
        "partitions_evaluated": sorted(results),
        "partitions_skipped": skipped,
        "partitions_with_negative_fold": sorted(negatives),
        "n_negative": len(negatives),
        "Q1_negative_on_multiple_partitions": len(negatives) > 1,
        "interpretation": (
            "Fold 0 negative on multiple partitions: the plan requires the damage "
            "regime be characterised before promotion is arguable regardless of "
            "the mean. See by_horizon_bucket and by_negative_class."
            if len(negatives) > 1 else
            "Fold 0 is not negative on multiple partitions among those evaluated; "
            "on this evidence it looks partition-specific rather than a property "
            "of the mechanism. Incomplete until all four partitions exist."
        ),
        "gates": False,
        "note": "Phase 1 §1.5 runs alongside and does not gate. Reports only.",
    }

    payload = {"verdict": verdict, "per_partition": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\n=== fold-{fold} diagnosis ===")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
