"""W4-E5: the apples-to-apples RT-131 audit.

The claim under audit, from sbr/production/calibration.py's own docstring:

    "this recovers 99.7% of the oracle rank-average's gain over the best single
     model"

That number was measured over the wave-2 `R` streams.  RT-131 -- the oracle it
is a recovery OF -- was built from the wave-1 streams, and five of the seven
differ (see research/EXPERIMENT_ID_MAP.md section 4).  So the headline recovery
figure describes a stream set that RT-131 was never computed on.

This computes, on ONE platform, for BOTH stream sets:

    single champion            the legal floor
    ORACLE within-t rank avg   the illegal ceiling  (this is what RT-131 is)
    LEGAL SCDF ensemble        the deployable system
    recovery = (legal - single) / (oracle - single)

Every calibration is cross-fitted.  The oracle is a DIAGNOSTIC CEILING: the
Crunch runner is series-sequential and single-pass, so the live cross-section at
time t does not exist at inference and no oracle number is ever a champion.
"""
from __future__ import annotations

import json, os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import numpy as np
from sbr.pipeline import Data
from wave4_ensemble import Blends, FOLDS, score, within_t_rank
from wave4_lib import ORIGINAL_ALIAS, ORIGINAL_SET, SPECIALIST_ALIAS, SPECIALIST_SET

OOFDIR = f"{ROOT}/research/oof"
OUT = f"{ROOT}/research/reports/rt131_original_audit.json"
SINGLE = "RT-300"


def main():
    d = Data()
    rows = {f: d.rows_for([f]) for f in FOLDS}
    dev = d.rows_for(list(FOLDS))

    sets = {"wave1_original": (ORIGINAL_SET, ORIGINAL_ALIAS),
            "wave2_R_streams": (SPECIALIST_SET, SPECIALIST_ALIAS)}
    need = sorted({e for s, _ in sets.values() for e in s})
    missing = [e for e in need if not os.path.exists(f"{OOFDIR}/{e}.npy")]
    if missing:
        sys.exit(f"MISSING OOF: {missing}")
    P = {e: np.load(f"{OOFDIR}/{e}.npy") for e in need}

    single_m, single_pf = score(P[SINGLE], d, rows)
    res = {"single_champion": {"id": SINGLE, "mean": single_m, "per_fold": single_pf},
           "sets": {}}
    print(f"single champion {SINGLE}: {single_m:.5f}\n")

    B = Blends(P, d, rows)
    for name, (streams, alias) in sets.items():
        print(f"=== {name} ===")
        ind = {e: score(P[e], d, rows)[0] for e in streams}
        for e in streams:
            print(f"  {e} (= {alias[e]}) {ind[e]:.5f}")
        orc = np.full(len(d.y), np.nan)
        orc[dev] = np.column_stack([within_t_rank(P[s][dev], d.t[dev]) for s in streams]).mean(1)
        om, opf = score(orc, d, rows)
        entry = {"streams": list(streams), "alias": {e: alias[e] for e in streams},
                 "individual": ind,
                 "oracle_within_t_rank_ILLEGAL": {"mean": om, "per_fold": opf},
                 "legal": {}}
        print(f"  ORACLE (illegal, this is what RT-131 is): {om:.5f}")
        for kind in ("scdf_t", "scdf_nseen", "gcdf", "logitmean", "raw"):
            v = B.crossfit(kind, streams)
            m, pf = score(v, d, rows)
            rec = (m - single_m) / max(om - single_m, 1e-12)
            entry["legal"][kind] = {"mean": m, "per_fold": pf,
                                    "gain_over_single": m - single_m,
                                    "pct_of_oracle_gain_recovered": 100.0 * rec}
            print(f"  legal {kind:11s} {m:.5f}  gain {m-single_m:+.5f}  "
                  f"recovers {100*rec:.1f}% of the oracle gain")
        entry["oracle_gain_over_single"] = om - single_m
        res["sets"][name] = entry
        print()

    a = res["sets"]["wave1_original"]["legal"]["scdf_t"]["pct_of_oracle_gain_recovered"]
    b = res["sets"]["wave2_R_streams"]["legal"]["scdf_t"]["pct_of_oracle_gain_recovered"]
    res["headline"] = {
        "claimed_in_calibration_docstring": 99.7,
        "measured_wave2_R_streams": b,
        "measured_wave1_original_streams_APPLES_TO_APPLES": a,
        "note": ("RT-131 is the ORACLE over the wave-1 streams.  The apples-to-apples "
                 "recovery figure is the wave1_original one; the wave-2 R figure "
                 "describes a different stream set."),
    }
    print(f"HEADLINE: docstring claims 99.7%; wave-2 R streams give {b:.1f}%; "
          f"the wave-1 streams RT-131 was actually built from give {a:.1f}%")
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
