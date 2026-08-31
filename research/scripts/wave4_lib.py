"""WAVE-4 experiment definitions.  Pre-registered in research/RDOF_LEDGER.md
BEFORE any of them was run.  Never edits sbr core files.

Two seven-member sets are defined here and nothing else.  They exist to answer
one question: is the deployable seven-stream ensemble's gain over the single
champion bought by SPECIALIST DIVERSITY, or by ORDINARY BAGGING?

  SPECIALIST set  the wave-2 streams, configurations taken verbatim from
                  research/scripts/wave2_streams.py (which is itself the
                  manifest the Crunch-tested RT-150 artifact was built from)
  SEED set        the champion configuration seven times, changing ONLY the
                  seed, from a seed list fixed before the first run

Member 1 of BOTH sets is the champion configuration at seed 0, which on this
machine is already on disk as RT-300.  The sets are therefore maximally paired:
they share a member, the folds, the feature cache and the protocol.
"""
from __future__ import annotations

import os, sys

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from wave2_lib import CHAMP, FULL
from wave2_streams import JOBS

#: PRE-REGISTERED SEED LIST.  Fixed 2026-08-20 before the first wave-4 run.
#: Seven seeds, no substitutions, no "best seven of ten".
SEEDS = [0, 1, 7, 42, 2026, 31415, 271828]

#: seed clones: champion protocol, champion modules, ONLY the seed varies.
SEED_CLONES = {
    "RT-300": 0,        # already on disk -- the champion itself
    "RT-401": 1,
    "RT-402": 7,
    "RT-403": 42,
    "RT-404": 2026,
    "RT-405": 31415,
    "RT-406": 271828,
}

#: specialist streams: RT-300 is the wave-2 RT-100R configuration exactly, and
#: RT-410..RT-415 are wave-2 RT-120R..RT-125R re-run on this machine.
SPECIALIST_ALIAS = {
    "RT-300": "RT-100R",
    "RT-410": "RT-120R",
    "RT-411": "RT-121R",
    "RT-412": "RT-122R",
    "RT-413": "RT-123R",
    "RT-414": "RT-124R",
    "RT-415": "RT-125R",
}

SPECIALISTS = {new: dict(JOBS[old]) for new, old in SPECIALIST_ALIAS.items() if old != "RT-100R"}


# ---------------------------------------------------------------------------
# W4-E5: the EXACT ORIGINAL wave-1 streams that RT-131's oracle was built from.
#
# RT-120R..RT-125R are NOT these.  research/scripts/wave2_streams.py says so in
# its own docstring, and research/EXPERIMENT_ID_MAP.md section 4 tabulates the
# differences.  Only RT-100 and RT-123 carry their wave-1 configuration into the
# wave-2 set, so a "99.7% of the oracle gain recovered" claim measured over the
# R streams is not the apples-to-apples answer.
#
# Configurations below are copied verbatim from the scripts that produced the
# wave-1 RESULTS.csv rows:
#   RT-120, RT-121, RT-122  <- research/scripts/agent0_diversity.py
#   RT-124                  <- research/scripts/agent0_streams2.py
#   RT-125                  <- research/scripts/agent0_stream_f.py
#   RT-123                  <- research/scripts/agent0_streams2.py (== RT-413)
#   RT-100                  <- the champion (== RT-300)
#
# RT-125 / GOSS, the config flagged as version-sensitive: agent0_streams2.py and
# agent0_stream_f.py disagree about whether bagging is disabled alongside GOSS.
# Under lightgbm 4.7.0 the two forms train to bitwise-identical predictions
# (GOSS ignores bagging_fraction), so the ambiguity is immaterial and the stream
# IS reproducible under the current environment.  The stream_f form is used
# because it is the one whose docstring says it was the re-run.
ORIGINAL_ALIAS = {
    "RT-300": "RT-100",     # champion, unchanged between waves
    "RT-430": "RT-120",
    "RT-431": "RT-121",
    "RT-432": "RT-122",
    "RT-413": "RT-123",     # wave-2 RT-123R is byte-identical to wave-1 RT-123
    "RT-433": "RT-124",
    "RT-434": "RT-125",
}
ORIGINAL_SET = list(ORIGINAL_ALIAS)

_ORIG = {
 "RT-430": dict(modules=["m00_core", "m01_seq", "m07_bayes"], seed=0, max_train_rows=900_000,
                params={"n_estimators": 700, "learning_rate": 0.04, "num_leaves": 127,
                        "min_data_in_leaf": 150, "feature_fraction": 0.35,
                        "bagging_fraction": 0.7, "lambda_l2": 5.0, "max_bin": 127},
                note="wave-1 stream A verbatim: evidence-recursion heavy, deep trees"),
 "RT-431": dict(modules=["m02_dist", "m03_dyn", "m04_resid", "m06_loc"], seed=1,
                max_train_rows=900_000,
                params={"n_estimators": 500, "learning_rate": 0.08, "num_leaves": 31,
                        "min_data_in_leaf": 500, "feature_fraction": 0.7,
                        "bagging_fraction": 0.7, "lambda_l2": 10.0, "max_bin": 127},
                note="wave-1 stream B verbatim: shape/dynamics heavy, shallow trees"),
 "RT-432": dict(modules=FULL, seed=7, max_train_rows=900_000, sample_mode="per_series",
                params={"n_estimators": 500, "learning_rate": 0.06, "num_leaves": 255,
                        "min_data_in_leaf": 400, "feature_fraction": 0.3,
                        "bagging_fraction": 0.6, "lambda_l2": 20.0, "max_bin": 63,
                        "extra_trees": True},
                note="wave-1 stream C verbatim: extra-trees, per-series sampling"),
 "RT-433": dict(modules=["m07_bayes", "m06_loc", "m01_seq"], seed=3, max_train_rows=700_000,
                params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63,
                        "min_data_in_leaf": 200, "feature_fraction": 0.6,
                        "bagging_fraction": 0.8, "lambda_l2": 3.0, "max_bin": 255},
                note="wave-1 stream E verbatim: recursion-only, 170 cols"),
 "RT-434": dict(modules=FULL, seed=11, max_train_rows=700_000,
                params={"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 95,
                        "min_data_in_leaf": 250, "feature_fraction": 0.45, "boosting": "goss",
                        "top_rate": 0.25, "other_rate": 0.15, "lambda_l2": 8.0,
                        "max_bin": 127, "bagging_fraction": 1.0, "bagging_freq": 0},
                note="wave-1 stream F verbatim: GOSS gradient sampling"),
}

SEED_SET = list(SEED_CLONES)
SPECIALIST_SET = list(SPECIALIST_ALIAS)


def job(exp_id):
    """Return the kwargs for one wave-4 run."""
    if exp_id in SPECIALISTS:
        j = dict(SPECIALISTS[exp_id])
        note = j.pop("note")
        old = SPECIALIST_ALIAS[exp_id]
        return dict(
            exp_id=exp_id, agent="claude-wave4", folds=(0, 1, 2, 3, 4),
            hypothesis=f"macOS/arm64 reconstruction of the wave-2 specialist stream {old}, "
                       f"config verbatim from research/scripts/wave2_streams.py, so that the "
                       f"specialist ensemble can be scored against a seed-clone ensemble on ONE platform",
            falsification="n/a (reconstruction, not a candidate); the ensemble comparison it feeds "
                          "is falsified if the seed-clone ensemble is not beaten by more than 0.0030",
            notes=f"WAVE-4 specialist stream, alias of {old}; {note}", **j)
    if exp_id in _ORIG:
        j = dict(_ORIG[exp_id])
        note = j.pop("note")
        old = ORIGINAL_ALIAS[exp_id]
        return dict(
            exp_id=exp_id, agent="claude-wave4", folds=(0, 1, 2, 3, 4),
            hypothesis=f"macOS/arm64 reconstruction of the EXACT wave-1 stream {old}, config verbatim "
                       f"from the agent0 script that produced its RESULTS.csv row, so that RT-131's "
                       f"oracle gain and its legal SCDF recovery can be measured apples-to-apples",
            falsification="n/a (reconstruction)",
            notes=f"WAVE-4 exact wave-1 stream, alias of {old}; {note}", **j)
    if exp_id in SEED_CLONES:
        return dict(
            exp_id=exp_id, modules=FULL, agent="claude-wave4", seed=SEED_CLONES[exp_id],
            hypothesis="A seed clone of the champion carries no new information, so a seven-way "
                       "seed-clone ensemble measures the BAGGING component of the seven-stream "
                       "ensemble's gain",
            falsification="n/a (control arm)",
            notes=f"WAVE-4 seed clone, seed {SEED_CLONES[exp_id]}, champion protocol/modules",
            **CHAMP)
    raise KeyError(exp_id)


if __name__ == "__main__":
    from sbr.pipeline import run
    for e in sys.argv[1:]:
        run(**job(e))
