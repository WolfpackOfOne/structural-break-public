"""WAVE-2 ensemble streams.

NOTE (recorded honestly): these are NOT byte-reproductions of wave-1 RT-120..125.
The verbatim wave-1 configurations live in research/scripts/agent0_diversity.py,
agent0_streams2.py and agent0_stream_f.py; the configs below differ (different
n_estimators / learning_rate / min_data_in_leaf / lambda_l2 / max_bin).  They are
legitimate diverse streams in their own right and are labelled RT-1xxR, but any
comparison of RT-1xxR against the wave-1 RT-1xx number measures CONFIGURATION,
not reproducibility.  Only RT-100R and RT-123R share their wave-1 configuration.

Rebuild the six non-champion ensemble streams (RT-120..RT-125) so that the
DEPLOYABLE-ENSEMBLE question can be answered with real OOF predictions.

Configurations are taken verbatim from research/RESULTS.csv wave-6 rows.
"""
import os, sys
ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")
from sbr.pipeline import run
from wave2_lib import FULL

BASE = dict(learning_rate=0.05, lambda_l2=5.0, max_bin=127, n_estimators=600)

JOBS = {
 "RT-120R": dict(modules=["m00_core", "m01_seq", "m07_bayes"], seed=0, max_train_rows=900_000,
                 params=dict(BASE, num_leaves=127, min_data_in_leaf=300, feature_fraction=0.35,
                             bagging_fraction=0.7),
                 note="stream A: evidence-recursion heavy, deep trees"),
 "RT-121R": dict(modules=["m02_dist", "m03_dyn", "m04_resid", "m06_loc"], seed=1, max_train_rows=900_000,
                 params=dict(BASE, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.7,
                             bagging_fraction=0.7),
                 note="stream B: shape/dynamics heavy, shallow trees"),
 "RT-122R": dict(modules=FULL, seed=7, max_train_rows=900_000, sample_mode="per_series",
                 params=dict(BASE, num_leaves=255, min_data_in_leaf=500, feature_fraction=0.5,
                             bagging_fraction=0.7, extra_trees=True),
                 note="stream C: extra-trees, per-series row sampling, different seed"),
 "RT-123R": dict(modules=FULL, seed=0, max_train_rows=700_000,
                 params=dict(BASE, objective="pairwise_t", num_leaves=63, min_data_in_leaf=300,
                             feature_fraction=0.5, bagging_fraction=0.7),
                 note="stream D: pairwise logistic, groups = online index t"),
 "RT-124R": dict(modules=["m07_bayes", "m06_loc", "m01_seq"], seed=3, max_train_rows=700_000,
                 params=dict(BASE, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.6,
                             bagging_fraction=0.7),
                 note="stream E: 170 cols, no window-bank features at all"),
 "RT-125R": dict(modules=FULL, seed=11, max_train_rows=700_000,
                 params=dict(BASE, boosting="goss", num_leaves=63, min_data_in_leaf=300,
                             feature_fraction=0.5, bagging_fraction=1.0, bagging_freq=0),
                 note="stream F: GOSS keeps large-gradient rows instead of sampling uniformly"),
}

def _main():
    """Guarded: importing JOBS must NEVER launch training runs."""
    for exp in (sys.argv[1:] or list(JOBS)):
        j = dict(JOBS[exp])
        note = j.pop("note")
        run(exp_id=exp, agent="B-streams", folds=(0, 1, 2, 3, 4),
            hypothesis=f"{exp[:-1]} rebuilds under wave-2 reproducibility discipline and gives usable OOF",
            falsification="mean OOF TS-AUC differs from the wave-1 record by more than 0.004",
            notes=f"wave-2 rebuild; {note}", **j)


if __name__ == "__main__":
    _main()
