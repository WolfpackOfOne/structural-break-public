"""W5-E9: does a TS-AUC-shaped ranking objective add information?

Pre-registered in research/WAVE5_PREREG.md section 6.

HYPOTHESIS
    The official metric pools concordant pairs over sum_t n_pos(t)*n_neg(t), so
    every within-timestep (positive, negative) pair counts EQUALLY.  The
    incumbent `pairwise_t` stream (RT-123R / RT-413) samples a FIXED m_neg=8
    negatives per positive row, which makes each POSITIVE ROW count equally
    instead -- systematically under-weighting the timesteps where many series
    are still alive, which D1 shows is where most of the metric's mass sits.
    Correcting the pair weighting to the metric's own structure should produce a
    model that is better aligned with what is scored.

MECHANISM
    W5-E9a `pairwise_w`  same pairs, weight proportional to n_neg(t).
    W5-E9b `pairwise_h`  same pairs, same weights, squared-hinge margin loss --
                         isolates the loss shape from the weighting.

FALSIFICATION
    Neither variant beats the incumbent `pairwise_t` control (RT-702, re-run
    here under the identical hook so the three arms differ ONLY in the loss),
    or the ensemble S+C fails the section-4 matched seed-control bar.

CONTROL
    RT-702 is the incumbent objective routed through the SAME hook.  It exists
    so that the comparison cannot be contaminated by the dispatch mechanism.
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

import wave5_obj
from wave2_lib import FULL
from sbr.pipeline import run

BASE = dict(n_estimators=600, learning_rate=0.05, lambda_l2=5.0, max_bin=127,
            num_threads=2, num_leaves=63, min_data_in_leaf=300,
            feature_fraction=0.5, bagging_fraction=0.7)

#: the RT-123R protocol exactly: FULL modules, seed 0, 700k rows.  Only the
#: objective moves between arms.
JOBS = {
    "RT-702": dict(kind="pairwise_t", note="CONTROL: incumbent pairwise_t via the W5 hook"),
    "RT-700": dict(kind="pairwise_w", note="W5-E9a: pairs weighted by n_neg(t), the metric's own structure"),
    "RT-701": dict(kind="pairwise_h", note="W5-E9b: same weights, squared-hinge margin loss"),
}


def main(which):
    for exp in which:
        j = JOBS[exp]
        wave5_obj.SPEC = {"kind": j["kind"], "m_neg": 8}
        wave5_obj.install()
        t0 = time.time()
        try:
            run(exp_id=exp, modules=FULL, agent="claude-wave5", seed=0,
                folds=(0, 1, 2, 3, 4), max_train_rows=700_000,
                params=dict(BASE, objective="pairwise_t"),
                hypothesis="The official TS-AUC pools every within-timestep (pos,neg) pair equally; "
                           "the incumbent pairwise objective instead weights every POSITIVE ROW "
                           "equally. Correcting the pair weighting to n_neg(t) should align the "
                           "model with the metric and add information a seed clone cannot.",
                falsification="fails to beat the RT-702 incumbent-objective control, or the "
                              "S+C ensemble fails the matched seed-control bar of +0.0030",
                notes=f"W5-E9 {j['note']}; RT-123R protocol, only the objective differs",
                extra={"objective": j["kind"]})
        finally:
            wave5_obj.restore()
        print(f"[{exp}] {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(JOBS))
