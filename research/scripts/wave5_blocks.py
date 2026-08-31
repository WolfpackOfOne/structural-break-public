"""Stage-C runs for the three new feature blocks, at the wave-3 ABL protocol.

Pre-registered in research/WAVE5_PREREG.md section 6 (W5-E2, E4, E5, E6, E7, E10).

The protocol is deliberately the one wave 3 used for m09_back, and the control
arms are the ones wave 3 already ran and left on disk:

    RT-301   control, 7 production modules, ABL 400k     (on disk)
    RT-303   seed clone of that control, seed 1          (on disk)

so every new block is directly comparable to the last block that failed, and no
compute is spent re-running a control that already exists.

    RT-730   + m11_focus    exact max over candidate tau              (W5-E7)
    RT-740   + m10_persist  outlier-vs-bulk scale, D2-driven          (W5-E2)
    RT-750   + m12_rdep     residual distances / CUSUMSQ / dep LR     (W5-E4/5/6)
    RT-760   + all three                                              (W5-E10)
"""
from __future__ import annotations

import os, sys, time

ROOT = os.environ.get(
    "SBR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/research/scripts")

from sbr.pipeline import run
from wave2_lib import ABL, FULL

JOBS = {
    "RT-740": (["m10_persist"],
               "W5-E2. The champion's highest-ranked no-break series are dominated by "
               "heavy-tail/outlier and variance-burst mechanisms (W5-D2: tail_rate_online "
               "+1.89 IQR in the top 1%). Nothing in the bank computes a trimmed statistic, "
               "an energy-concentration statistic or an exceedance run length, so an "
               "outlier-driven scale excursion and a bulk scale break arrive looking alike. "
               "The untrimmed-minus-trimmed contrast should separate them."),
    "RT-750": (["m12_rdep"],
               "W5-E4/E5/E6. m02_dist computes its distances on the RAW PIT only, m01_seq "
               "runs CUSUM/CUSUMSQ paths on the raw series only, and no module computes a "
               "likelihood ratio for a change in the AR coefficient with the innovation "
               "variance profiled out. Residualising first should expose breaks that leave "
               "the marginal law intact and rewire the dynamics."),
    "RT-760": (["m10_persist", "m11_focus", "m12_rdep"],
               "W5-E10. If the three blocks target genuinely different mechanisms their "
               "contributions should be at least partly additive; if they are all reading "
               "the same saturated signal, the union will not beat the best single block."),
}


def main(which):
    for exp in which:
        mods, hyp = JOBS[exp]
        t0 = time.time()
        run(exp_id=exp, modules=FULL + mods, agent="claude-wave5", seed=0,
            hypothesis=hyp,
            falsification="mean ABL delta vs RT-301 <= 0, OR positive on <= 2 of 5 folds, OR "
                          "the two-model blend delta does not beat the RT-303 seed clone's",
            notes=f"W5 stage C: RT-301 + {'+'.join(mods)}; ABL protocol, arms identical to "
                  f"the wave-3 m09_back test",
            **ABL)
        print(f"[{exp}] {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(JOBS))
