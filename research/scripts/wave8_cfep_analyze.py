"""CFEP pilot analysis: CFEP-C (future-predictive embedding) vs CFEP-B
(matched-capacity BCE embedding) -- the binding comparison per
WAVE8_FUTURE_AWARE_PREREG.md section 5.4 -- plus CFEP-A (RT-990 reused) and
the RT600 ensemble marginal.
"""
from __future__ import annotations

import json

import numpy as np

import wave8_common as W8
from wave5_lib import Ctx, OOFDIR, REPORTS
from wave7_d3r import cell_mask, score_on

OUTER_F = 0


def main():
    c = Ctx()
    y, t, age = c.d.y, c.d.t, c.age
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])

    A = np.load(f"{OOFDIR}/RT-990.npy")
    B = np.load(f"{OOFDIR}/RT-1041.npy")
    Cc = np.load(f"{OOFDIR}/RT-1042.npy")

    def whole(v):
        return score_on(v[r0], np.ones(len(r0), bool), y[r0], t[r0])

    def cell(v):
        rr = r0[cellmask0]
        return score_on(v[rr], np.ones(len(rr), bool), y[rr], t[rr])

    ens = W8.ensemble_marginal(Cc, c=c, fold=OUTER_F, label="cfep_c")

    out = {
        "A_control_whole": whole(A), "A_control_cell": cell(A),
        "B_bce_embedding_whole": whole(B), "B_bce_embedding_cell": cell(B),
        "C_future_embedding_whole": whole(Cc), "C_future_embedding_cell": cell(Cc),
        "delta_C_minus_B_whole": whole(Cc) - whole(B),
        "delta_C_minus_B_cell": cell(Cc) - cell(B),
        "delta_C_minus_A_whole": whole(Cc) - whole(A),
        "delta_C_minus_A_cell": cell(Cc) - cell(A),
        "ensemble": ens,
    }
    gate_binding = out["delta_C_minus_B_whole"] >= 0.002
    gate_standalone = (out["delta_C_minus_A_whole"] >= 0.003) or (ens["marginal_vs_clone"] >= 0.0015)
    out["continuation_gate_cleared"] = bool(gate_binding and gate_standalone)
    print(json.dumps(out, indent=2))

    with open(f"{REPORTS}/wave8_cfep_compare.json", "w") as f:
        json.dump(out, f, indent=2)
    md = ["# WAVE 8 -- CFEP PILOT RESULT (fold 0)\n",
          "Binding comparison is C (future-predictive) vs B (matched-capacity BCE), "
          "not vs A -- a future-predictive-loss embedding must beat a same-architecture "
          "BCE-trained one, per WAVE6_NEURAL_PREREG's prior neural-track failure mode.\n",
          "| arm | whole fold0 | dominant cell |", "|---|---:|---:|",
          f"| A control (`RT-990`) | {out['A_control_whole']:.5f} | {out['A_control_cell']:.5f} |",
          f"| B BCE embedding (`RT-1041`) | {out['B_bce_embedding_whole']:.5f} | {out['B_bce_embedding_cell']:.5f} |",
          f"| C future embedding (`RT-1042`) | {out['C_future_embedding_whole']:.5f} | {out['C_future_embedding_cell']:.5f} |",
          f"\n**C - B (binding, whole):** {out['delta_C_minus_B_whole']:+.5f}\n",
          f"\n**C - B (binding, cell):** {out['delta_C_minus_B_cell']:+.5f}\n",
          f"\n**C - A (whole):** {out['delta_C_minus_A_whole']:+.5f}\n",
          "\n## Ensemble marginal (fold 0)\n", "```json", json.dumps(ens, indent=2), "```",
          f"\n## CONTINUATION GATE: {'CLEARED' if out['continuation_gate_cleared'] else 'NOT CLEARED'}\n"]
    with open(f"{REPORTS}/wave8_cfep_compare.md", "w") as f:
        f.write("\n".join(md))
    print(f"wrote {REPORTS}/wave8_cfep_compare.{{md,json}}")
    return out


if __name__ == "__main__":
    main()
