"""Wave 8 five-pilot comparison table (execution-brief section 22) +
redundancy matrix (section 26). Reads each mechanism's already-written
research/reports/wave8_{sst,orr,pcfb,cfep,tgmc}.json -- computes nothing new
about model fit, only assembles + adds the shared pair-repair diagnostic
against the RT-990 control and (for T2) against research/oof/RT-995.npy.

Usage:
    python wave8_pilot_comparison.py --compare
    python wave8_pilot_comparison.py --redundancy
"""
from __future__ import annotations

import json, os

import numpy as np

import wave8_common as W8
from wave5_lib import Ctx, OOFDIR, REPORTS
from wave7_d3r import cell_mask, score_on

OUTER_F = 0

PRIMARY = {
    "SST": ("RT-1006", f"{REPORTS}/wave8_sst.json"),
    "ORR": ("RT-1021", f"{REPORTS}/wave8_orr.json"),
    "PCFB": ("RT-1033", f"{REPORTS}/wave8_pcfb.json"),
    "CFEP": ("RT-1042", f"{REPORTS}/wave8_cfep_compare.json"),
    "TGMC": ("RT-1052", f"{REPORTS}/wave8_tgmc.json"),
}

LABELS = {"CLEARED": "PASS", "NOT CLEARED": "FAIL"}


def cmd_compare():
    c = Ctx()
    y, t, age = c.d.y, c.d.t, c.age
    r0 = c.rows[OUTER_F]
    cellmask0 = cell_mask(y[r0], t[r0], age[r0])
    A = np.load(f"{OOFDIR}/RT-990.npy")

    rows_table = []
    for mech, (exp_id, report_path) in PRIMARY.items():
        if not (os.path.exists(f"{OOFDIR}/{exp_id}.npy") and os.path.exists(report_path)):
            rows_table.append({"mechanism": mech, "status": "NOT RUN"})
            continue
        cand = np.load(f"{OOFDIR}/{exp_id}.npy")
        with open(report_path) as f:
            rep = json.load(f)
        valid_rows = r0[~np.isnan(cand[r0])]
        standalone_whole = score_on(cand[valid_rows], np.ones(len(valid_rows), bool),
                                    y[valid_rows], t[valid_rows])
        control_whole = score_on(A[valid_rows], np.ones(len(valid_rows), bool),
                                 y[valid_rows], t[valid_rows])
        cm = cellmask0[np.isin(r0, valid_rows)]
        cell_rows = valid_rows[np.isin(valid_rows, r0[cellmask0])]
        cell_cand = score_on(cand[cell_rows], np.ones(len(cell_rows), bool), y[cell_rows], t[cell_rows]) \
            if len(cell_rows) else None
        cell_ctrl = score_on(A[cell_rows], np.ones(len(cell_rows), bool), y[cell_rows], t[cell_rows]) \
            if len(cell_rows) else None
        pair = W8.pair_repair_stats(A, cand, y, t, valid_rows, n_pairs_per_t=10, seed=0)
        # PCFB's compare-mode report nests each method's ensemble dict under
        # its own key (RT-1033 is PLS16) rather than exposing one top-level
        # "ensemble", unlike every other mechanism's report.
        ens = rep.get("ensemble") or rep.get("pls16", {}).get("ensemble", {})
        cand_key = next((k for k in ens if k.startswith("rt600_plus_") and k != "rt600_plus_seedclone"), None)
        gate = rep.get("continuation_gate_cleared")
        rows_table.append({
            "mechanism": mech, "exp_id": exp_id,
            "matched_control": "RT-990",
            "standalone_delta": (standalone_whole - control_whole)
                if (standalone_whole is not None and control_whole is not None) else None,
            "dominant_cell_delta": (cell_cand - cell_ctrl)
                if (cell_cand is not None and cell_ctrl is not None) else None,
            "rt600_plus_candidate": ens.get(cand_key) if cand_key else None,
            "rt600_plus_clone": ens.get("rt600_plus_seedclone"),
            "marginal_vs_clone": ens.get("marginal_vs_clone"),
            "repairs": pair["repairs"], "damage": pair["damage"], "net_pair_lift": pair["net_pair_lift"],
            "runtime_s": rep.get("training_runtime_s"),
            "verdict": "CLEARED" if gate else ("NOT CLEARED" if gate is not None else "UNKNOWN"),
        })

    with open(f"{REPORTS}/wave8_pilot_comparison.json", "w") as f:
        json.dump(rows_table, f, indent=2, default=float)

    def fmt(v):
        return f"{v:+.5f}" if isinstance(v, (int, float)) else "n/a"

    md = ["# WAVE 8 -- FIVE-PILOT COMPARISON (fold 0)\n",
          "| mechanism | matched control | standalone Δ | dominant-cell Δ | marginal Δ vs clone | "
          "repairs | damage | net pair lift | verdict |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows_table:
        if r.get("status") == "NOT RUN":
            md.append(f"| {r['mechanism']} | - | - | - | - | - | - | - | NOT RUN |")
            continue
        mech_label = r['mechanism'] + ("*" if r['mechanism'] == "ORR" else "")
        md.append(f"| {mech_label} | {r['matched_control']} | {fmt(r['standalone_delta'])} | "
                  f"{fmt(r['dominant_cell_delta'])} | {fmt(r['marginal_vs_clone'])} | "
                  f"{r['repairs']} | {r['damage']} | {r['net_pair_lift']} | {r['verdict']} |")
    md.append("\n*ORR's `RT-1021` is a blend on the RT600 7-stream ENSEMBLE score, not a "
              "single-model score like every other row's candidate -- its standalone/cell "
              "deltas here compare ensemble-scale to single-model-scale and read misleadingly "
              "large. The honest ORR comparison is RT600-alone vs RT600+repair, both at "
              "ensemble scale: whole -0.00005, dominant-cell -0.00004 (`research/reports/"
              "wave8_orr.json`). Its `marginal_vs_clone` and `net_pair_lift` columns above are "
              "computed consistently with the other rows and remain valid.")
    with open(f"{REPORTS}/wave8_pilot_comparison.md", "w") as f:
        f.write("\n".join(md))
    print(json.dumps(rows_table, indent=2, default=float))
    print(f"wrote {REPORTS}/wave8_pilot_comparison.{{md,json}}")
    return rows_table


def cmd_redundancy():
    c = Ctx()
    y, t = c.d.y, c.d.t
    r0 = c.rows[OUTER_F]
    vecs = {"RT600_A": np.load(f"{OOFDIR}/RT-990.npy")}
    if os.path.exists(f"{OOFDIR}/RT-995.npy"):
        vecs["T2"] = np.load(f"{OOFDIR}/RT-995.npy")
    for mech, (exp_id, _) in PRIMARY.items():
        p = f"{OOFDIR}/{exp_id}.npy"
        if os.path.exists(p):
            vecs[mech] = np.load(p)

    names = [k for k in vecs if k != "RT600_A"]
    out = {"pairwise": []}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            va, vb = vecs[a], vecs[b]
            rows = r0[(~np.isnan(va[r0])) & (~np.isnan(vb[r0]))]
            if len(rows) < 100:
                continue
            corr = float(np.corrcoef(va[rows], vb[rows])[0, 1])
            rp_a = W8.pair_repair_stats(vecs["RT600_A"], va, y, t, rows, n_pairs_per_t=10, seed=0)
            rp_b = W8.pair_repair_stats(vecs["RT600_A"], vb, y, t, rows, n_pairs_per_t=10, seed=0)
            out["pairwise"].append({
                "a": a, "b": b, "within_t_rank_corr": corr,
                "a_repairs": rp_a["repairs"], "b_repairs": rp_b["repairs"],
            })
    with open(f"{REPORTS}/wave8_futureaware_redundancy.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    md = ["# WAVE 8 -- REDUNDANCY MATRIX (fold 0)\n",
          "| a | b | within-t rank corr | a repairs (vs RT600 control) | b repairs |",
          "|---|---|---:|---:|---:|"]
    for r in out["pairwise"]:
        md.append(f"| {r['a']} | {r['b']} | {r['within_t_rank_corr']:.4f} | {r['a_repairs']} | {r['b_repairs']} |")
    with open(f"{REPORTS}/wave8_futureaware_redundancy.md", "w") as f:
        f.write("\n".join(md))
    print(json.dumps(out, indent=2, default=float))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--redundancy", action="store_true")
    args = ap.parse_args()
    if args.compare:
        cmd_compare()
    elif args.redundancy:
        cmd_redundancy()
    else:
        raise SystemExit("pass --compare or --redundancy")


if __name__ == "__main__":
    main()
