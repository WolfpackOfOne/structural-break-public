# Leaderboard Alpha 2026 Final

Date: 2026-08-26

Program preregistration: `PROGRAM_PREREG.md` at `ed84d00`.
Branch: `research/leaderboard-alpha-2026`.

## Final Verdict

**PROGRAM EXHAUSTED. No promotion. Production `RT-600` remains unchanged.**

All three preregistered experiments ran on the full five canonical folds from
the start. None survived its frozen gate, so the combination rule does not open.

| experiment | IDs | final verdict | key number | reason |
|---|---|---|---:|---|
| LA-01 specialist replacement salvage | `RT-1243` / `RT-1244` | KILL | `-0.000005408` marginal vs clone | replacement did not beat seed replacement and dominant net was `-10` |
| LA-02 counterfactual synthetic augmentation | `RT-1245` / `RT-1246` | KILL | `+0.005157709` marginal vs synthetic clone | metric-class MAJOR, but dominant net `-326`, mature-vs-never net `-366`, and candidate below RT600 |
| LA-03 per-series history adaptation | `RT-1247` / `RT-1248` / `RT-1249` | KILL | `+0.001676065` integrated marginal vs global clone | WEAK metric signal, but adapted standalone loses to fixed null by `-0.021495290` |

## What Changed Scientifically

LA-02 and LA-03 both show that weak controls can be beaten: paired synthetic
labels beat null-only synthetic rows, and affine adaptation beats a global
no-adaptation AR(5). Neither beat the binding incumbent-relevant control. The
fixed per-series null remains the strongest legal history representation, and
the residual pair-flow diagnostics stayed negative.

## Files

- `LA01_NESTED_REPLACEMENT.md`, `la01_nested_replacement.{json,csv}`
- `LA02_COUNTERFACTUAL_AUGMENTATION.md`, `la02_counterfactual_augmentation.{json,csv}`
- `LA03_PER_SERIES_ADAPTATION.md`, `la03_per_series_adaptation.{json,csv}`

No lockbox, reduced test set, protected branch, or external submission was
touched.
