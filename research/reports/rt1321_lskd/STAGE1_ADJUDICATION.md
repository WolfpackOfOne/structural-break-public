# RT-1321 Stage 1 — adjudication against the preregistered kill rule

Evidence: `STAGE1_SCREEN.json`, canonical fold 0, anchor = frozen RT-1320.
The anchor rebuilt to mean fold TS-AUC **0.629253722**, matching the recorded
`E2` of the RT-1320 addition contract (`0.6292537222254164`) to nine decimals —
the incumbent is bit-for-bit the one that scored 0.6303 externally.

## The numbers

Standalone fold-0 TS-AUC (one LightGBM on the 28-column block alone):

| cut | RT-1320 | JOINT `m19_lskd` | MARGINAL `m19_lskm` |
|---|---:|---:|---:|
| whole fold | 0.640281 | 0.542467 | 0.544396 |
| dominant cell | 0.680891 | 0.557594 | 0.567139 |
| mature vs never-break | 0.679726 | 0.561493 | 0.572684 |
| mature vs pre-break | 0.684141 | 0.546716 | 0.551667 |

Within-`t` rank rho vs RT-1320 (dominant cell): joint **0.162**, marginal 0.140.
Candidate/control rho (dominant cell): **0.303**.

Conditional AUC on the same-`t` pairs RT-1320 *inverts* (dominant cell):
joint **0.4712**, marginal 0.4892 — both below chance.

Fixed-perturbation grid, dominant cell (`repairs − damage`, and the TS-AUC the
perturbation actually buys):

| eps | joint net | joint ΔAUC | marginal net | marginal ΔAUC | joint − marginal net |
|---:|---:|---:|---:|---:|---:|
| 0.01 | +7 | +0.000107 | −3 | +0.000200 | +10 |
| 0.02 | +8 | +0.000197 | +3 | +0.000402 | +5 |
| 0.04 | +6 | +0.000305 | +19 | +0.000724 | −13 |

Mature vs pre-break net rate, joint: −0.00045 / −0.00060 / −0.00082.

## Verdict against the five preregistered conditions

1. *dominant-cell repair−damage not greater than the control* — joint is greater
   at 2 of 3 epsilons; the grid mean of `joint − marginal` net is **+0.67 pairs**
   out of ~15,600 sampled. A tie at noise level, **not an unambiguous trigger**.
2. *mature-vs-pre clearly negative (net rate ≤ −0.002 at every epsilon)* —
   observed −0.00045 / −0.00060 / −0.00082. **Not triggered.**
3. *no epsilon at which joint − marginal dominant-cell net is positive* — two of
   three are positive. **Not triggered.**
4. *rho > 0.90 with no advantage over the control* — rho is 0.162.
   **Not triggered.**
5. *legality/determinism not exact* — bitwise prefix invariance at `atol=0`
   passes on 12 real store series and 10 synthetic families; determinism,
   order-independence and the `n_online` gate pass. **Not triggered.**

**No binding Stage-1 kill condition fires. Proceed to the matched ninth-member
contract.** The goalposts are not moved in either direction: the screen is not
treated as a pass, and it is not converted into a kill it does not license.

## What the screen already says, stated plainly

The evidence is discouraging and it is worth writing down before Stage 2 so it
cannot be reinterpreted afterwards:

* Standalone alpha is weak — 0.5425 against RT-1320's 0.6403.
* The joint candidate is **worse than its own marginal control on every
  standalone cut**, and the marginal control buys a **larger** dominant-cell
  ΔAUC at every epsilon in the perturbation grid (mean +0.000442 vs +0.000203).
* On exactly the pairs the incumbent gets wrong, the joint block scores
  **0.471** — below chance. It is not a repair mechanism for the incumbent's
  errors; it is mildly anti-aligned with them.
* Decorrelation is real (rho 0.16) but `RT-1201` already established at rho 0.38
  that decorrelation alone converts into damage, not repair.

This is the `RT-1201` / `RT-1202` signature — different, weak, and not aimed at
the incumbent's errors. If Stage 2 returns `E2 − E1 ≈ 0`, the reading is CASE B
plus CASE C, not an inconclusive result.

## Note on the fold-0 authorization gate

The preregistration intended the fold-0 gate as a cheap stop before a five-fold
run. It cannot be that here, and the reason is structural rather than a change
of plan: the fold-pure `SCDF_NSEEN` calibration of a new member computes fold
0's calibration map from that member's out-of-fold predictions on folds 1–4, so
evaluating fold 0 *at all* requires all five fits. The gate is still applied
first and is still binding — a fold-0 failure stops the experiment and is
reported as a KILL rather than being overridden by the five-fold mean — but it
saves no compute.
