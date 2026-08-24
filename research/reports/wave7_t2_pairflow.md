# WAVE 7 — T2 (RT-995) PAIR-FLOW AND DIVERSITY

Fold 0. `wave8_common.pair_repair_stats` (raw, uncalibrated scores — this
diagnostic explains the mechanism behind the calibrated ensemble result in
`wave7_t2_ensemble_integration.md`, it is not itself the binding number).

## Same-t pair repair/damage vs the E0 (seven-specialist) blend

| candidate | repairs | damage | net pair lift | total pairs sampled |
|---|---:|---:|---:|---:|
| T2 (`RT-995`) | 708 | 723 | **−15** | 9860 |
| seed clone (`RT-401`) | 665 | 731 | **−66** | 9860 |

Neither T2 nor the seed clone repairs more same-t inversions than it
introduces against the raw E0 blend — both are net-negative on this
uncalibrated diagnostic. T2's net lift (−15) is less bad than the clone's
(−66), consistent with T2 contributing a small positive over the clone in
the calibrated ensemble test, but neither shows the clear
repair-dominant signature a genuinely new information source would.

## Within-t rank correlation

| stream | corr with T2 |
|---|---:|
| `RT-300` | 0.9122 |
| `RT-410` | 0.8742 |
| `RT-411` | 0.8557 |
| `RT-412` | 0.8830 |
| `RT-413` | 0.7147 |
| `RT-414` | 0.8785 |
| `RT-415` | 0.9032 |
| E0 blend (all seven) | 0.6740 |
| seed clone `RT-401` | **0.9099** |

## Reading

T2 correlates with the seven individual specialists at 0.71–0.91 — the same
order of magnitude as `RT-401`'s own correlation with T2 (0.91), and
`RT-401` is by construction an exchangeable clone carrying no new
information. T2 does not show the low-correlation, independent-repair-
reservoir signature that would indicate a genuinely new information source
(the pattern Wave 8's redundancy matrix looked for and did not find among
its own five mechanisms either — see
`research/reports/wave8_futureaware_redundancy.md`). T2's correlation with
the *blend* (0.674) is lower than with any individual specialist only
because the blend averages out specialist-specific noise, not because T2
carries orthogonal signal — if it did, `E2 − E1` would be materially larger
than `E1 − E0`, and it is not (+0.00024 vs +0.00031).

This is consistent with, and explains, the ensemble-integration verdict:
T2's standalone edge over `T0` (+0.00943) is real (it clears its own
promotion legs against a matched single-model control) but it is largely the
**same** information the seven specialists (via their own overlapping
feature banks) already extract, refined into a cleaner single-model score —
not new information the ensemble was missing.
