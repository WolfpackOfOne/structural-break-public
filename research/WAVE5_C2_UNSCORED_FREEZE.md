# WAVE 5 — C2 UNSCORED FEATURE-BANK FREEZE

**Frozen 2026-08-22 on `research/wave5-alpha` at parent SHA `d39f2b28da9025a60e7c79c130bb1e25f1d2b034`.**

This is a scientific checkpoint. Every module below was designed, built, audited
and certified **without any of them ever having been scored on competition
data**. That is the point of the checkpoint: when C3 runs, the mechanisms were
fixed before the evidence, so the evidence can falsify them.

> **TRAINING RUNS = 0 · NEW TS-AUC VALUES OBSERVED = 0 · LEADERBOARD SUBMISSIONS = 0**

Column counts measured live at freeze time: m10_perm=24 m11_focus=21 m12_deplr=14 

---

## 1. FROZEN MODULES

| module | cols | ms/series | state | complexity / step | nearest existing | hypothesis | internal control |
|---|---:|---:|---|---|---|---|---|
| `m10_perm` | 24 | 40.5 | O(LMAX) window buffers | O(W) per window rung | `m00_core` (window means) | arrangement statistics distinguish a transient burst from a persistent shift; window means cannot, being arrangement-invariant | `md` (mean−median) vs `cn` (concentration) — two independent routes to the same claim |
| `m11_focus` | 21 | 62.3 | O(1) cumsums | O(LMAX) amortised | `m01_seq` (dyadic max-GLR) | an ADAPTIVE online baseline removes the persistent online-vs-historical offset every historical-null detector carries | `hg` (exact search, historical baseline) vs `ag` (adaptive) — if `hg` wins, the gain is search, not baseline, and H1 falls |
| `m12_deplr` | 14 | 55.1 | O(1) cumsums | O(LMAX) amortised | `m04_resid` (`e_acf1`) | the AR COEFFICIENT can move without the autocovariance moving, and vice versa; `m04` freezes its coefficients and never asks | `dlr` (coefficient ratio) vs `plr` (lag-product mean) — identical scan, only the dependence model differs |

**Total: 59 new columns, 157.9 ms/series if all three ship together**, against a
~80 ms/module engineering screen and RT-600's measured 6.8× cloud headroom.
Worst-case state growth is bounded by `LMAX = 256` in `m11_focus` and
`m12_deplr`; `m10_perm` holds trailing windows bounded by its largest rung (256).

### Module hashes at freeze

```
m10_perm.py             e35a2cc09e3e84b866ecba76c99dc7c74c221d9c4b80f063fe6c4379cdbb2d60
m11_focus.py            25d5a6b39b709f0165c019c587af9bd2645b8eda2ffd2792573c8bffca9e3757
m12_deplr.py            fb68fc3ce9962986923d719a29678f50836c127b989b4c93a7868542f85557e4
wave5_hard_negative.py  c97a654f1821e642f0e98cf90a6f713ccc182254249d4c2f706a6eea3903f79f
```

## 2. TRAINING-PROTOCOL CANDIDATE (not a module)

| item | status |
|---|---|
| hard-negative mining | pre-registered, coded, **fold-purity proven by test**, unscored |
| constants | `LAMBDA = 1.0`, `TOP_Q = 0.20`, weights bounded [1, 2], positives never reweighted |
| arms | `RT-540` uniform control · `RT-541` weighting · `RT-542` oversampling |

## 3. TEST STATE

```
99 passed in 44.20s
```

Covering, for every module: bitwise prefix invariance at `atol = 0.0` (6 length
profiles × 4 break kinds × 7 truncation points), no `n_online` access, single-pass
streaming, per-series state reset, no cross-series state, finite outputs, runtime,
a synthetic mechanism test, a synthetic falsification test, and a redundancy check
against the nearest existing module. `m11_focus` and `m12_deplr` additionally
prove their vectorised changepoint scans equal an O(n²) brute-force reference at
0.0e+00 — those loops are optimisations, not approximations.

## 4. WITHDRAWN BEFORE ANY SCORE

| idea | why | cost |
|---|---|---|
| mean-vs-robust-mean contrast | **algebraically dead.** `(x−mu)/sd` and `(x−med)/mad` are affine images; null standardisation removes any affine map exactly. Measured residual 1.8e-15 on t(3); the median version measures 4.0. Regression test prevents its return | 0 |
| generic robust distribution distances | `m02_dist` already emits Wasserstein, energy, KS, CvM, JS, Hellinger, TV, chi-square against a length-matched null. A new module would re-state an existing channel — the failure mode behind `tl_` (−0.0041) and `rk_` (−0.0027) | 0 |
| residual CUSUMSQ (`m13_rcsq`) | incumbent `cmb_e_abs` beats it on its OWN mechanism (|d| 1.979 vs 1.142), bootstrap CI excludes zero, and 94% of the magnitude channel is already determined by `m04` alone. The change-age channel is non-redundant but carries no signal (|d| 0.054) | 0 |
| absorbing-state BOCPD (`m14_absorb`) | `m07_bayes` **is** an absorbing-state model and already emits all nine proposed outputs; it integrates over every tau exactly rather than maximising | 0 |

## 5. BINDING PRE-COMMITMENTS CARRIED INTO C3

1. **W5-E1 mixed-result rule.** If `m10_perm` raises aggregate TS-AUC but the
   **0–5 and 5–10 age buckets do not improve**, the intended mechanism is
   **REJECTED**. No rescue by aggregate.
2. **W5-E2 internal control.** If `m11_focus`'s exact-search arm gains and the
   adaptive-baseline arm does not, **H1 is rejected** and the cheaper lesson
   belongs in `m01_seq`, not in promoting the module.
3. **W5-E3 internal control.** If `plr` carries `m12_deplr`'s gain and `dlr`
   does not, **H1 is rejected**; the win would be generic max-over-tau and belongs
   as a channel in `m11_focus`.
4. **Seed-clone bar on everything.** No family — feature or training-protocol —
   is promoted on a blend delta that a same-strength seed clone also produces.
5. **Age-bucket reporting is mandatory**, not an appendix, for every candidate.
6. **The 0.6268 leaderboard score selects nothing.** It is an external
   calibration point and is not an optimiser.

## 6. WHAT IS STILL BLOCKED

No competition data of either year is present in this container. The 2026 store
(`cache/store`) is absent; `crunchdao.com` is refused by network policy. The 2025
reproduction's dependency gate now passes (polars, lightgbm, shap all installed
and verified) but R25-040/050 additionally need TabPFN weights, which are
unobtainable here — `huggingface.co` is blocked and the 8.4.0 checkpoint is gated.

**C3 cannot begin until the store arrives.** `research/scripts/wave5_ingest_store.py`
verifies it byte-exact against the recorded manifest before any experiment runs.

---

## 7. AMENDMENT — PRE-C3 METRIC-ALIGNMENT CORRECTION (2026-08-22)

Made **before any Wave-5 TS-AUC was observed** (training runs = 0 at the time).
The freeze on the three feature modules is **unaffected**: `m10_perm`,
`m11_focus` and `m12_deplr` are byte-identical to their C2 hashes above, and
`LMAX`, window grids, thresholds, column counts and internal-control definitions
are untouched.

Two things changed, both outside the feature bank:

1. **Hard-negative hardness is now time-conditional**, matching the metric: a
   negative's hardness is the fraction of positives **at its own online index**
   that it outranks, with mid-rank tie handling to match `sbr.metric`. The
   previous global ranking could rank negatives exactly backwards. Constants,
   arms and falsification conditions unchanged. Full amendment in
   `research/reports/wave5_hard_negative_protocol.md`.

2. **Package-matrix wording corrected.** The matrix previously implied that
   `requirements.txt` alone settles deployability and that no whitelist exists.
   Absence of a whitelist in *our record* is not evidence one does not exist, so
   everything RT-600 has not shipped is now
   **ENGINEERING-COMPATIBLE / CRUNCH WHITELIST STATUS UNVERIFIED** rather than
   "likely deployable". No research conclusion depends on this.

Neither correction touches a scored quantity, because none exists.
