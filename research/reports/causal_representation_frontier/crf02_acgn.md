# CRF-02 · ACGN — AMORTIZED CONDITIONAL GENERATIVE NULL

## VERDICT: **KILL — on three independent mandatory grounds**

| | |
|---|---|
| candidate | `RT-1237` |
| C1 mandatory control — **fixed** null (AR(5) + history residual ECDF) | `RT-1238` |
| C2 mandatory control — **deranged** `h_i` | `RT-1239` |
| shared-8 diagnostic (declared in advance, no RT id) | — |
| program preregistration | `CRF_PROGRAM_PREREG.md` @ `85d121f` |
| execution preregistration | `CRF02_EXECUTION_PREREG.md` @ `9a3d3c7` |
| preflight / correction | `ad6ecd7` / `89149d5` |
| folds trained | fold 0 only |
| compute | 0.437 h |
| lockbox / test / production / submission | none |

---

## 1. THE THREE GATES THAT FAILED

| gate | required | measured | |
|---|---:|---:|---|
| cheap abandon (§0.2, checked first) | standalone ≥ 0.600 **or** ρ > 0.60 | **0.527807** at ρ **+0.2212** | **FIRES** |
| **learned-null isolation** (§2.8) | candidate − C1 ≥ **+0.0005** | **−0.052779** whole, **−0.060600** dominant | **FAIL** |
| **derangement** (§2.8, the `m05_ctx` rule) | candidate − C2 ≥ **+0.0005** | **−0.000981** whole, **−0.000845** dominant | **FAIL** |

Any one of these is a kill. All three fired.

`marginal_vs_clone` is **NOT COMPUTED** for any arm: the abandon gate fired, folds
1–4 were deliberately never trained, and no fold-pure five-fold OOF exists to
calibrate against. The isolation comparisons are therefore reported on **standalone
fold-0 TS-AUC**, which is exactly what `CRF02_EXECUTION_PREREG.md` §8 stage S1
preregistered for this case — both controls were run *before* the gate was read
precisely so the scientific comparison survives a headline failure.

---

## 2. ALL ARMS, CANONICAL FOLD-0 SAMPLE

| | **`RT-1237` learned null** | **`RT-1238` FIXED null** | **`RT-1239` deranged `h_i`** | shared-8 diag | RT-600 |
|---|---:|---:|---:|---:|---:|
| standalone whole-fold TS-AUC | 0.527807 | **0.580586** | 0.528788 | 0.531864 | 0.638276 |
| standalone dominant-cell TS-AUC | 0.539309 | **0.599909** | 0.540154 | 0.533872 | 0.677710 |
| within-`t` ρ vs RT600 | +0.2212 | +0.3683 | +0.2242 | +0.2037 | — |
| whole-fold repairs / damage / **net** | 9,666 / 16,832 / **−7,166** | 9,510 / 13,529 / **−4,019** | 9,631 / 16,734 / **−7,103** | 9,729 / 17,607 / **−7,878** | — |
| dominant repairs / damage / **net** | 6,815 / 13,531 / **−6,716** | 6,919 / 11,049 / **−4,130** | 6,786 / 13,483 / **−6,697** | 6,919 / 14,902 / **−7,983** | — |
| mature-vs-never **net** | **−7,045** | **−4,403** | **−7,049** | **−8,301** | — |
| mature-vs-pre-break **net** | **−6,868** | **−3,419** | **−6,800** | **−7,254** | — |
| unique repair coverage vs `RT-401` | 0.3313 | 0.3392 | 0.3294 | 0.3435 | — |
| repair Jaccard vs `RT-401` | 0.1731 | 0.1678 | 0.1740 | 0.1583 | — |
| **pre-break** damage rate on RT600-correct | **0.4144** | 0.3073 | 0.4117 | 0.4217 | cap 0.0150 |

RT-600 sentinel reproduced exactly: fold-0 `E0 = 0.638276`, `E1 = 0.638586`.

Generative null: pinball loss `0.364706 → 0.243881` over 10 epochs, state
`e58bc20e05d9…`, 1,284.6 s, fitted on 6,383 training-fold series' break-free
histories only.

---

## 3. THE FIXED NULL BEATS THE LEARNED NULL BY 0.0528

This is the headline finding and it is not close. `RT-1238` — the project's existing
fixed apparatus, AR(5) coefficients plus a 256-knot history residual ECDF, both
fitted per series on that series' own history — feeds **identical** downstream
statistics into an **identical** ranking head and reaches 0.580586 against the
learned null's 0.527807.

**Why.** The candidate's own C2 result explains it. Permuting `h_i` across series
moves the score by `−0.00098`, i.e. by nothing: the 8-dimensional history bottleneck
carries **no usable series-specific information at all**. The learned null is
therefore, in effect, a **population-average** predictive distribution applied to
every series alike. The fixed null is a **per-series** fit with, in effect, hundreds
of free parameters per series — five AR coefficients and a 256-knot empirical
residual distribution — paid for by that series' own history at zero generalisation
cost, because the history is complete at `t = 0` and is break-free by construction.

Series heterogeneity in this data is large. That is the whole reason the project's
foundation is per-series historical calibration. An 8-float bottleneck is not a
wide enough channel to recover what a per-series fit gets for free, and amortizing
across series therefore *loses* information rather than adding any.

Note the failure is **not** the `m05_ctx` failure mode in its usual form. C2 did not
beat the candidate because `h_i` was a series *identifier* the model had memorised;
it beat it because `h_i` was doing essentially nothing, so destroying it cost
nothing. Both readings kill the arm under §2.8, but the mechanism matters for what
it closes.

The declared shared-8 diagnostic confirms the isolation gate was not flattered: the
candidate refit on the 8 features the fixed-null control also has reaches 0.531864,
**above** the 10-feature candidate. The two encoder-derived features the control
lacks were not carrying the comparison — they were slightly harmful.

---

## 4. WHAT THIS CLOSES

`CRF_PROGRAM_PREREG.md` §2.9 wrote the interpretation before the number existed:

> If ACGN fails, and specifically if it does not beat its **fixed**-null control by
> +0.0005, then learned generative nulls are closed for this problem: the per-series
> historical calibration the project already ships is the right null, and the
> never-break false-positive mass is not a null-misspecification problem.

It did not beat its fixed-null control by +0.0005. It lost to it by **0.0528**. So:

* **learned generative nulls are closed** for this problem. Amortization across
  series is the wrong move: the per-series historical calibration the project
  already ships **is** the right null, and this is now measured against a matched
  learned alternative rather than assumed.
* **the never-break false-positive mass is not a conditional-null misspecification
  problem** in the sense CRF-02 hypothesized. The D3 signature — heavy tails, long
  memory, few excursions then wandering — is not explained by the null being
  mis-specified, because a strictly better-specified null was built and it did not
  help.
* **`h_i`-style amortized conditioning is closed** as a route: an 8-dimensional
  learned bottleneck over the history is not a substitute for fitting that history
  directly.

Combined with CRF-01, this closes **H-A** (representation saturation), **H-B**
(objective mismatch) and **H-E** (null misspecification), leaving **H-D**.

---

## 5. WHAT WAS NOT DONE

No tuning. No `CRF-02b`. No MDN, no flow, no wider bottleneck, no deeper head, no
different quantile grid, no second epoch count, no seed re-roll. All are forbidden
by `CRF_PROGRAM_PREREG.md` §0.8/§0.9, and none is supported by the evidence: the
failure is not that the learned null was under-trained — its pinball loss converged
cleanly — but that amortization discards per-series information the fixed null
keeps for free. A bigger bottleneck is a different experiment, and the program does
not authorise one.

**CRF-03 does not open.** `CRF_PROGRAM_PREREG.md` §3.1 requires CRF-01 **or** CRF-02
to reach `marginal_vs_clone ≥ +0.0015` on fold 0 **and** pass its own mandatory
isolation control. Both are KILL; neither has a marginal at all; CRF-02 failed both
of its isolation controls. The program terminates. See `CRF_FINAL.md`.

---

## 6. REPRODUCTION

```bash
"$TORCH"   research/scripts/crf02_acgn.py --fold 0        # 1284.6 s null + 4 heads
"$NOTORCH" research/scripts/crf01_integrate.py --stage abandon --arms acgn \
             --out .../crf02_abandon.json
"$NOTORCH" research/scripts/crf01_integrate.py --stage report_fold0 \
             --arms acgn,fixed_null,deranged,shared8_diag --out .../crf02_report_fold0.json
```

Evidence: `crf02_acgn.json`, `crf02_abandon.json`, `crf02_report_fold0.json`,
`cache/crf02/fold0_meta.json`, `research/oof/RT-123{7,8,9}.npy` (fold-0 rows finite,
folds 1–4 NaN, lockbox empty). Null checkpoint `cache/crf02/null_fold0.pt`, state
sha256 `e58bc20e05d9…`.
