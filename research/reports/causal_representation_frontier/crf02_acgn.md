# CRF-02 · ACGN — AMORTIZED CONDITIONAL GENERATIVE NULL

## VERDICT: **KILL** — abandon gate fired and the learned-null isolation gate failed

> **This report describes the CORRECTED run, `RT-1240`/`RT-1241`/`RT-1242`.**
> The earlier `RT-1237`/`RT-1238`/`RT-1239` run is **VOID and retired**: it
> silently loaded a 24-series, 1-epoch, `HWIN = 128` null written by a unit test
> instead of the preregistered one. See `EXPERIMENT_ID_MAP.md` and §7 below. The
> execution preregistration is **unchanged** — the defect was that the run did not
> execute it.

| | |
|---|---|
| candidate | `RT-1240` |
| C1 mandatory control — **fixed** null (AR(5) + history residual ECDF) | `RT-1241` |
| C2 mandatory control — **deranged** `h_i` | `RT-1242` |
| shared-8 diagnostic (declared in advance, no RT id) | — |
| program preregistration | `CRF_PROGRAM_PREREG.md` @ `85d121f` |
| execution preregistration | `CRF02_EXECUTION_PREREG.md` @ `9a3d3c7` (unchanged) |
| preflight / correction / void | `ad6ecd7` / `89149d5` / `fb24c39` |
| folds trained | fold 0 only |
| compute | 0.411 h |
| lockbox / test / production / submission | none |

---

## 1. THE GATES

| gate | required | measured | |
|---|---:|---:|---|
| cheap abandon (§0.2, checked first) | standalone ≥ 0.600 **or** ρ > 0.60 | **0.559140** at ρ **+0.2887** | **FIRES** |
| **learned-null isolation** (§2.8) | candidate − C1 ≥ **+0.0005** | **−0.021446** whole, **−0.020337** dominant | **FAIL** |
| **derangement** (§2.8, `m05_ctx` rule) | candidate − C2 ≥ **+0.0005** | **+0.003377** whole, **+0.008391** dominant | **PASS** |

Two kills and one pass. `CRF-02 = KILL`.

`marginal_vs_clone` is **NOT COMPUTED** for any arm: the abandon gate fired, folds
1–4 were deliberately never trained, so no fold-pure five-fold OOF exists. The
isolation comparisons are on **standalone fold-0 TS-AUC**, exactly as
`CRF02_EXECUTION_PREREG.md` §8 stage S1 preregistered for this case. Both mandatory
controls were run **before** the gate was read, deliberately, so the scientific
comparison would survive a headline failure.

---

## 2. ALL ARMS, CANONICAL FOLD-0 SAMPLE

| | **`RT-1240` learned** | **`RT-1241` FIXED** | **`RT-1242` deranged `h_i`** | shared-8 diag | RT-600 |
|---|---:|---:|---:|---:|---:|
| standalone whole-fold TS-AUC | 0.559140 | **0.580586** | 0.555763 | 0.551432 | 0.638276 |
| standalone dominant-cell TS-AUC | 0.579573 | **0.599909** | 0.571182 | 0.568920 | 0.677710 |
| within-`t` ρ vs RT600 | +0.2887 | +0.3683 | +0.2438 | +0.2623 | — |
| whole-fold repairs / damage / **net** | 9,723 / 15,356 / **−5,633** | 9,510 / 13,529 / **−4,019** | 9,582 / 16,164 / **−6,582** | 9,677 / 15,855 / **−6,178** | — |
| dominant repairs / damage / **net** | 6,991 / 12,264 / **−5,273** | 6,919 / 11,049 / **−4,130** | 6,935 / 13,313 / **−6,378** | 6,829 / 12,983 / **−6,154** | — |
| mature-vs-never **net** | **−5,176** | **−4,403** | **−6,492** | **−6,228** | — |
| mature-vs-pre-break **net** | **−6,734** | **−3,419** | **−6,790** | **−7,676** | — |
| unique repair coverage vs `RT-401` | 0.3417 | 0.3392 | 0.3378 | 0.3371 | — |
| repair Jaccard vs `RT-401` | 0.1706 | 0.1678 | 0.1727 | 0.1617 | — |
| **pre-break** damage rate on RT600-correct | **0.3929** | 0.3073 | 0.4043 | 0.4163 | cap 0.0150 |

RT-600 sentinel reproduced exactly: fold-0 `E0 = 0.638276`, `E1 = 0.638586`.

Generative null: 6,383 training-fold series' break-free histories, 10 epochs, pinball
`0.364706 → 0.243881`, 1,364.9 s, state `e58bc20e05d9…`, provenance fingerprint
recorded and verified.

---

## 3. THE CONDITIONING WORKS. THE AMORTIZATION STILL LOSES.

**The derangement control passes.** Permuting `h_i` across series within fold costs
the candidate `−0.003377` whole-fold and `−0.008391` on the dominant cell. So the
8-dimensional history bottleneck **does** carry usable series-specific information,
and the model **is** genuinely conditioning on it rather than memorising a series
identifier. This is a real, if small, positive finding and it is reported as one.

**And the fixed null still wins by 0.021446.** `RT-1241` — the project's existing
apparatus, AR(5) coefficients plus a 256-knot history residual ECDF, fitted per
series on that series' own history — feeds **identical** downstream statistics into
an **identical** ranking head and reaches 0.580586 against the learned null's
0.559140.

That is the finding, and it is sharper than a failed derangement would have been.
The learned null is not broken and it is not memorising: it is **conditioning
correctly and still losing**, because an 8-float amortized bottleneck simply cannot
carry what a per-series fit gets for free. The fixed null has, in effect, five AR
coefficients plus a 256-knot empirical residual distribution **per series**, paid for
by that series' own break-free history at zero generalisation cost — the history is
complete at `t = 0` and contains no label. Series heterogeneity in this data is
large; that is the whole reason the project's foundation is per-series historical
calibration.

**Amortizing the null across series loses more than the learned nonlinearity
gains.** Not because the bottleneck is inert — it demonstrably is not — but because
8 floats is the wrong order of magnitude for the job.

### The shared-8 diagnostic, declared in advance

The candidate's head refit on the eight features the fixed-null control also has
(dropping `lshift` and `peak_lshift`, which have no fixed-null analogue) reaches
0.551432, **below** the 10-feature candidate's 0.559140. So the two encoder-derived
features do help the candidate — and the isolation conclusion is **conservative in
the candidate's favour**: measured on the eight shared features alone the candidate
loses to the fixed null by **−0.029154**, worse than the headline −0.021446. The
gate was not flattered by a feature-count advantage; if anything it was generous.

---

## 4. WHAT THIS CLOSES

`CRF_PROGRAM_PREREG.md` §2.9, written before any number existed:

> If ACGN fails, and specifically if it does not beat its **fixed**-null control by
> +0.0005, then learned generative nulls are closed for this problem: the per-series
> historical calibration the project already ships is the right null, and the
> never-break false-positive mass is not a null-misspecification problem.

It lost to its fixed-null control by **0.0214**. So:

* **learned amortized generative nulls are closed** for this problem. The per-series
  historical calibration the project already ships **is** the right null — now
  measured against a matched learned alternative, not assumed.
* **the never-break false-positive mass is not a conditional-null misspecification
  problem** in CRF-02's sense. A strictly more flexible, correctly-conditioned null
  was built and it did not help.
* **amortization, specifically, is what fails.** The derangement result rules out
  "the conditioning was inert" as the explanation, which is what makes this a
  statement about amortization rather than about this particular implementation.

Combined with CRF-01 this closes **H-A**, **H-B** and **H-E**, leaving **H-D**.

---

## 5. WHAT WAS NOT DONE

No tuning. No `CRF-02b`. No MDN, no flow, no wider bottleneck, no deeper head, no
different quantile grid, no second epoch count, no seed re-roll. Forbidden by
`CRF_PROGRAM_PREREG.md` §0.8/§0.9 and unsupported by the evidence: the null's pinball
loss converged cleanly, so it was not under-trained.

A wider bottleneck is the one thing this result genuinely motivates — and it is a
**different experiment**, which this program does not authorise. It is recorded in
`CRF_FINAL.md` as a hypothesis, not run.

**CRF-03 does not open.** §3.1 requires CRF-01 or CRF-02 to reach
`marginal_vs_clone ≥ +0.0015` on fold 0 **and** pass its mandatory isolation control.
Both primaries are KILL and neither produced a marginal. The program terminates.

---

## 6. RUNTIME AND STREAMING COST

Pretraining 1,364.9 s (fold 0). Signal generation 177 s learned / 4 s fixed / 173 s
deranged. Ranking heads ≈ 33 s each. Total 0.411 h.

Streaming state per series ≈ **21 KiB**: `h_i` (8 floats) + `p_i` (32 floats) +
AR(5) `φ` + a 256-knot residual ECDF + an 8 × 253 float32 ring buffer + five `O(1)`
accumulators and their peaks. `fit_history(H)` is one encoder pass over the last
1024 history points; `update(x_t)` is one incremental causal pass, one 21-knot head
evaluation, one PIT interpolation and five `O(1)` updates.

---

## 7. THE VOID RUN, AND HOW IT WAS CAUGHT

The first filed CRF-02 result (`RT-1237`/`RT-1238`/`RT-1239`, commit `9a5ecc0`) was
**void**. Its fold-0 run silently loaded a null checkpoint written by a unit test —
`fit_series = 24`, `pretrain_epochs = 1`, `HWIN = 128`, pinball `2.1786` — instead of
the preregistered 6,383-series, 10-epoch, `HWIN = 1024` null.

**How it was caught.** Routine compute accounting for `CRF_FINAL.md` reported the
CRF-02 arm at 0.026 h. That was implausible against the 1,284.6 s the pretraining had
visibly taken on the earlier attempt, and the recorded `pretrain_runtime_s = 0.2`
gave it away.

**Why the existing check missed it.** The checkpoint loader compared the stored
`state_sha256` against its own recorded value. That proves **integrity, not
provenance** — a toy null is perfectly self-consistent.

**Why it mattered scientifically, not just procedurally.** The void run's numbers
would have supported a *different and wrong* mechanistic conclusion. With the toy
null, the derangement control **tied** the candidate (`−0.000981`), which reads as
"the history bottleneck carries no usable information" — the `m05_ctx` failure. With
the real null the derangement gate **passes** (`+0.003377`), and the correct reading
is the opposite: the conditioning works and amortization loses anyway. The verdict
(KILL) is the same; the reason is not.

**Fixes, both asserted in code and covered by tests.** Checkpoints now carry a
provenance fingerprint over fold, seed, epochs, `HWIN`, batch, widths, level count,
lr, wd, fit-series count and the **sha256 of the sorted fit-series ids**, and a
mismatch raises `CHECKPOINT PROVENANCE MISMATCH` and stops the run rather than
falling through. Every test now gets a throwaway cache directory.
`test_ckpt_provenance_mismatch_is_refused_loudly` reproduces the exact defect and
requires the refusal. **27 gates pass.**

The three void ids are **retired, not recycled**. Their `RESULTS.csv` rows are left
exactly as written — the ledger is append-only — and are marked void in
`EXPERIMENT_ID_MAP.md`. `research/oof/RT-123{7,8,9}.npy` are the toy null's output
and are equally void. **CRF-01 is unaffected**: `crf01_nncsr.py` writes no
checkpoint, its tests never call `emit` or `build_channels`, and its emitted
metadata records the full 6,383-series, 20-epoch runs.

---

## 8. REPRODUCTION

```bash
"$TORCH"   research/scripts/crf02_acgn.py --fold 0
"$NOTORCH" research/scripts/crf01_integrate.py --stage abandon --arms acgn \
             --out .../crf02_abandon.json
"$NOTORCH" research/scripts/crf01_integrate.py --stage report_fold0 \
             --arms acgn,fixed_null,deranged,shared8_diag --out .../crf02_report_fold0.json
```

Evidence: `crf02_acgn.json`, `crf02_abandon.json`, `crf02_report_fold0.json`,
`cache/crf02/fold0_meta.json`, `research/oof/RT-124{0,1,2}.npy` (fold-0 rows finite,
folds 1–4 NaN, lockbox empty), null checkpoint `cache/crf02/null_fold0.pt`
(state `e58bc20e05d9…`, fingerprint verified).
