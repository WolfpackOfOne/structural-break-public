# STATE OF RESEARCH V2 — 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time)
**Research Director · wave 2 · 2026-08-19**

Wave 1 found the signal. Wave 2 was asked to make it reproducible, legal and
deployable, and to keep looking for alpha without contaminating validation.

**The single most important thing in this document:** the ensemble gain that wave 1
could only obtain with an illegal offline transform is **99.7 % recoverable by a
legal one**. The champion is now a system that can actually run under the
competition's inference interface, not an offline diagnostic.

Read `research/VALIDATION_V2.md` before quoting any number here. Every score below
is labelled **DISCOVERY**, **CONFIRMATION** or **DEPLOYABLE** and those words mean
what that document says they mean.

---

# REPRODUCIBILITY STATUS

| | |
|---|---|
| immutable checkpoint tag | `research-checkpoint-20260818-1` |
| checkpoint commit | `e98f5b969416585216502ed2d00075c6e86d7da9` |
| wave-2 branch | `research/wave2-2026` |
| manifest | `research/REPRODUCIBILITY_MANIFEST.json` |
| environment | Python 3.11.15, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, LightGBM 4.7.0, pyarrow 25.0.1, 2 vCPU / 7 GB, no GPU |
| streaming feature manifest | `1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced` (500 columns) |

**RT-100 reproduces with delta exactly 0.0.** From a genuinely clean
`git clone` of the tag into a separate directory, in a fresh container, with the
store, the folds and all 500 feature columns rebuilt from raw parquet and no
fitted object reused:

| | recorded | `RT-100R` | delta |
|---|---|---|---|
| mean OOF TS-AUC | 0.615103 | 0.6151034246587253 | **0.0** |
| pooled OOF | 0.615000 | 0.6149965423542226 | **0.0** |
| per fold | 0.62903 / 0.61061 / 0.62688 / 0.60747 / 0.60152 | identical | 0 |

`RT-123R` (the pairwise-ranking stream) also reproduces exactly. The fold file
regenerates to the identical SHA256 `6e114f80…`. This is stronger than required:
three of the four core dependencies changed major version since wave 1.

`nogit` is fixed — it was environmental (wave 1 ran outside a git checkout), and
every wave-2 row carries a real SHA. 66 of 78 wave-1 rows remain unattributable
and are flagged as such in `research/RDOF_LEDGER.md`.

**Correction on the record.** Mid-wave I briefly reported that `RT-121R`,
`RT-122R` and `RT-124R` failed to reproduce and read that as evidence about the
`nogit` rows. That was wrong: a file rewrite had silently failed and those runs
used reconstructed configurations. Every experiment whose configuration actually
matched reproduced exactly. The three variants are kept as legitimate diverse
streams, and the ledger says plainly what they are.

Full detail: `research/reports/reproducibility_v2.md`.

---

# VALIDATION STATUS

**The old lockbox is spent and was not touched in this wave.** `lockbox_touched`
is `no` on every wave-2 row. It has been observed twice, both times as a single
confirming scalar. One further use exists — a single pre-registered final
confirmation of one already-chosen deployable system — and it requires the
Research Director's written authorisation *before* the number is computed.

**There is no untouched training data left.** All 10,000 series have participated
in a scored evaluation. Wave 2 does not pretend otherwise. Confirmation is bought
with nesting, not with fresh data: for outer fold `k`, every selection step is
refit using only folds `≠ k`.

**Three alternative grouped fold partitions** (`alt1/alt2/alt3`) exist for
robustness only, each preserving break rate × τ quartile × history-length tertile
× online-length tertile, with ≈0.20 label agreement with the canonical partition.
They may be reported as a distribution and never used to select anything.

Selection-bias risks that remain, in order of size:

1. The 500-column bank and the seven stream configurations were chosen on the dev
   folds in wave 1. The lockbox bounded that cost at −0.0072 for the single model
   and −0.0116 for the 4-stream ensemble. Those bounds cannot be refreshed without
   spending the lockbox.
2. The screen store is a subset of the dev folds, so screen-driven choices are not
   independent of dev-fold scores.
3. Composition risk, newly quantified by the red team (below): if the organiser's
   test set has a different online-length or tail-heaviness mix, **±0.02–0.03 is on
   the table from composition alone** — larger than almost every win in the ledger.

---

# PROVEN SINGLE MODEL — `RT-100` / `RT-100R`

| | |
|---|---|
| architecture | 7 causal feature modules, 500 columns → one LightGBM booster, binary logloss |
| training | 1,000,000 sampled rows/fold, 600 trees, lr 0.05, 63 leaves, ff 0.5, seed 0 |
| **OOF TS-AUC (DISCOVERY)** | **0.61510**, pooled 0.61500 |
| per fold | 0.62903 / 0.61061 / 0.62688 / 0.60747 / 0.60152 (std 0.01091) |
| previous lockbox (wave 1, not re-run) | 0.60791 |
| streaming status | **bitwise-identical port, verified end to end** |
| runtime | 4.54 ms/point, 431 ms/series setup (contended 2-core box) |

---

# ORACLE ENSEMBLE — `RT-131`, and why it is not deployable

`infer()` receives a **single-pass iterator of series**, each consumed to
completion, with a single-pass online iterator inside it. The simultaneous
cross-section of all alive series at index `t` — the object RT-131's rank average
requires — **never exists during a legal run**. Organiser statements independently
confirm that cross-series state is defeated by `INFER_PARALLELISM` plus the
determinism recheck, and that persisting state between runs is disqualifying.

> **RT-131's 0.62541 is an ORACLE / DIAGNOSTIC upper bound. It is not a score and
> must never be called the champion.**

Evidence and the full interface contract: `research/reports/runner_semantics.md`.

---

# BEST DEPLOYABLE ENSEMBLE — `RT-150`

The oracle's rank transform does one thing: it puts every stream on a common,
time-conditional scale before averaging. That object can be **frozen from training
data** and applied per series:

    F_m(s | t) ≈ P[ model m scores below s among rows at online index ≈ t ]

estimated at 12 log-spaced time anchors, interpolated linearly in log t, each
anchor a 256-point quantile grid. Blend = mean of the seven calibrated scores.

| blend | OOF TS-AUC | % of the oracle gain recovered |
|---|---|---|
| `RT-100R` single model (legal floor) | 0.61510 | — |
| B1.1 raw probability mean | 0.62304 | 73.4 % |
| B1.2 logit mean | 0.62481 | 89.7 % |
| B1.3 global OOF CDF | 0.62552 | 96.2 % |
| B1.4 time-bucket CDF | 0.62584 | 99.2 % |
| **B1.5 smooth time-conditional CDF** | **0.62589** | **99.7 %** |
| oracle within-t rank average (ILLEGAL) | 0.62593 | 100 % |

Every map is cross-fitted: fold `k` is transformed by a map built on folds `≠ k`.
Fold `k`'s own score distribution never informs fold `k`'s ranking.

**This survives the checks that usually kill a result like this.**

- **Nested family selection (CONFIRMATION).** Choosing among the five families on
  all five folds would be selection on the evaluation data. Selecting the family
  on inner folds only, the inner folds pick `scdf` on **all five** outer folds and
  the held-out mean is **0.62589** — the confirmation number equals the discovery
  number, so nothing here is selection-inflated.
  Per fold: 0.63806 / 0.62066 / 0.63439 / 0.61685 / 0.61952. **Positive on 5/5
  folds** against `RT-100R`, deltas +0.0090 / +0.0101 / +0.0075 / +0.0094 / +0.0180.
- **Paired series-level bootstrap** vs `RT-100R`: **+0.01057, 95 % CI
  [+0.00763, +0.01320], 200/200 replicates positive.**
- **Deployment realism.** A 256-point grid per anchor reproduces the full
  empirical CDF to 5 decimals (0.62589 at q=256, q=1024, q=4096 and q=∞), so the
  payload is 3,072 floats per stream.

**Stream diversity, measured where the metric actually looks.** Within-timestep
rank correlations run **0.400–0.781**; global Pearson runs 0.594–0.930. The global
figure is inflated by a shared time trend the metric never scores. Marginal
contributions under the *deployable* blend (not the oracle):

| stream | mean without it | marginal | folds hurt by removal |
|---|---|---|---|
| `RT-121R` shape/dynamics | 0.62439 | **+0.00150** | 4/5 |
| `RT-124R` recursion-only | 0.62456 | **+0.00133** | 4/5 |
| `RT-122R` extra-trees | 0.62512 | +0.00078 | 4/5 |
| `RT-125R` GOSS | 0.62559 | +0.00030 | 4/5 |
| `RT-100R` full bank | 0.62606 | −0.00016 | 3/5 |
| `RT-120R` evidence-heavy | 0.62602 | −0.00013 | 2/5 |
| `RT-123R` pairwise | 0.62610 | −0.00020 | 3/5 |

Three streams — including the champion itself — have marginals that are slightly
negative. **Do not act on this.** All six non-trivial marginals are inside noise,
and wave 1 already showed that honest leave-one-fold-out subset selection loses
0.0008 against simply taking everything. The equal average has no parameters and
therefore no parameter variance. Keep all seven.

**Known approximation, stated plainly.** The deployed boosters are fitted on all
five dev folds while the calibration grids come from four-fold OOF. A full-data
model's scores are slightly sharper than its out-of-sample scores, so `F_m` is
marginally miscalibrated at deployment. It is conservative (it compresses rather
than inflates extremes), it points the same way for every stream so it largely
cancels in the mean, and the alternative — deploying 35 fold-models — costs 5×
the inference for an unmeasured gain. It is nonetheless a real approximation and
the first thing to test if the leaderboard disappoints.

---

# PRODUCTION SUBMISSION

## The shared streaming engine

```
stream ─▶ StreamCtx  (ONE expensive state update per observation)
              ├─▶ m00_core   151          ┐
              ├─▶ m01_seq     60          │  one 500-column vector,
              ├─▶ m02_dist    59          │  immutable ordering
              ├─▶ m03_dyn     60          ├─▶ 7 boosters read SLICES of it
              ├─▶ m04_resid   60          │
              ├─▶ m06_loc     60          │
              └─▶ m07_bayes   50          ┘
```

**Batch/stream parity: BITWISE, every production module.** Each module was ported
independently against the batch module as specification, then I verified the whole
thing myself rather than trusting the ports: the full 500-column vector is
bit-identical (`atol=0`, NaN==NaN) on real store series and on adversarial cases
(constant history, near-zero variance, t3 tails, 1e9 outliers, τ=0, 10-point and
999-point online segments). End to end,
**max |streaming prediction − batch prediction| = 0.0 over 12,727 points.**

Getting bitwise rather than "close" required care that is worth recording:
`ar_filter_causal` must be called on a *constant-length* causal window because
NumPy's BLAS `gemv` path depends on matrix height; `math.log10` differs from
`np.log10` in the last ulp on ~22 % of inputs; NumPy's pairwise summation differs
from sequential accumulation on signed zeros. Each is documented at the site.

## Runtime

| | before | after |
|---|---|---|
| all 7 modules, per observation | 2,524 µs | **1,887 µs** |
| `m00_core` | 2,008 | 223 (9.0×) |
| `m02_dist` | 706 | 360 (1.9×) |
| `m03_dyn` | 566 | 230 (2.5×) |
| `m07_bayes` | 441 | 364; its `fit_historical` 594 → 137 ms (4.3×) |
| booster predict, per point | 1,568 µs | **41 µs** (38×) |
| end-to-end through the runner harness | 14.0 ms/point | **4.5 ms/point** |

Every optimisation preserved bitwise parity; several were rejected at a
differential-fuzz gate *before* adoption precisely because they would not have.
The booster win alone was worth more than all the feature work: LightGBM's default
`predict` rebuilds a feature-name check and spins up a thread pool for a single
row. `validate_features=False, num_threads=1` is bit-identical and 38× faster.

Projected for a 10,000-series test set: ≈1.2 CPU-h of per-series setup + ≈2.7
CPU-h of stepping ≈ **3.9 CPU-hours, ≈1 h wall at `INFER_PARALLELISM=4`** — and
this is measured on a *contended* 2-core box, so the real figure is better.

**numba is not installed in this container**, so both the batch modules and the
streaming ports take the pure-Python fallback in `m01_seq` and `m07_bayes`. With
numba those two would drop by roughly an order of magnitude. It was deliberately
not installed: the feature cache was built without it, and installing it now would
risk changing values and invalidating the parity that everything rests on. It is
the largest single runtime lever left, and it needs its own parity sweep.

## Artifacts

| submission | what | size | status |
|---|---|---|---|
| **A** `submissions/A_rt100_streaming.ipynb` | streaming `RT-100`, one booster, 500 features | 2.8 MB | **PASS**, 4.54 ms/point |
| **C** `submissions/C_ensemble_deployable.ipynb` | `RT-150`: 7 boosters on ONE engine + frozen CDF calibration | 28.0 MB | **PASS**, 4.57 ms/point |
| B | reduced-feature variant | — | pending the cross-fitted feature-count study |
| D | experimental | — | not built |

That C costs 0.03 ms/point more than A is the whole argument for the shared-engine
architecture: seven models for the price of one feature pass.

The build is automated (`research/scripts/build_submission.py`). The notebook
embeds a **sha256-verified zip of the exact `src/sbr` tree the tests ran against**
plus the model directory, and unpacks both at run time — nothing is transformed,
so the notebook cannot drift from the tested code. `ProductionModel.load`
re-derives the feature manifest from the engine and **raises** if it disagrees
with the model's, so a mismatched pair fails loudly instead of scoring nonsense.

## Legality — what passed, and what could not be run

`research/scripts/test_submission_notebook.py` executes the built notebook in an
isolated directory **with this repository removed from `sys.path`** (asserted), then
runs it through a harness that enforces the contract more strictly than the
official runner:

| check | A | C |
|---|---|---|
| single-pass `datasets` and single-pass `x_online` (re-iteration raises) | ✅ | ✅ |
| readiness `yield`, then exactly one finite score per point, in [0,1] | ✅ | ✅ |
| determinism (two full runs compared exactly) | ✅ | ✅ |
| **order-independence** (series shuffled; every score unchanged) — the real test for illegal cross-series state | ✅ | ✅ |
| future poisoning (rewrite the tail; earlier scores bit-identical) | ✅ | ✅ |
| streaming == batch predictions | 0.0 over 12,727 pts | — |

Plus 78 production-contract tests, including
`test_n_online_is_never_a_feature` — the exact leak that invalidated every
pre-2026-06-08 prediction.

**`crunch test` itself has NOT been run.** `api.hub.crunchdao.com` is blocked by
this container's egress proxy, and the device VM has no network at all. This is
the one gate in the mandate that could not be closed here. The command to close
it is in the final answer.

---

# FEATURE-FAMILY ABLATIONS — and the control that makes them mean anything

Full-scale leave-one-module-out at the battery protocol (5 canonical folds,
400,000 training rows, champion hyperparameters, identical seed, identical rows).

**The naive table says five of seven modules are useless. The naive table is wrong.**

| module | cols | delta vs control |
|---|---|---|
| minus `m01_seq` | 60 | **+0.00203** |
| minus `m02_dist` | 59 | **+0.00158** |
| minus `m03_dyn` | 60 | **+0.00142** |
| minus `m00_core` | 151 | **+0.00064** |
| minus `m06_loc` | 60 | **+0.00058** |
| minus `m04_resid` | 60 | −0.00287 |
| minus `m07_bayes` | 50 | −0.00483 |

Five of seven removals *improve* the score. Read literally that is a discovery
that most of the feature bank is harmful. It is nothing of the kind: at 400k rows
a 500-column model is over-wide, so dropping ~60 columns helps **whatever they
are**. Leave-one-module-out confounds "these columns carried information" with
"this model prefers fewer columns".

**So I measured the nuisance directly** — random drops of the same size, same
protocol, same rows, same seed (`research/scripts/wave2_lomo_control.py`):

| control | delta vs control |
|---|---|
| random 60 columns, draw 0 / 1 / 2 | +0.00375 / +0.00002 / +0.00027 (mean **+0.00135**) |
| random 151 columns, draw 0 / 1 | +0.00191 / +0.00310 (mean **+0.00250**) |

Net module value = random-drop baseline − module-drop delta:

| module | cols | net value | below EVERY random draw of its size? |
|---|---|---|---|
| **`m07_bayes`** absorbing-state / BOCPD / e-values | 50 | **+0.0062** | yes (3/3) |
| **`m04_resid`** AR & volatility representations | 60 | **+0.0042** | yes (3/3) |
| **`m00_core`** calibrated multi-scale null | 151 | **+0.0019** | yes (2/2) |
| `m06_loc` online localisation | 60 | +0.0008 | 2 of 3 |
| `m03_dyn` dependence / spectral | 60 | −0.0001 | no |
| `m02_dist` PIT / occupancy | 59 | −0.0002 | no |
| `m01_seq` sequential bank | 60 | −0.0007 | no |

**Largest leave-one-out contribution: `m07_bayes`**, independently confirming
wave 1's screen finding that it is the best family per column. `m04_resid` is
second. Those two are resolvable; the bottom three are **indistinguishable from
deleting 60 random columns** at this protocol — which is a statement about the
experiment's resolution, not a verdict that they are worthless.

## The row-count confound, confirmed by re-running at the champion protocol

The two most surprising results were re-tested at 1,000,000 rows against `RT-100R`:

| | 400k rows | 1M rows (champion) | |
|---|---|---|---|
| minus `m01_seq` (`RT-232`, 440 cols) | +0.00203 | **+0.00123** (0.61633) | sign holds |
| minus `m00_core` (`RT-231`, 349 cols) | +0.00064 | **−0.00134** (0.61376) | **sign FLIPS** |

`m00_core` looks droppable at 400k rows and is worth +0.0013 at 1M. This is wave
1's operating rule reproduced exactly: *a result obtained on a data-starved model
does not transfer to one that is not data-starved.* It is also a warning about
this wave's own battery — every delta in the tables above is measured in the
starved regime and is biased toward "this module does not matter".

**Actionable:** dropping `m01_seq` holds up at champion scale — 440 columns,
+0.00123 (4/5 folds positive), and 281 µs/observation cheaper. That is a
deployment win (cost), not an alpha win: +0.0012 is inside the noise band this
ledger's own promotion rules set. It justifies **Submission B** as a cheaper
variant, not a new champion.

## Value per millisecond

Using the measured per-observation cost of each streaming module:

| module | µs/obs | net value | value per ms |
|---|---|---|---|
| `m04_resid` | 222 | +0.0042 | **+0.0189** — best |
| `m07_bayes` | 364 | +0.0062 | +0.0170 |
| `m00_core` | 263 | +0.0019 | +0.0072 |
| `m06_loc` | 326 | +0.0008 | +0.0025 |
| `m03_dyn` | 230 | −0.0001 | ~0 |
| `m02_dist` | 360 | −0.0002 | ~0 |
| `m01_seq` | 281 | −0.0007 | **worst** |

**Worst gain/runtime ratio: `m01_seq`** — it is the only module that is both
statistically indistinguishable from random columns and non-trivially expensive.
`m02_dist` is the most expensive module with no resolvable value, so it is the
next candidate if inference time ever becomes binding.

Caveat that applies to all of the above: these are single-seed, single-partition
measurements at a reduced protocol, and the two confirmations show the protocol
itself can flip a sign. Nothing here should be deployed on the strength of one
number.

---

# FOLD-ASSIGNMENT AND SEED STABILITY

## Alternative grouped fold partitions (RED TEAM 6: "the canonical folds are lucky")

Three alternative partitions of the *same* 8,000 dev series, each preserving
break rate × τ quartile × history-length tertile × online-length tertile, label
agreement with the canonical partition ≈0.20. Battery protocol, identical model.

| partition | mean TS-AUC | per fold |
|---|---|---|
| **canonical** (`RT-200`) | **0.61282** | 0.62567 / 0.60843 / 0.62018 / 0.60506 / 0.60475 |
| alt1 (`RT-221`) | 0.60471 | 0.60998 / 0.59722 / 0.61583 / 0.59924 / 0.60130 |
| alt2 (`RT-222`) | 0.61693 | 0.62223 / 0.62819 / 0.60842 / 0.60719 / 0.61861 |
| alt3 (`RT-223`) | 0.61072 | 0.59406 / 0.61959 / 0.60153 / 0.61426 / 0.62417 |
| alternatives | mean **0.61079**, sd **0.00499**, range 0.0122 | |

**RED TEAM 6 is REJECTED.** The canonical partition sits +0.0020 above the mean
of the three alternatives — 0.4 standard deviations, comfortably inside sampling
variation. It is not a lucky draw.

**But the partition itself carries about as much uncertainty as the folds do.**
Partition-to-partition spread (range 0.0122, sd 0.0050) is comparable to the
canonical fold-to-fold spread (std 0.0085). This must be added to the error bar on
every dev number in this project:

> **Any architecture difference below roughly ±0.01, measured on a single fold
> partition, is not distinguishable from the partition draw.**

That threshold retroactively covers most of the wins in `RESULTS.csv`, and it is
the single most useful calibration this wave produced for reading the ledger.
It is also why the deployable-ensemble result is trustworthy and the
`m01_seq`-removal result is not: +0.0106 with a bootstrap CI excluding zero and
5/5 folds positive clears the bar; +0.0012 does not.

## Seed stability (battery protocol, full bank, 5 canonical folds)

| seed | mean TS-AUC |
|---|---|
| **0** (preregistered) | 0.61282 |
| 1 | 0.61485 |
| 7 | 0.61584 |
| 42 | 0.61497 |
| 2026 | 0.61621 |
| | **mean 0.61494, sd 0.00118, min 0.61282, max 0.61621, range 0.0034** |

Two things worth saying out loud.

First, the **noise hierarchy is now measured**, and it is not what people usually
assume:

| source | sd |
|---|---|
| fold-to-fold (within one partition) | 0.0085 |
| **fold-partition draw** | **0.0050** |
| seed | 0.0012 |

Seed variation is the *smallest* of the three by a factor of four. Reporting a
seed sweep and calling a model "stable" while running one fold partition gets the
uncertainty budget backwards.

Second, **the preregistered seed 0 is the WORST of the five.** Every architecture
comparison in this project has been run at seed 0, so if anything the ledger is
mildly pessimistic about the architecture rather than seed-lucky. Nothing here was
selected on seed, and the final model keeps the preregistered seed.

---

# NEGATIVE CONTROLS

| control | result | verdict |
|---|---|---|
| whole-series label permutation | 0.50946 / 0.48804 (mean 0.4988) | **PASS** — chance |
| time-only model (`t_online`, `log_t_online` only) | **exactly 0.50000** on both folds | **PASS** — and exactly as theory demands: under TS-AUC any function of `t` alone is constant inside every comparison |
| future poisoning | bit-identical earlier rows | **PASS** |
| `n_online` leakage | AUC(score → `n_online` > median) = 0.488–0.514 across six timestep bands | **PASS** |
| prefix invariance at deep cuts (60/150/300/450) | bitwise, all 7 modules | **PASS** |
| random 60-column noise block | **INCOMPLETE** — OOM-killed after the first arm | rerun scripted |
| series-id / context permutation | not re-run; wave 1's derangement control stands (−0.057) | — |

The label-permutation control is whole-series, not row-level: each series inherits
another's *relative* τ, which preserves the label-trajectory shape. Row-level
permutation destroys the time structure and makes the control trivially easy.

---

# RED TEAM FINDINGS

## RED TEAM 4 — "the 500-column bank contains generator artifacts". **REJECTED.**

Every direct test came back at chance: a 500-column model trained on a *single*
timestep predicts `has_break` at OOF AUC **0.4952 (t=0)**, 0.4974 (t=4), 0.4915
(t=19); best univariate column 0.4792, exactly the null maximum for 500 draws;
history-only multivariate 0.5106; placebo relabelling of 4,025 no-break series
with fake τ gives TS-AUC **0.49904 ± 0.00649**; no cross-fold near-duplicate
series (nearest-neighbour label agreement 0.4942).

Three findings that matter anyway:

1. **`n_online` is worth more than our entire champion.** Because τ is ~uniform
   over the online segment, the ratio `(t+1)/n_online` alone scores **0.62948** —
   beating `RT-100` and even the oracle `RT-131` — and a rank blend with the
   champion reaches **0.66625**. It is illegal, it is what the organisers
   invalidated submissions over, and it is currently guarded only by convention.
   It is now a *measured* gate, not a code review.
2. **Gain importance is nearly useless as metric attribution: Spearman 0.163**
   against within-timestep metric value. Four pure clocks hold 6.0 % of total
   gain; `m07_bayes::bo_mean_rel` is gain rank 6 with clock-R² **0.9923**;
   `t_online`/`log_t_online` have within-timestep AUC **exactly 0.5000**. This
   means **RT-140's feature pruning ranked on the wrong quantity**, and so does
   the gain-ranked arm of this wave's feature-count study. Both are the honest
   reproduction of what was done before; neither is the right selector.
3. **A dormant trap.** Every history is standardised with `ddof=0`, so a `ddof=1`
   sample sd is a deterministic bijection of `n_hist` (spread 1.0001–1.0005). Inert
   today because `n_hist` is not label-predictive (0.493) — but any future feature
   that ratios a `ddof=1` online sd against a `ddof=1` historical sd silently
   encodes `n_hist`. The online segment is *not* separately standardised, which
   was the catastrophic case and is cleared.

**Composition sensitivity (not an artifact, but the honest answer to "will 0.615
transfer"):** champion OOF recomputed within strata gives 0.58167 / 0.60354 /
0.63976 across `n_online` tertiles and 0.62623 / 0.61685 / 0.58758 / 0.61986
across historical-kurtosis quartiles. **A different length or tail mix in the test
set is worth ±0.02–0.03 by itself** — comparable to the entire dev→lockbox haircut
and larger than most wins in the ledger.

Full report: `research/reports/redteam_generator_artifacts.md`.

## RED TEAM 2 — "RT-131's gain is entirely an artifact of illegal ranking".
**PARTIALLY CONFIRMED, then resolved.** The *mechanism* was confirmed: RT-131 as
computed is illegal. But the gain is not an artifact of the illegality — 99.7 % of
it survives a legal frozen transform. The right conclusion is not "the ensemble
was fake" but "the ensemble was computed the lazy way".

---

# PUBLIC IDEA MAP

`research/PUBLIC_IDEA_MAP.md` — 36 sourced entries, 43 URLs retrieved, a ranked
shortlist of 10 unimplemented ideas, and 13 publicly reported dead ends.

What the public record actually says:

- **The 2025 winners' code is essentially not public.** The forum thread titled
  "Winning Solutions" contains no technical content. The one rich account is a
  Medium post from the winning team (**Alphabot**): a two-level stack of eight
  XGB/RF models over **four independently authored feature blocks**, 90.65 % CV
  AUC, pushed to 91.99 % by absorbing **three competitors' models**. That +1.34 pp
  from foreign models is the largest single gain in the public record and is direct
  support for adding more, differently-wrong streams.
- Only 2nd place (**aParsecFromFuture**) published code: 6 transformations ×
  statistics + hypothesis tests = 2,408 features, SHAP/gain selection, TabPFN
  features, LightGBM.
- **"Inni Dynamics" returns nothing on the open web.** Unverified.
- **No 2026 real-time solution exists publicly anywhere.** A forum post indicates
  no public post-fix baseline exceeds **~0.61 TS-AUC** — our deployable 0.62589
  dev figure is at or above the public state of the art, with the usual caveat
  that our number is on our folds.
- Organiser confirmations we now rely on: cross-series state is defeated by
  parallelism + the determinism recheck; disk persistence between runs is
  disqualifying; the pre-June-8 invalidation was an `n_online` recovery leak.

Highest-value unimplemented ideas from the academic sweep:

1. **AR(p)-FOCuS (2026)** — exact maximum GLR over *all* change points for AR(p)
   data at O(log n) per step, strictly better than our dyadic grid, whose peak
   channels are already top-10 features.
2. **Run the existing detector bank on `x²` and on `x_t·x_{t−1}`** — the cheapest
   idea in the whole map, and it re-aims our machinery at the break types that
   actually exist here (scale and dependence) rather than the one that does not
   (location).
3. **E-detectors as O(1) recursions** (`M_n = L_n·(M_{n−1}+1)`) with a
   variance-adaptive `v(X) = (X−m)²`.
4. **Backward CUSUM**, which targets our measured late-τ weakness (0.571 vs 0.628).
5. **Robust BOCPD**, aimed squarely at our 17 %-transient problem.

---

# NEGATIVE RESULTS AND FAILURES IN THIS WAVE

- **The random-feature-block negative control OOM-killed** after one arm. Not run.
- **`RT-125R` could not be reproduced as configured**: LightGBM 4.7.0 rejects
  `boosting=goss` alongside the pipeline's default bagging, which wave 1's
  LightGBM accepted. A genuine environment-dependence finding.
- **My own process error**: a failed file write meant three stream rebuilds used
  reconstructed configurations, and I briefly drew a wrong conclusion about
  reproducibility from the resulting deltas. Corrected in `RDOF_LEDGER.md`.
- **`crunch test` could not be run** (blocked egress).
- **Whole research areas were not attempted at all** — see the next section. On a
  2-core box with no GPU, that was the correct trade, but it is a gap, not a
  finding.

---

# WHAT WAS NOT DONE, AND WHY

The mandate listed 35 items. This wave had **2 vCPUs, 7 GB RAM and no GPU**.
Not attempted, with no partial credit claimed:

| item | why |
|---|---|
| Alpha teams 10, 11 (deep temporal models, self-supervised encoders) | no GPU; a GRU/TCN/Mamba sweep is not a 2-core activity |
| Alpha teams 1–9, 12–15 (localisation v2, transient/permanent, 2025 transfer, hard negatives, synthetic augmentation, Bayesian v2, likelihood-ratio learning, ranking v2, multi-task, dependence v2, distribution v2, spectral v2, interaction search) | each needs a full-scale confirmation run (~40 min) per variant; the compute went to reproducibility, the streaming port and the deployability question, which were the mandate's stated Wave A and the submission blocker |
| Optuna hyperparameter search | correctly gated behind validation-v2, which only became operational in this wave |
| Red teams 1, 3, 5, 6, 7, 8 | 3 and 6 are partly covered by the LOMO and alt-fold battery; the rest need dedicated full-scale runs |
| Lockbox confirmation of anything | deliberately: the lockbox is spent and I had no authorisation |

---

# CURRENT TRUE CHAMPION

> **`RT-150` — seven LightGBM boosters over one shared incremental 500-column
> causal feature engine, blended by an equal average of frozen, cross-fitted,
> time-conditional CDF calibrations.**
>
> **Deployable OOF TS-AUC 0.62589** (nested confirmation 0.62589; per fold
> 0.63806 / 0.62066 / 0.63439 / 0.61685 / 0.61952), **+0.01057 over the single
> model with a 95 % bootstrap CI of [+0.00763, +0.01320] and 200/200 replicates
> positive**. 4.57 ms/point. Bitwise streaming parity. Passes every legality check
> that can be run without network access.
>
> It is reproducible, causal, legally deployable and streaming-tested. It has
> **not** been confirmed on data outside the dev folds, and the wave-1 dev→lockbox
> haircut for a comparable ensemble was −0.0116. A realistic expectation for the
> leaderboard is therefore **≈0.61, not 0.626**.
