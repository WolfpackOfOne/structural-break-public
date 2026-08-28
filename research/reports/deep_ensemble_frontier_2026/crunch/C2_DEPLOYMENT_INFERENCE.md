# C2 — DEPLOYMENT INFERENCE BENCHMARK (RT-1257, and the k-scaling law)

**Lane:** CRUNCH. **Task:** C2. **Date:** 2026-08-28.
**Engineering measurement. Allocates no RT ID. Appends no ledger. Not a promotion,
not a leaderboard submission. No test or lockbox data touched.**

---

## Headline

**Deployment cost does not gate any promotion in this program — at any `k`.**

| | measured | projected / 10,000 series | vs 15 h quota |
|---|---|---|---|
| RT-600 (k=0) | 1.586 ms/pt | **2.70 h** | PASS, 5.5× headroom |
| RT-1257 (k=2) | 2.152 ms/pt | **3.50 h** | PASS, 4.3× headroom |
| all-CatBoost (k=7) | 3.284 ms/pt | **5.08 h** | PASS, **3.0× headroom** |

Even the most expensive configuration the mechanism could ever produce — every one of
the seven slots CatBoost — lands at **5.08 h against a 15 h budget.** There is no value
of `k*` that CSA-04 can return which puts the hybrid outside budget.

**Two corrections to the brief's premises, both material.**

1. **The review had partly happened.** `CODEX_CRUNCH_LANE_PROMPT.md` §C2 states "That
   review has never happened." Per `AGENTS.md` ("trust the repository over the prompt and
   say so explicitly"): `engineering/rt1257-deployment-2026` already carried a full
   `crunch test` (PASSED, determinism passed) and an
   `engineering/reports/rt1257_deployment/RT1257_BENCHMARK.json`. What had *not* happened
   is a k-scaling measurement, and the k=2 number that existed was wrong — see below.
2. **The recorded `ratio_vs_rt600 = 0.912` is wrong, and wrong in the dangerous
   direction.** It says RT-1257 is ~9% *cheaper* than RT-600. Under a paired design it is
   **~36% more expensive** (paired ratio **1.357**, CPU; 1.360 wall). The recorded figure
   would have understated the cost of the entire mechanism.

---

## 1. Why the recorded k=2 number was wrong

`research/scripts/rt1257_deployment.py::benchmark_one` draws its series sample as:

```python
rng = np.random.default_rng(600 if out.name == "final10k_ensemble" else 1257)
idx = [int(x) for x in rng.choice(store.n_series, 40, replace=False)]
```

**The two arms are benchmarked on two different random samples of 40 series** — hence the
mismatched `n_points` in the recorded output (RT-1257 22,418 vs RT-600 21,283). Series
composition and model cost are confounded, and per-point cost varies strongly with series
shape (the RT-600 `RUNTIME_CONTRACT.json` shows 1.72–9.20 ms/pt across four series
profiles — a 5× swing driven by the series alone).

It is also a single unreplicated shot per arm. I re-ran the RT-1257 arm unchanged in a
faithfully reconstructed environment (`requirements-rt1257.txt` exactly: lightgbm 4.7.0,
catboost 1.2.10, numpy 2.4.6, pandas 3.0.5, pyarrow 25.0.1, numba 0.67.0, Python 3.11.6):

```
score_sum_guard  11170.88391577259   <- bitwise identical to the recorded run
ms_per_online_point  2.4663           <- recorded 2.2775  (+8.3%)
```

The predictions reproduce exactly, so the artifact and environment are faithful. **The
timing moved 8.3% on the same machine, same sample, same model** — the same magnitude as
the 8.8% advantage the recorded ratio claimed. That number could not have supported its
conclusion even had the samples matched.

---

## 2. What I ran instead

`research/reports/deep_ensemble_frontier_2026/crunch/c2_inference_benchmark.py`
→ `C2_RT1257_INFERENCE_BENCHMARK.json`

Same methodology as the existing benchmark and as the RTX 4090 hardware benchmark —
real streaming `ProductionModel.start_series()` / `.step()`, per-point cost projected to
10,000 series at the mean online horizon 503.6517, reported against the quota — with
three fixes:

1. **One shared series sample** (seed `20260828`, 40 series, 17,716 online points), used
   by both arms. Removes the confound.
2. **5 repeats with the arm order alternated** each repeat, so monotone machine drift
   cancels instead of loading onto whichever arm runs second. The reported estimator is
   the **paired per-repeat ratio**, not a difference of independent means.
3. **CPU time (`process_time`) alongside wall time.** This machine is shared with a
   concurrent agent session. All member predicts are pinned single-threaded
   (`num_threads=1` / `thread_count=1`), so CPU time is the contention-robust estimator.

Pairing is what makes the result usable: per-arm spread across repeats was ~32%, while
the **paired ratio's spread was 8.7%** (stdev 0.118 on 1.357). The absolute per-point
numbers on this host are noisy; the ratio and the per-slot decomposition are not.

I did not modify anything in `engineering/rt1257-deployment-2026`. Its model artifacts,
`src/`, and the `structural-break-claude-wave3` feature store were read only.

---

## 3. The measurement that actually answers the k question

`ProductionModel.step()` decomposes into a shared part and a per-slot part:

```
step() = engine.step(x)          # feature engine -- shared, model-independent
       + Σ_slots predict_one()   # one single-row predict per slot
       + _blend(ps, t)           # SCDF calibration + mean
```

Timed per component (CPU ms per online point, 1,664 points):

| component | RT-600 | RT-1257 |
|---|---|---|
| feature engine | 0.8036 | 0.7916 |
| blend | 0.0399 | 0.0392 |
| all seven slots | 0.7465 | 1.2112 |
| **sum** | **1.590** | **2.042** |
| *measured end-to-end* | *1.586* | *2.152* |

The RT-600 decomposition closes to **0.25%** against its own end-to-end measurement, and
the engine cost is the same in both arms (0.804 vs 0.792) — confirming the shared
component is genuinely independent of the model mix, which is what the projection rests
on.

**Per-slot cost, measured:**

| slot family | CPU ms/pt | |
|---|---|---|
| LightGBM | **0.1066** | (range across 7 slots: 0.0652–0.1650) |
| CatBoost | **0.3486** | (two slots: 0.3524, 0.3449) |
| **CatBoost / LightGBM** | **3.27×** | |

The LightGBM per-slot cost agrees across arms (0.1066 in RT-600, 0.1028 for the five
slots retained in RT-1257), and the two replaced slots (RT-600 slot 0 at 0.1041, slot 4
at 0.0970) sit near the LightGBM mean — so using the mean is not flattering either arm.

### The k-scaling law

A CatBoost specialist **replaces** a LightGBM one; it is not added. Member count stays
seven for every `k`, and only the mix changes:

```
cost(k) = engine + blend + (7 − k)·lgb_slot + k·cat_slot
        = 0.8435 + (7 − k)·0.1066 + k·0.3486     [CPU ms per online point]
```

Each CatBoost slot adds **+0.2420 ms/pt**, i.e. **+0.338 h per 10,000 series**.

| k | CatBoost slots | ms/pt | h / 10,000 series | vs 15 h |
|---|---|---|---|---|
| 0 | none — RT-600 | 1.5900 | 2.710 | PASS, 5.5× |
| 1 | | 1.8320 | 3.048 | PASS, 4.9× |
| **2** | **RT-1257 as it stands** | **2.0740** | **3.387** | **PASS, 4.4×** |
| **3** | | **2.3160** | **3.725** | **PASS, 4.0×** |
| **4** | | **2.5580** | **4.064** | **PASS, 3.7×** |
| 5 | | 2.8000 | 4.402 | PASS, 3.4× |
| 6 | | 3.0421 | 4.741 | PASS, 3.2× |
| 7 | every slot CatBoost | 3.2841 | 5.080 | PASS, 3.0× |

The brief asks for `k=2` and `k=k*` specifically. **k=3 → 3.73 h; k=4 → 4.06 h.** Both
pass with ~4× headroom. `k*` is not yet known (it arrives with **H2**), but the table
covers every value it can take, so **C2's second scope item is answered in advance and
does not need to wait on H2.**

**Stated assumption:** cost is additive and linear in the slot mix. That holds here by
construction — the slots are independent single-row predicts in a Python loop with no
shared state, which is exactly what the decomposition measures. It would *not* hold if
slots were batched into one call. If a future integration batches them, this table must
be re-measured.

Peak RSS 644 MB across both arms loaded simultaneously; the `crunch test` on RT-1257
alone reported 1.37 GB consumed / 2.75 GB max RSS, comfortably within a normal container.
Memory is not a constraint at any `k`.

---

## 4. Limits of this measurement — read before quoting the absolute hours

- **This is a local Apple Silicon host, not the Crunch platform runtime.** The brief asks
  for the platform's own runtime. The **ratio** (1.357) and the **per-slot structure**
  (3.27×, additive) should transfer, since they are properties of the two libraries'
  single-row predict paths. **The absolute hours should not be quoted as platform
  numbers.** The verdict is nonetheless robust: it would take a **>2.95× slowdown** versus
  this host, applied uniformly, to push even k=7 past 15 h.
- **Per-arm absolute timings on this host are noisy** (~32% spread across repeats, in CPU
  time as well as wall, so it is not purely contention). Only the paired ratio and the
  decomposition are stable. This is why the report leads with those.
- **The 15 h figure is inherited repo convention**, used by `RT1257_BENCHMARK.json`
  (`quota_h: 15.0`), the RT-600 readiness gate 9, and `FULL_OOF_PREREG.md`. I did not
  independently verify it against platform documentation.
- **Three prior RT-600 local projections disagree by ~2×** — 2.64 h (readiness gate 9),
  4.04 h (`RT1257_BENCHMARK.json`), 5.11 h (`RUNTIME_CONTRACT.json`, whose mean is skewed
  by a 9.2 ms/pt short-series profile). Mine is 2.70 h. They agree on the verdict and
  disagree on the magnitude; nobody should treat any single one as precise.

**What would close this properly:** a platform-runtime inference benchmark, which is a
separate cloud submission from C1's. I have not launched one — C1's submission is the
committed next cloud action and I am not spending a second submission slot on C2 without
that being the user's call, given the verdict already has ~3× margin at the worst `k`.

---

## 5. Consequences for the program

- **`RT-1257` is not blocked by deployment cost.** The `catboost_specialist_2026/FINAL.md`
  action item — *"review surviving CatBoost specialist(s) against deployment cost before
  any production work"* — is discharged for k=2, subject to §4.
- **The `PROGRAM_PLAN` rule** that no arm is called promotable until its online per-timestep
  cost is measured is now **satisfied for the whole ensemble lane**, for every `k`, not
  just the current best. The LOCAL agent's CSA-04 can return any `k*` without a deployment
  contingency.
- **It costs more than recorded, and it is still cheap.** The correction matters for
  honesty and for anything that extrapolates further (a k=7 configuration is ~1.9× RT-600
  per point, not ~0.9×), but it changes no gate.
- **C4 is a different question.** §C4.2 notes a selective SSM running causally at every
  online timestep is a fundamentally different inference profile from a tree. Nothing here
  transfers to it. A neural arm needs its own C2, and the 3.27× per-slot penalty for
  merely swapping *tree libraries* is a useful prior on how much a neural slot might cost.

---

## 6. Status

- **C2 scope item 1 (`RT-1257`, k=2): complete**, with the recorded figure corrected.
- **C2 scope item 2 (best-`k` hybrid): answered in advance for all `k` ∈ 0..7.** Does not
  block on **H2**; when `k*` arrives, read the row.
- **Open:** platform-runtime confirmation (§4), deliberately not launched.
- No RT IDs allocated. No ledger appended. `STATUS.md` untouched. No writes outside this
  lane's directory.
