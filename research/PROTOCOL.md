# RESEARCH PROTOCOL — 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time)
**Read this file completely before writing any code. It is binding.**

Objective: maximise TRUE out-of-sample **Time-Stratified AUC (TS-AUC)**.

## 0. Where everything lives
```
/home/claude/sb
  src/sbr/                 shared research library  (DO NOT EDIT core files)
    store.py               per-series data store (float32 memmap)
    metric.py              official TS-AUC, verified against sklearn reference
    transforms.py          HistParams + per-point transforms
    nullcal.py             per-series historical-null calibration engine
    pipeline.py            leakage-safe OOF training / evaluation / ledger
    features/base.py       feature-module registry + causality harness
    features/driver.py     computes modules over a store, caches to disk
    features/m00_core.py   reference module (read it as the template)
  cache/store              FULL store: 10,000 series
  cache/store_screen       SCREEN store: 2,500 series (500 per dev fold)
  cache/features           full-store feature caches   <module>.npy/.cols.json
  cache/features_screen    screen-store feature caches
  research/folds/folds.parquet          PERMANENT folds. NEVER regenerate.
  research/RESULTS.csv                  experiment ledger (append via pipeline only)
  research/oof/<EXP>.npy                OOF prediction artifacts
  research/reports/<name>.md            your written report
  research/FAILED_EXPERIMENTS.md        negative results (append yours)
```

## 1. Absolute rules
- **The lockbox (fold == -1, 2,000 series) is untouchable.** Never load it, never
  look at it, never select anything with it. Agent 0 alone may open it, once.
- **`X_test.reduced.parquet` is FROZEN.** Never read it for anything.
- **No row-wise random splits.** Splits are series-level and already fixed in
  `research/folds/folds.parquet`. Never make your own split.
- **No future information.** Row `t` of a feature may use only `hist` and
  `online[:t+1]`. This is checked, not trusted — see §3.
- **No editing shared files.** You own `src/sbr/features/<your module>.py` and
  `research/reports/<your report>.md` and nothing else. If you believe a core
  file is wrong, say so in your report; do not edit it.
- **Model selection is TS-AUC on held-out trajectories**, never row AUC, never
  synthetic-data performance, never "it looked better".
- Record every experiment through `sbr.pipeline.run(...)`, which appends to the
  ledger under a file lock. Never hand-edit `RESULTS.csv`.

## 2. Compute reality (read this, it changes what you should attempt)
This container has **2 CPU cores and 7 GB RAM**, shared with every other agent
running right now. Disk is ~28 GB free and each 150-column module over the full
store costs ~3 GB.
- Keep a module to **at most ~60 columns** unless you clear it with Agent 0.
- **Develop and screen on `cache/store_screen`** (2,500 series). A module build
  there takes ~1–2 minutes; on the full store it takes ~7 minutes per module.
- Do NOT launch more than one multi-core job at a time. Use `--workers 1` unless
  you are the only thing running.
- Prefer vectorised NumPy over per-point Python loops. The whole feature engine
  is built on cumulative sums for exactly this reason.

## 3. The causal contract (non-negotiable)
Every feature module is a function registered with `@register("<name>", ...)`
taking a `SeriesCtx` and returning `(list_of_column_names, float32 array of
shape (n_online, k))`.

Before you report ANY number, run:
```python
from sbr.features.base import load_all, check_prefix_invariance
load_all()
ok, msg = check_prefix_invariance("your_module", hist, online)   # atol=0.0
```
on at least 8 series of different lengths. It must pass **exactly** (bitwise).
If it does not, your feature is looking into the future and every number you
derive from it is worthless. Paste the result into your report.

Helpers on `ctx`:
- `ctx.roll(name, w)` trailing-window mean of transform `name` (NaN before w points)
- `ctx.expand(name)` expanding mean over `online[0..t]`
- `ctx.nc.pct(name, w, v)` / `ctx.nc.z(name, L, v)` / `ctx.nc.surprise(name, w, v)`
  historical-null calibration (grid `ctx.nc.grid`)
- `ctx.hp` historical params (mu, sd, med, mad, quantiles, AR coefs, `pit()`)
- `ctx.tr` / `ctx.cum` raw and cumsum'd online transforms
NaN is allowed and is the correct value for "not enough online points yet" —
LightGBM handles it natively. Never back-fill a trailing window from history.

## 4. Every feature needs a story
In your report, for each feature family, state:
- **SIGNAL / ALPHA** — what kind of structural break makes this move, and why.
- **FALSE SIGNAL** — what *non-structural* behaviour moves it the same way
  (isolated outlier, transient vol burst, temporary level shift, heavy tail).
- **DISAMBIGUATOR** — which companion feature separates the two.
A feature with no false-signal story has not been thought about.

## 5. Screening protocol (use this for everything exploratory)
```python
from sbr.pipeline import run
run(exp_id="RT-0xx", modules=["m00_core","your_module"], agent="agentN",
    hypothesis="...", falsification="...",
    folds=(0,), max_train_rows=300_000, screen=True)
```
`screen=True` uses the screen store + screen feature cache. Report the delta
against the same call **without** your module — always run that control
yourself, in the same session, with the same seed. A number without its paired
control is not evidence.

Promotion to the full 5-fold protocol is Agent 0's decision, not yours.

## 6. What you must return
```
AGENT NAME / HYPOTHESIS / FALSIFICATION CONDITION / FILES CHANGED /
EXPERIMENT IDs / DATA USED / FOLDS USED / MODEL+FEATURES /
TS-AUC (screen and/or OOF) / PER-FOLD / DELTA VS CONTROL / RUNTIME /
CAUSALITY CHECK OUTPUT / LEAKAGE RISKS / SIGNAL STORY / FALSE-SIGNAL STORY /
OOF ARTIFACT PATH / CONCLUSION: KEEP | REJECT | RETEST | PROMISING-BUT-UNPROVEN
```
Negative results are deliverables. Append them to
`research/FAILED_EXPERIMENTS.md` with hypothesis, what you did, the number, why
you think it failed, and whether a retry is warranted.

## 7. Known facts about the 2026 data (verified by Agent 0)
- 10,000 training series; **49.67 %** have a break.
- Historical length 1,000–5,000 (mean 3,000); online length 10–999 (mean 504).
- `tau_index` is the 0-based index of the first post-break online point;
  relative break position is ~uniform (mean 0.489 of the online segment).
- 5,036,517 online rows in total; 25.6 % of rows are post-break positives.
- Folds: 5 dev folds of ~1,600 series each + a 2,000-series lockbox, stratified
  on break/no-break × tau quartile × history-length tertile × online-length tertile.
