# BRIEF — CODEX · WAVE-3 ENGINEERING & SOURCE-RESEARCH LANE
### 2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)

You have **two tasks and a hard boundary**. A second agent (Claude) is running the
research program in parallel and owns all validation and all promotion decisions.
Your lane is deliberately chosen to touch **no validation surface at all**, so the
two of you cannot contaminate each other.

Read `research/HANDOFF_WAVE3.md` first — full project state, scoreboard and traps.
Work on a branch off **`research/wave2-2026`** (HEAD `8e76ad2`). **Do not merge your
own work.**

---

## TASK 1 — BUILD THE SUBMISSION AND RUN THE OFFICIAL `crunch test`

**This is the highest-value unblocked item in the entire project.** The production
system (`RT-150`, deployable OOF ≈ 0.62589) currently exists only as a report:
`research/STATE_OF_RESEARCH_V2.md` states that
`submissions/C_ensemble_deployable.ipynb` passes at 4.57 ms/point, but

* `.gitignore` line 65 excludes `submissions/*.ipynb` as a build artifact,
* the trained model directory is uncommitted,
* the wave-2 OOF `.npy` vectors are uncommitted,

so **none of it is verifiable from a clean checkout**, and the official test has
never been run by anyone. Earlier attempts failed because cloud containers cannot
reach `api.hub.crunchdao.com` (403 at the egress proxy) and the sandboxed Linux VM
on the Mac has no network and no LightGBM. **Only the user's macOS environment can
do this.**

### Steps

1. Environment: use the repo's macOS `.venv`;
   `pip install -r requirements-research.txt` (numpy, pandas, pyarrow, scipy,
   scikit-learn, lightgbm, numba). Record exact versions — wave 2 found environment
   sensitivity around GOSS configuration.
2. **Known gotcha:** `research/scripts/build_submission.py` and
   `research/scripts/wave2_lib.py` hardcode `ROOT = "/home/claude/sb"`. Repoint them
   at the real repository path **before** running, or they will silently build
   against nothing. Do not "fix" this by editing shared library files beyond the path
   constant; prefer an env var or a local override, and say what you changed.
3. Rebuild the store, folds and 500-column feature cache if absent
   (`research/scripts/build_store.py`, then the feature driver). ~45–60 min.
   **Do not regenerate `research/folds/folds.parquet` if it exists** — the canonical
   fold partition is immutable and regenerating it silently invalidates every past
   result. Verify its SHA256 is `6e114f80…`.
4. Train the production model and build the notebook:
   `python3 research/scripts/build_submission.py`.
5. Run the repo's own isolated harness first:
   `python3 research/scripts/test_submission_notebook.py`.
6. Then the real thing: `crunch test`. Do **not** pass `--no-determinism-check` —
   determinism to 1e-8 on a 10 % re-run is a **reward-eligibility condition**, so the
   check is the point.

### Record, in `research/reports/crunch_test_rt150.md`

Exact command · environment and package versions · full runtime and **ms per online
point** · the determinism check result · every warning · forbidden-library check ·
pass/fail · and any dependency problem. If it **fails**, fix the submission and run
again — and say precisely what was broken, because a build that never worked is a
different fact from one that regressed.

Then **commit the built notebook and model artifacts**, or if they are too large, a
checksum manifest plus exact rebuild instructions. The current situation — a
production claim with no reproducible artifact — must not survive your lane.

Also project total competition runtime against the platform budget: **15 h/week**,
10,000 public + 10,000 private series (~5.0M / ~10.1M scoring points), which allows
≈10.7 ms/point at parallelism 1 and ≈42.9 ms at `INFER_PARALLELISM = 4`.

---

## TASK 2 — READ THE 2025 SOLUTIONS AND TRANSLATE THEM

Nobody has read these. Wave 2 recorded them as unreachable; they are publicly
indexed:

* `https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution` (self-described 2nd place)
* `https://github.com/StefanConstantin707/adia-lab-structural-break-challenge`

The 2025 problem was **offline with a known boundary**; 2026 is **causal and
streaming with unknown τ**. So for every idea the only question that matters is:

> What is the causal 2026 equivalent, computable from the historical segment plus
> `online[:t+1]` alone?

Example: 2025 "Wasserstein between pre- and post-boundary samples" → 2026
"historical reference distribution vs a causal trailing window, or vs the estimated
post-τ̂ segment, calibrated at matched window length."

### Deliverable: `research/reports/codex_2025_translation.md`

One row per idea: **source · original idea · why it plausibly worked · offline or
online · causal 2026 translation · which break family it targets · the false-signal
mechanism · a disambiguating companion feature · compute cost · leakage risk ·
proposed experiment ID in the `RT-9xx` range.**

Context that should shape your triage: 2026 forensics found **unconditional
location breaks essentially do not exist** (median |mean shift| 0.054σ, identical to
a placebo split) while **scale and dependence breaks do**, and 92 % of break series
are individually indistinguishable from a placebo split. A 2025 idea aimed at mean
shifts is probably dead on arrival; the same machinery aimed at a squared or
lag-product stream may not be. Also note that a **transformed detector bank on
`z²`/`|z|`/lag-products was already built and rejected** (delta −0.00073, ensemble
delta +0.00023) — see `FAILED_EXPERIMENTS.md` before proposing it again.

Append verified findings to `research/PUBLIC_IDEA_MAP.md` using its four-state
vocabulary: **VERIFIED** (retrieved and read) · **ATTESTED** (a prior source in this
repo describes it, primary not retrieved) · **FAILED TO RE-ACCESS** · **NOT FOUND**.
**Never write "does not exist"** for something you merely failed to find — that error
is already in this repo's history and has been corrected once.

---

## THE BOUNDARY — do not cross it

| | you (Codex) | Claude |
|---|---|---|
| experiment IDs | **`RT-9xx`** | `RT-3xx` |
| you may write | `submissions/`, `research/reports/codex_*`, `research/PUBLIC_IDEA_MAP.md`, `research/reports/crunch_test_rt150.md` | everything else |
| you may **not** write | `src/sbr/features/`, `research/RESULTS.csv`, `research/RDOF_LEDGER.md`, `research/folds/`, `STATE_OF_RESEARCH_V3.md`, any model or ensemble code | — |
| may promote a model? | **no** | yes, sole authority |

You report facts; Claude decides what they mean. A `crunch test` result is a fact.
"This idea should be adopted" is a decision, and it is not yours — propose it with an
`RT-9xx` ID and let Claude run it under its own pre-registration.

**Never** evaluate anything against the five canonical dev folds, and **never** touch
`X_test.reduced.parquet`. The original lockbox is spent; there is no clean holdout
left, so every uncounted evaluation against the dev folds permanently degrades the
project's ability to tell signal from selection. This is the entire reason your lane
was drawn this way.

If you finish both tasks, do **not** start alpha research. Report back and ask.
