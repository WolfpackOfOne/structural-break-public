# PACKAGE CAPABILITY MATRIX — WAVE 5

**Built 2026-08-22 on branch `research/wave5-alpha`.**
Every row below was **measured in this container**, not inferred from documentation.
Probe scripts and raw output: `research/reports/wave5_package_probe.json`.

**Environment of measurement.** Linux x86_64, Python 3.12.3, 4 vCPU, 15 GB RAM,
**no GPU** (`nvidia-smi` absent, `torch.cuda.is_available()` False), venv
`.venv-wave5`, `OMP_NUM_THREADS=2`. Python 3.12 is deliberate — it is the Crunch
runner's Python, so a package that resolves to a cp312 wheel here resolves there.

---

## 0. THE RULE THAT DECIDES THE "PRODUCTION" COLUMN

**AMENDED 2026-08-22 (PRE-C3), before any Wave-5 TS-AUC was observed.** The
original wording of this section overstated its conclusion. It is corrected here
rather than rewritten silently.

No published Crunch package whitelist was **found** in this repository. That is
an absence of evidence, **not evidence of absence**: the competition may well
operate an allow-list, a size cap, a build timeout or a network policy on the
runner that nothing in our record would reveal. What the record does establish
(LB-002 → LB-003, `research/reports/rt600_baseline_submission.md` §15) is
narrower than "requirements.txt alone determines deployability":

> `crunch push` uploads `requirements.txt` and **the runner builds its environment
> from it** — `using original file: requirements.txt` appears in the push log of
> every submission made.

So the runner builds from a file we control. That makes deployability **at least**
an engineering question, with four gates we can test:

1. **cp312 wheel resolves** (`--only-binary=:all:`, Python 3.12) — tested per row below;
2. **install footprint** the runner will tolerate;
3. **single-row inference cost** inside the 15 h budget;
4. **determinism**, because the competition re-runs and compares at 1e-08.

A package failing gate 1 or 3 is not deployable whatever any policy says. But
passing all four does **not** establish the converse. Only a submission proves a
package runs on the runner, and a submission costs a leaderboard slot.

**Category language, corrected.** For anything RT-600 has not already shipped,
rows read **ENGINEERING-COMPATIBLE / CRUNCH WHITELIST STATUS UNVERIFIED**, never
"likely deployable". The earlier phrasing implied a permission finding we have
not made.

| category | means |
|---|---|
| **VERIFIED DEPLOYABLE** | has actually run on the Crunch runner and produced a score |
| **ENGINEERING-COMPATIBLE / WHITELIST UNVERIFIED** | passes all four local gates; acceptance by the runner is **unknown** |
| **RESEARCH ONLY** | valuable offline, never uploaded |
| **NOT SUITABLE / BLOCKED** | cannot run even offline here |

## 0.1 The runtime budget the cost column is read against

RT-600 measured: shared feature engine **1.069 ms/pt**, marginal LightGBM booster
**0.078 ms/pt**, seven boosters **1.734 ms/pt** end to end. The successful cloud
run (LB-004, P=4, 16 vCPU / 64 GB) took **2 h 11 m** against a **15 h** budget.

**Runtime is not the binding constraint right now, and Wave 5 must not spend
itself optimising it.** There is roughly 6.8× headroom. The cost column exists to
catch a family that would eat all of it, not to shave milliseconds.

---

## 1. THE MATRIX

| package | version tested | installs here | cp312 wheel | disk | CPU | GPU | deterministic | serialisation | single-row cost | category |
|---|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** | 4.7.0 | yes | yes | 10 MB | yes | no (CPU build) | **yes, bitwise** — `deterministic=True`, `num_threads` fixed | `Booster.save_model()` text | **0.115 ms** | **VERIFIED DEPLOYABLE** |
| **scikit-learn** | 1.9.0 | yes | yes | 39 MB | yes | no | yes via `random_state` | joblib/pickle, version-sensitive | n/a (not a booster here) | **VERIFIED DEPLOYABLE** |
| **XGBoost** | 3.4.1 | yes | yes | 84 MB | yes | CUDA in wheel, unusable (no GPU) | **yes, bitwise** — `tree_method=hist`, `nthread` fixed | `save_model()` → `.json`/`.ubj`, version-stable | **0.278 ms** (2.4× LGB) | **ENGINEERING-COMPATIBLE / WHITELIST UNVERIFIED** |
| **CatBoost** | 1.2.10 | yes | yes | 268 MB | yes | needs CUDA build | **yes, bitwise** — `random_seed` + `thread_count` | `save_model()` → `.cbm` binary | **0.564 ms** (4.9× LGB) | **ENGINEERING-COMPATIBLE / WHITELIST UNVERIFIED** — cost and size are real |
| **Polars** | 1.43.2 | yes | yes | 10 MB | yes | needs `cudf` (absent) | order-stable only with `maintain_order` / 1 thread | n/a (dataframe lib) | n/a — offline only | **RESEARCH ONLY** (by role, not by capability) |
| **SHAP** | 0.52.0 | yes | yes | 4 MB | yes | no | **yes** — `TreeExplainer` is exact | n/a (explainer) | n/a — training-time | **RESEARCH ONLY** |
| **PyTorch** | 2.13.0+cu130 | yes | yes | **1.1 GB + 2.7 GB CUDA + 0.7 GB triton** | yes | **compiled cu130, no GPU present** | yes — `use_deterministic_algorithms(True)` + `manual_seed` | `state_dict` via `torch.save` | not benchmarked (no model) | **RESEARCH ONLY / teacher** |
| **TabPFN** | 8.4.0 | imports | yes | 4 MB (+ weights) | yes | prefers GPU | `random_state`; inference is a forward pass | joblib + pretrained checkpoint | **cannot run** | **NOT SUITABLE (production)** / **BLOCKED (research)** |

---

## 2. ROW NOTES — WHAT THE MEASUREMENT ACTUALLY SHOWED

### LightGBM 4.7.0 — the incumbent, and the only VERIFIED row
The only family with an external result: RT-600 = **0.6268** Crunch TS-AUC. Bitwise
repeatable across two fits in-process. Everything else in this table is a candidate
measured in a container; this one has shipped and scored.

### XGBoost 3.4.1 — the strongest new production CANDIDATE (acceptance unverified)
Cheapest of the two new boosters and the easiest to trust. Bitwise repeatable with
`tree_method="hist"` and a fixed `nthread`; native JSON/UBJ serialisation is
explicitly version-stable, which matters because the artifact must load under a
runner-built environment. **Cost in context:** adding one XGBoost stream to
RT-600's 1.734 ms/pt is **+0.278 ms/pt, about +16%** — against 6.8× headroom, that
is affordable.

The PyPI wheel bundles CUDA kernels it cannot use here, inflating it to 84 MB. Not
disqualifying, but it is 8× LightGBM's footprint for a CPU-only runner.

### CatBoost 1.2.10 — viable, but the expensive option
Ordered boosting and symmetric trees make genuinely different errors from LightGBM,
which is exactly the specialist-diversity property W4-E1 showed is worth
**+0.0033**. That is the case for testing it. Against it: **0.564 ms/row is 4.9×
LightGBM**, and 268 MB is 27×. Adding one CatBoost stream costs **+33%** of
RT-600's per-point budget. Still inside the envelope, but it is the first family
where cost is a real term in the decision rather than a rounding error.

### Polars 1.43.2 — offline feature engineering, categorically not production
Capable of production in the abstract (small, fast, cp312-clean), and **not
production here for an architectural reason**: RT-600's inference path is a numba/
numpy streaming engine that sees one observation at a time and holds no dataframe.
Polars' value is offline — it is what the aParsec 2025 pipeline is written in, and
executing that pipeline is what §3 below is about. Default thread pool is 4;
`maintain_order` or a single thread is required for reproducible row order.

### SHAP 0.52.0 — research-only by role, and that is not a limitation
`TreeExplainer` ran exactly and deterministically against a LightGBM model
(shape (200, 20) on the probe). SHAP is a **training-time** procedure: it selects a
feature list, and the *list* ships while the explainer stays behind. This is
precisely how the 2025 solution uses it (SHAP top-200 / top-500). Nothing about
research-only status weakens it.

### PyTorch 2.13.0 — teacher only, and the disk cost says why
Installs and runs deterministically on CPU. **`torch.cuda.is_available()` is
False** — there is no GPU in this container, so anything torch-based trains at CPU
speed on 4 cores.

The footprint is the finding: PyPI's `torch` drags **~2.7 GB of NVIDIA CUDA wheels
plus 0.7 GB of triton** onto a machine with no GPU, ~4.5 GB total.
`download.pytorch.org`, which serves the CPU-only build, is **blocked by this
environment's network policy** (403 at CONNECT), so the slim wheel is not
obtainable here. Shipping this to the runner is out of the question; as an offline
teacher whose *outputs* become features or targets, it is fine.

### TabPFN 8.4.0 — installs, imports, and CANNOT RUN. Two independent blockers.
This is the row that matters most for §3, so the failure is stated precisely.

1. **Network policy.** `huggingface.co` returns **403 at CONNECT** in this
   container. No checkpoint can be fetched by any TabPFN version.
2. **Gating, independent of the network.** TabPFN 8.4.0's default checkpoint
   `Prior-Labs/tabpfn_3` is a **gated HuggingFace repo**: it requires accepting
   terms in a browser and an auth token. The probe surfaced
   `TabPFNHuggingFaceGatedRepoError` explicitly.

**Both were tested against the version aParsec actually pinned.** A clean
`tabpfn==2.1.3` install fails the same way at
`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` — so this is not a
"use the old version" problem.

**What unblocks it:** sideload the checkpoint into `/root/.cache/tabpfn/`, or run
TabPFN work in an environment with HuggingFace reachable and terms accepted. Until
then every TabPFN line in the 2025 reproduction ladder stays blocked, and no
Wave-5 plan may assume TabPFN is available.

---

## 3. WHAT THIS UNBLOCKS — AND WHAT IT DOES NOT

`codex/reproduce-2025-public-solution` (`422e4b2`) failed its calibration gate for
one stated reason: *"the required public-solution dependencies are absent and
installing them from PyPI was rejected by the permission reviewer"* — namely
**polars, lightgbm, shap, tabpfn**. That was never a scientific failure.

Three of those four are now installed and verified:

| dependency | 2025 status | Wave-5 status |
|---|---|---|
| polars | blocked | **1.43.2 installed, verified** |
| lightgbm | blocked | **4.7.0 installed, verified** |
| shap | blocked | **0.52.0 installed, TreeExplainer verified exact** |
| tabpfn | blocked | **still blocked** — gated repo *and* network policy |

So the reproduction ladder splits cleanly:

| rung | stage | now |
|---|---|---|
| R25-010 | public transform/stat bank, single LGBM | **unblocked (deps)** |
| R25-020 | + exact public feature selection (SHAP / gain) | **unblocked (deps)** |
| R25-030 | + four-LightGBM ensemble | **unblocked (deps)** |
| R25-040 | + TabPFN OOF meta-feature | **still blocked** |
| R25-050 | closest faithful full pipeline | **blocked** — needs R25-040 |

**A second blocker, and it is the binding one.** Installing the libraries does not
run the ladder, because **this container holds no competition data of either
year**. The 2025 audit read its parquet files from a path on the user's Mac
(`/home/user/…/quickstarters/baseline/data`); the 2026 feature store defaults to
`/home/claude/sb/cache/store` (override: `SBR_STORE` / `SBR_ROOT`) and is likewise
absent. `data/` in this repo holds three small synthetic files, ~16 KB, for the
baseline demo only.

Honest statement of what was achieved: **the dependency blocker is cleared for
three of four rungs; the data blocker is not, and no rung was executed.** R25-040
needs a TabPFN checkpoint on top of that.

---

## 4. CATEGORY ASSIGNMENTS, STATED PLAINLY

| category | packages | meaning |
|---|---|---|
| **VERIFIED DEPLOYABLE** | LightGBM 4.7.0, scikit-learn 1.9.0 | already ran on the Crunch runner and produced 0.6268 |
| **ENGINEERING-COMPATIBLE / WHITELIST UNVERIFIED** | XGBoost 3.4.1, CatBoost 1.2.10 | cp312-clean, CPU-deterministic, serialisable, affordable — **acceptance by the Crunch runner is unknown and untested** |
| **WHITELIST STATUS UNKNOWN** | XGBoost, CatBoost, PyTorch, TabPFN — i.e. everything RT-600 has not shipped | no whitelist was found in our record, which is not the same as none existing |
| **RESEARCH ONLY** | Polars, SHAP, PyTorch | valuable offline; never uploaded. Polars and SHAP by role, PyTorch by footprint |
| **NOT SUITABLE / BLOCKED** | TabPFN | cannot obtain weights here; not a production candidate under any reading |

## 5. TEACHER SUITABILITY

A teacher's cost is paid offline; only its *output* touches production. That makes
the deployability column irrelevant for this use and changes the ranking:

| package | teacher suitability | why |
|---|---|---|
| PyTorch | **high, if a GPU is available** | arbitrary sequence models over the raw series; distil into a feature or a target. No GPU here → slow on 4 CPU cores |
| TabPFN | **high in principle, zero today** | strongest known tabular prior; blocked twice over |
| CatBoost | **moderate** | different inductive bias, cheap offline; but if it is good enough to teach, ship it directly |
| XGBoost | **moderate** | same argument, cheaper still |
| SHAP | **n/a — selector, not a teacher** | produces a feature list, not predictions |

---

## 6. WHAT I DID NOT MEASURE, AND WILL NOT CLAIM

* **Nothing here is a TS-AUC number.** No package in this table has been shown to
  add a single point of alpha. This is a capability audit; the alpha question is
  untouched and stays untouched until data is present.
* **No runner verification.** Every ENGINEERING-COMPATIBLE row is an inference from wheel
  tags and local timings. LB-001 and LB-002 both passed locally and died in the
  cloud — this project has already been taught that lesson twice.
* **Latency is a proxy.** 600 trees / 500 features / `OMP_NUM_THREADS=2` on 4 vCPU
  is matched to an RT-600 stream, but the runner has 16 vCPU and a different
  memory system. Ratios between families will hold better than absolute values.
* **No GPU claim is tested.** There is no GPU here. Every GPU cell is read off
  build metadata, not exercised.

---

## 7. C2 UPDATE — 2026-08-22

Re-verified in `.venv-wave5`: polars 1.43.2, lightgbm 4.7.0, shap 0.52.0,
xgboost 3.4.1, catboost 1.2.10, torch 2.13.0, tabpfn 8.4.0, on the
production-matching core (numpy 2.4.6 / pandas 3.0.5 / scikit-learn 1.9.0).
Environment unchanged from §1; production `requirements.txt` still untouched.

### 7.1 The 2025 reproduction dependency gate now PASSES — with one caveat

`research/scripts/reproduce_aparsec_2025.py` was lifted read-only from
`codex/reproduce-2025-public-solution` and its **own** gate
(`missing_required_reproduction_deps`) executed against `.venv-wave5`:

| | |
|---|---|
| codex branch reported missing | `[polars, lightgbm, shap, tabpfn]` |
| missing now | **`[]` — none** |
| ladder status reported by the script | all six rungs `pending run / deps available` |

**The caveat, and it matters.** That gate checks whether the *package* imports,
not whether TabPFN has usable *weights*. TabPFN 8.4.0 is installed and importable,
so the gate is satisfied — but `huggingface.co` is refused by network policy and
8.4.0's default checkpoint `Prior-Labs/tabpfn_3` is a gated repo. **R25-040 and
R25-050 would pass this gate and then fail at runtime on the weight download.**

Honest ladder status:

| rung | stage | dependency gate | actually runnable |
|---|---|---|---|
| R25-010 | transform/stat bank, single LGBM | pass | **yes, once 2025 data is present** |
| R25-020 | + SHAP / gain feature selection | pass | **yes, once 2025 data is present** |
| R25-030 | + four-LightGBM ensemble | pass | **yes, once 2025 data is present** |
| R25-040 | + TabPFN OOF meta-feature | pass | **no — weights unobtainable** |
| R25-050 | closest faithful full pipeline | pass | **no — depends on R25-040** |

A second mismatch for exactness: aParsec pins `tabpfn==2.1.3`; the installed line
is 8.4.0, a different major version with a different API and a different (also
unreachable) checkpoint. Faithful reproduction of R25-040 needs both a reachable
HuggingFace **and** the 2.x line.

The 2025 reproduction remains **EXECUTION BLOCKED, not scientifically failed** —
now blocked on 2025 *data* (absent from this container) for three rungs and on
TabPFN *weights* for the other two, rather than on four missing packages.

### 7.2 Model-family work is deliberately NOT started

XGBoost and CatBoost remain ENGINEERING-COMPATIBLE (whitelist unverified) and unused. No model-family
experiment has been designed or run, because C2's purpose is to finish the
feature-mechanism bank before any score is consumed, and letting a model-family
comparison in now would contaminate feature-mechanism selection with a second
moving part. They are staged for C3 and later.
