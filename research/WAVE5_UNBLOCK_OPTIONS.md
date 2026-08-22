# WAVE 5 — UNBLOCK OPTIONS

**Written 2026-08-22 on `research/wave5-alpha`. Every number below was measured in
this container, not estimated.** Raw evidence:
`research/reports/wave5_package_probe.json`.

---

## 1. THE BLOCKER, DECOMPOSED

Wave 5 cannot run experiments here because there is no competition data. That is
actually **two** independent blockers, and they have different remedies:

| # | blocker | status |
|---|---|---|
| B1 | the 2026 feature store is absent from this container | remedy = move ~134 MB |
| B2 | **every `crunchdao.com` host is refused by network policy** (403 at CONNECT) | remedy = allowlist, or move the data by another route |

B2 is the one that makes B1 non-self-serviceable: `crunch-cli` installs fine from
PyPI (11.11.1, verified), but it cannot reach the API to download anything.

### 1.1 What is actually reachable from here

Measured, this container, this session:

| host | result |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | **reachable** (direct, in the proxy `noProxy` list) |
| `api.github.com`, `github.com`, `objects.githubusercontent.com` | **reachable** |
| `s3.amazonaws.com`, `storage.googleapis.com` | **reachable** |
| `api.crunchdao.com`, `hub.crunchdao.com`, `www.crunchdao.com` | **BLOCKED** |
| `huggingface.co` | **BLOCKED** (this is what also kills TabPFN) |
| `download.pytorch.org` | **BLOCKED** (forces the 4.5 GB CUDA torch install) |

**Throughput on a reachable host: ~97 MB/s.** A 134 MB store transfers in about
two seconds. Bandwidth is not a constraint anywhere in this document.

---

## 2. THE SIZE FINDING THAT MAKES THIS EASY

The thing that has to move is **small**, because the expensive artifact is
*derived*, not primary:

| asset | size | nature |
|---|---|---|
| `cache/store/values.npy` | **140,145,984 B ≈ 134 MB** | primary — 35.0M float32, all 10,000 series |
| `cache/store/meta.parquet` | negligible | primary — id, off, n_hist, n_online, tau_index |
| `research/folds/*.parquet` | negligible | primary — canonical + alt1/2/3 + final10k |
| 500-column feature cache | **~10 GB** | **derived** — recomputable, ~7 min/module |

Source: `store_values_bytes` in `research/REPRODUCIBILITY_MANIFEST.json`;
per-module cost from `research/PROTOCOL.md` §2.

**So the transfer is 134 MB, not 10 GB.** The feature cache is rebuilt locally by
running the 7 modules over the store. Both hashes are recorded in the manifest
(`store_values_sha256`, `store_meta_sha256`), so a transferred store can be
**verified byte-exact** before a single experiment runs — no trust required.

---

## 3. THIS CONTAINER IS A CAPABLE RESEARCH BOX

Measured here, so the "where do we run it" question can be answered on evidence
rather than assumption:

| | wave-1/2 container | **this container** | user's Mac (wave 3/4) |
|---|---|---|---|
| cores | 2 | **4** | 10 |
| RAM | 7 GB | **15 GB** | 16 GB |
| free disk | ~28 GB | **21 GB** | — |
| platform | Linux x86_64 | **Linux x86_64** | macOS/arm64 |

**LightGBM training at true RT-100 scale** (500 features, 600 trees, 63 leaves,
`num_threads=2`), extrapolated from two measured points (50k rows = 35.5 s,
200k rows = 112.7 s):

* **≈9.4–11.8 min per fold** at 1,000,000 rows
* **≈50–60 min per 5-fold stream**
* **≈6–7 h for a full 7-stream ensemble**
* peak matrix memory 1.86 GB, comfortable inside 15 GB

**This box is strictly better than the container that produced the wave-1/2
ledger, and it is the same platform as the Crunch runner** — which the Mac is
not. Every wave-3/4 number carries an "all wave-4 numbers are macOS/arm64; the
runner is Linux" caveat (`STATE_OF_RESEARCH_V4.md` §19). Running Wave 5 here
would retire that caveat instead of adding to it.

Disk is the one tight resource: 21 GB free against a ~10 GB feature cache. It
fits. ~4.5 GB can be reclaimed on demand by dropping torch's unusable CUDA wheels
(`nvidia` 2.7 GB + `triton` 0.7 GB + torch 1.1 GB), which no CPU experiment needs.

---

## 4. THE OPTION SPACE

### A — GET THE DATA HERE (unblocks everything)

| id | option | effort | verdict |
|---|---|---|---|
| **A1** | Upload `values.npy` + `meta.parquet` (134 MB) to any reachable host — S3/GCS presigned URL, GitHub release asset — and pull it in | minutes | **RECOMMENDED** |
| A2 | Upload raw `X_train.parquet` + `y_train_index.parquet`; rebuild with `research/scripts/build_store.py` | minutes + one conversion | good fallback; larger transfer, same endpoint |
| A3 | Add `*.crunchdao.com` to the environment's network allowlist, then `crunch-cli` downloads directly | needs an environment change by the owner | **best long-term**; makes the box self-serviceable and enables `crunch push` |
| A4 | Commit the store into the repo (Git LFS) | minutes | **not recommended** — competition-data redistribution, permanent repo bloat |

A1 and A3 are complementary, not alternatives: A1 unblocks *today*, A3 removes
the dependency on a human for every future refresh.

Verification for A1/A2 is already specified: compare against `store_values_sha256`
/ `store_meta_sha256` in `research/REPRODUCIBILITY_MANIFEST.json`.

### B — RUN THE WORK SOMEWHERE ELSE

| id | option | verdict |
|---|---|---|
| B1 | Run on the user's Mac, where the data already lives | works, and it is how waves 3–4 were done — but every number inherits the macOS/arm64-vs-Linux caveat, and local P=4 segfaults there |
| B2 | Provision a larger cloud box with data **and a GPU** | the only route that unlocks torch/TabPFN teachers at useful speed; highest cost |

### C — WORK THAT NEEDS NO DATA AT ALL (available right now)

This is the part worth emphasising, because it is not a consolation prize — it is
the gated prerequisite for every one of the eight untested alpha families.

**Proven in this container today:** `check_prefix_invariance()` takes raw numpy
arrays, not a store. All 9 registered modules (`m00_core`…`m09_back`) were
re-verified **bitwise prefix-invariant at `atol=0`** on synthetic series at three
length profiles — with zero competition data present.

| id | option | why it matters |
|---|---|---|
| **C1** | Build and causally certify new feature modules for the untested families — transient-vs-permanent, dependence likelihood v2, robust distribution distances, residual CUSUMSQ / AR regression break, robust BOCPD, AR(p)-FOCuS | 5–6 of the 8 families are feature modules. Certification is mandatory and is the slow, error-prone part — `FAILED_EXPERIMENTS.md` N6 records a causality bug that would have silently inflated every downstream number |
| **C2** | Write the XGBoost / CatBoost stream adapters **and the seed-clone control harness they must beat** | the control is non-negotiable under the wave-3 rule; building it now means a new family cannot be promoted without it |
| C3 | Implement the aParsec 2025 polars pipeline (R25-010→030) | dependencies are now verified present; only data is missing |
| C4 | Pre-register the Wave-5 experiments in `RDOF_LEDGER.md` | `VALIDATION_V2.md` requires pre-registration *before* the run; doing it now costs nothing and removes any suspicion of post-hoc bars |
| C5 | Synthetic-store plumbing smoke test of `sbr.pipeline.run` | catches integration breakage in the new adapters before real compute is spent |

**The boundary on C, stated explicitly.** `research/PROTOCOL.md` §1: *"Model
selection is TS-AUC on held-out trajectories, never row AUC, **never synthetic-data
performance**"*. Synthetic series may be used for **causal certification, plumbing
and runtime measurement only**. No synthetic number may rank a candidate, and none
will be reported as evidence of alpha.

### D — SEPARATELY BLOCKED: TabPFN

Independent of the data question, and it does not resolve with the others:

* `huggingface.co` is refused by network policy;
* TabPFN 8.4.0's default checkpoint `Prior-Labs/tabpfn_3` is a **gated** repo
  needing accepted terms plus a token;
* both were retested against `tabpfn==2.1.3`, the version aParsec pins — it fails
  identically, so this is **not** a version problem.

Remedies: sideload the checkpoint into `/root/.cache/tabpfn/`, or run TabPFN work
in an environment where HuggingFace is reachable and the terms are accepted. Until
one happens, rung **R25-040** stays blocked and no Wave-5 plan may assume TabPFN.

---

## 5. RECOMMENDATION

**A1 + C1/C2, in parallel.**

A1 because it is the shortest path from blocked to running, and because this
container is a better research box than the one that produced the ledger *and*
matches the runner's platform. A3 should follow as the durable fix.

C1/C2 because they are on the critical path regardless of when data arrives, they
cannot be short-cut later, and they are the one thing that can proceed **right
now** with no dependency on anyone. Certification is the slow part; doing it while
blocked converts dead time into the prerequisite for every promotion.

What must not happen while blocked: no synthetic-data score may be quoted, and no
parameter may be selected against RT-600's 0.6268.
