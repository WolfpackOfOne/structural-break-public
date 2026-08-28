# C1.1 — ARTIFACT REUSE ACROSS CRUNCH RUNS

**Lane:** CRUNCH. **Task:** C1.1, the question that must be answered before any GPU time.
**Date:** 2026-08-28. **Allocates no RT IDs. Spent no compute.**

**Bottom line:** Branch A (reuse the existing `RT-1258`/`RT-1259` neural artifacts) is
**NOT available** for a new submission. Branch B — a full technical-recovery run at
**4.545 GPU-hours** — is the operative path.

But the premise of Branch B changes in the program's favour. The eight control arrays
**can** be shipped into the cloud, at the exact path the loader reads, **with no shim,
no renamed files, no `.gitignore` edit, and nothing committed to git.** The binding test
therefore computes in-session on the first attempt, and C1 is **one run, not two**.

The previous run did not stop because the platform could not carry the controls. It
stopped because the controls **were not in the working tree when `crunch push` ran**.

---

## 0. Method, and what counts as evidence here

Everything below is read from `crunch-cli` **11.12.0** as installed
(`/home/user/anaconda3/lib/python3.11/site-packages/crunch/`) and from the
competition's own scoring runner, fetched live from the authoritative source used by
the CLI itself:

```
https://raw.githubusercontent.com/crunchdao/competitions/master/
    competitions/structural-break-real-time/scoring/runner.py
```

(`crunch/unstructured/_code_loader.py` builds exactly this URL from
`constants.COMPETITIONS_REPOSITORY = "crunchdao/competitions"`, branch `master`, and the
competition name `structural-break-real-time` read from `.crunchdao/project.json`.)

Claims marked **[verified]** were reproduced by executing the CLI's own file-selection
functions against a synthetic tree carrying this repository's real `.gitignore`. Claims
marked **[read]** are read from source but not executed. One question is marked
**[unresolved]** and is flagged in §6.

I attempted to confirm the submission-side state directly against the platform
(`project.submissions.list()`). It returns **403 `AuthorizationDeniedException`**: the
stored credential in `.crunchdao/token` is a *push* token and is not scoped to read
submission metadata. I did not attempt to widen that scope. The mechanism below is
established from the client and runner source instead, which is sufficient because the
client is the thing that decides what gets uploaded.

---

## 1. `model_directory_path` and `resources` are the same directory

Not two things. `crunch/constants.py`:

```python
DEFAULT_MODEL_DIRECTORY = "resources"
```

and every CLI entry point defaults `--model-directory` to that constant. So
"does `resources/` persist across runs?" and "does `model_directory_path` persist across
runs?" are one question. **[read]**

---

## 2. Across runs *of one submission*: the model persists automatically

`crunch/runner/cloud.py`, in the run's `initialize()`:

```python
with self._span("downloading model"):
    os.makedirs(self.model_directory_path, exist_ok=True)
    self.has_model = len(model_file_urls) != 0
    self.pre_model_files_modification = _download_files(
        file_urls=model_file_urls,           # == self.run.model
        directory_path=self.model_directory_path,
    )
```

and at the end, `_upload_files(category="model", directory_path=self.model_directory_path, ...)`
walks the directory and uploads it, reporting `use_initial_model=not has_model_changed`.

So `resources/` is **not** rebuilt from the submitted code tree each run — it is
populated from the server's model registry for that run, and written back at the end.
`RunnerContext` exposes `chain_height` ("how many consecutive runs are there with the
same parameters?") and `has_model` ("whether this submission originally had a model"),
which is the vocabulary of a persistent per-submission model chain. **[read]**

Combined with our own submission's logic this would be genuine, free reuse:
`_resolve_persisted_output_root()` checkpoints under
`model_directory_path/gpu_tabular_oof/`, `_report_existing_checkpoints()` logs what it
finds, and `_run_runner()` always passes `--resume`. A second run of *the same
submission* would find `{tabm,realmlp}/fold_{0..4}_pred.npy` present and skip all
training.

**Two reasons this does not help us.**

1. `train()` runs only `if context.force_first_train:` (competition `runner.py:122`).
   Whether a submission is re-run, and whether that re-run trains, is set by the
   platform — it is not something we trigger on demand.
2. It applies to *the same submission*. Our next action is a **new** submission — see §3.

---

## 3. A NEW submission does **not** inherit the previous submission's model

This is the finding that decides Branch A vs Branch B.

`crunch/command/push.py::push()` ends with:

```python
submission = project.submissions.create(
    ...,
    code_files={path: upload.id for path, upload in code_uploads.items()},
    model_files={path: upload.id for path, upload in model_uploads.items()},
)
```

`model_uploads` is filled **only** by `list_model_files(submission_directory_path,
model_directory_relative_path)` — i.e. by walking the **local** `resources/` directory at
push time. There is no server-side inheritance, no "previous model" reference, no flag to
mount a prior submission's artifacts. **A new submission's model is exactly, and only,
what the local `resources/` directory held when you pushed.** **[read]**

And the local `resources/` directory in the worktree the GPU submission was pushed from
is **empty**:

```
structural-break-gpu-tabular-2026/resources/     -> exists, zero files
```

So a new submission would arrive in the cloud with `has_model = False`, an empty
`resources/`, no fold checkpoints, `--resume` finding nothing, and **both learners
retraining in full: 4.545298927912005 GPU-hours.**

The only route to Branch A would be to download submission `76357`'s model artifacts to
the local machine and re-push them. That is precisely the workflow
`CODEX_CRUNCH_LANE_PROMPT.md` §2 and §8 prohibit ("Do not build a workflow that depends
on pulling `tabm_oof.npy` / `realmlp_oof.npy` down"; "Do not attempt to pull the neural
OOF arrays out of Crunch by any means"). **Branch A is closed — by policy, and in
practice by the absence of any mount mechanism.**

> **Cost, stated as a decision rather than a surprise, per §9 of the brief:**
> completing C1 costs a full **4.545 GPU-hour** regeneration of `RT-1258` and `RT-1259`
> against the 15-h / 2-learner / 0.90-fraction quota. There is no cheaper path that does
> not violate the extraction prohibition.

---

## 4. The controls **can** reach the cloud — as *code* files, unchanged

This is the part that makes the 4.5 hours buy a complete result instead of a second
`PENDING_LOCAL_BINDING_EVALUATION`.

`crunch push` selects code and model files by two **different** rules:

| | applies repo `.gitignore`? | source |
|---|---|---|
| `list_code_files()` | **No** | `_build_gitignore(..., use_parent_gitignore=False)` |
| `list_model_files()` | **Yes** | `_build_gitignore(..., use_parent_gitignore=True)` |

Code files are filtered only by `IGNORED_CODE_FILES` — `.git/`, `.crunchdao/`,
`__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, `__MACOSX`, `.DS_Store`,
`encryption.json`, `/data/`, `/prediction/` — plus the model directory.

**Executed against a tree carrying this repo's real `.gitignore`: [verified]**

```
research/oof/{RT-300,RT-401,RT-410..RT-415}.npy   -> shipped as CODE files
research/folds/folds.parquet                       -> shipped as CODE file
```

They are `.gitignore`d (`research/oof/` line 60, `*.npy` line 61) and they ship anyway,
because the code-file walk never consults `.gitignore`. `research/folds/folds.parquet` is
the existing proof of this: it is covered by `*.parquet` (line 47) and it reached the
previous run regardless.

They land at `<repo root>/research/oof/{ID}.npy` — **exactly** the path
`common.py::load_control_oof()` reads:

```python
oof_dir = root / "research" / "oof"
need = list(SPECIALISTS) + [MATCHED_LGBM]
return {name: np.load(oof_dir / f"{name}.npy") for name in need}
```

**No shim. No relocation. No renamed files. No `.gitignore` change. Nothing committed to
git.** This satisfies §C1.2.4's "follow the repository's existing convention rather than
inventing filenames" in the strongest possible way — the convention is already correct,
the files were merely absent.

### Why the previous run actually stopped

`structural-break-gpu-tabular-2026/research/oof/` — the worktree submission `76357` was
pushed from — **does not exist**. Not empty: absent.

The controls were never in the push. `_try_binding_replacement_test()` caught the
`SystemExit` from `load_control_oof`, returned `{"computed": False}`, and the run ended
at `PENDING_LOCAL_BINDING_EVALUATION` exactly as designed.

`FULL_OOF_PREREG.md` §"Infrastructure note" reasoned that the controls "are not present
in a cold Crunch-cloud checkout" because their paths are `.gitignore`d, and concluded the
binding test had to be a local post-run step. **That inference is incorrect for code
files**, and it is the root cause of the incomplete run. The prereg is a committed
historical record and is not being edited; this report is the correction, and it is what
changes the handoff direction (`PROGRAM_PLAN.md` §3.2) from "bring the arrays home" to
"send the controls out".

---

## 5. The trap: do **not** stage the controls under `resources/`

The obvious-looking alternative is silently destructive.

`list_model_files()` passes `use_parent_gitignore=True`, so the repo `.gitignore` **is**
applied to model files. Executed against the real `.gitignore`: **[verified]**

```
resources/RT-300.npy ... RT-415.npy   -> SILENTLY DROPPED
resources/controls.npz                -> SILENTLY DROPPED
resources/manifest.json               -> shipped
```

Cause isolated to a single line by differential test: **`.gitignore:61 *.npy`** (removing
it restores all eight; removing `resources/` on line 76 changes nothing). Line 63
`*.npz` closes the obvious workaround, and lines 45–47 also ignore `*.pkl`, `*.joblib`,
`*.parquet`.

There is **no warning and no error.** A run staged that way would upload zero controls,
train for 4.5 GPU-hours, reach `_try_binding_replacement_test()`, find nothing, and
return `PENDING_LOCAL_BINDING_EVALUATION` a second time. This is the single most
expensive available mistake and it is invisible until the run ends.

**Rule for C1: the controls go in `research/oof/` as code. `resources/` stays empty.**

---

## 6. Residual risk: total submission size **[unresolved]**

The maximum submission size is enforced **server-side**; the client only learns it by
being told (`ModelTooBigException` / `SubmissionTooBigException` carry a `maximum_size`
supplied by the API). It is not a client constant and I could not read it with a
push-scoped token.

Measured payload, by running the CLI's own selection over this worktree:

| | size |
|---|---|
| current code payload | **39.3 MiB** (555 files) |
| + eight control arrays | **+153.7 MiB** (8 × 20,146,196 B) |
| **total** | **≈ 193 MiB** |

**The risk is bounded and cheap.** A size rejection happens at *push* time, before any
GPU time is allocated — it costs a push, not a run.

Two free mitigations, in order:

1. **`crunch push --dry` first.** With `dry=True`, `push.py::handle()` prints
   `found code file: <name> (<size>)` for every file and returns *before* uploading
   anything. It is a complete, zero-side-effect manifest of what would ship. **Run this
   and confirm all eight `research/oof/*.npy` appear, before the real push.** **[read]**
2. **If the cap is hit, shrink the code side, not the arrays.** ~30 MiB of the 39.3 MiB
   is unrelated to this submission — `submissions/C_ensemble_deployable.py` alone is
   26.8 MiB, plus `research/archive/`. Removing those from the pushed tree is
   scientifically inert.

Do **not** shrink by storing only the finite entries and reconstituting the NaN mask. It
saves 20% and introduces exactly the misalignment failure mode §7 exists to prevent.

---

## 7. H1 controls: independently re-verified — **PASS**

The LOCAL agent's H1 package had already landed, so the §7-step-5 verification was done
before writing this. Verified from the tar, not the loose copies:

`H1_CONTROL_OOF_PACKAGE.tar`, 161,195,008 B,
sha256 `a40e118676567210530c7cbf969548fb6b8aa5b51b5c13048f41f984a1e85603` — matches the
manifest.

All eight, extracted and checked independently against
`local/H1_CONTROL_MANIFEST.json`:

| ID | bytes | shape | dtype | finite | sha256 | 82-byte symlink? |
|---|---|---|---|---|---|---|
| RT-300 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-401 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-410 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-411 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-412 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-413 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-414 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |
| RT-415 | 20,146,196 | (5036517,) | float32 | 4,032,524 | match | no — real file |

Finite fraction 0.8006572796239941 on every array — the five canonical dev folds, with
the ~20% lockbox correctly `NaN`. The symlink hazard is cleared: 19.21 MiB each, not
82 bytes.

**Two checks added beyond the brief**, because equal finite *counts* do not prove equal
finite *positions* — and a control that is misaligned rather than missing produces a
wrong `marginal_vs_clone` instead of an error:

- **Position-level NaN-mask identity.** All eight share one identical finite mask
  (`np.array_equal` against RT-300, not just a count comparison). Shared mask sha256
  `64eb8102a17e6a37f42822264b6b6c91…`. This is the check that actually rules out
  misalignment.
- **Distinctness.** No two of the eight are identical arrays. Within-`t`-free Pearson ρ
  of RT-401 against the seven specialists runs 0.715–0.936 — consistent with a matched
  exchangeable clone, and not a duplicate of any of them.

### One anomaly, resolved: RT-413 is not on a [0,1] scale

RT-413 ranges **−3.93 to 5.78**, mean −0.383 — a raw margin / logit scale. The other
seven are probabilities in [0,1].

**This is harmless, and it must not be "fixed".** Integration calibrates every stream
individually before averaging (`evaluate_gpu_oof.py::CalibratedFoldCache.stream` fits
`cal` on the train folds and applies it to the eval fold, and only then does `blend()`
average). `SCDF_NSEEN` is a quantile-grid CDF transform (`quantile_grid` +
`np.searchsorted`), so it depends only on the *ordering* of the stream.

Confirmed empirically: `SCDF_NSEEN(RT-413)` is bitwise identical to
`SCDF_NSEEN(sigmoid(RT-413))` and to `SCDF_NSEEN(3·RT-413 + 7)`. **[verified]**

Recorded so it is not re-investigated, and so that a naive "all controls must be in
[0,1]" preflight is not added — such a check would fail a correct control.

**Verdict: all eight controls PASS. No reason to halt on control grounds.**

---

## 8. What this means for C1 — recommendation

Run **Branch B**, once, with the controls shipped in the same push.

1. Stage the eight arrays at `research/oof/{ID}.npy` in the submission worktree
   (untracked, `.gitignore`d, **never committed**). Extract from
   `H1_CONTROL_OOF_PACKAGE.tar` at the repo root — its members already carry that prefix.
2. Leave `resources/` **empty**. (§5.)
3. `crunch push --dry` and confirm all eight `research/oof/*.npy` appear in the manifest
   and the total size is acceptable. (§6.)
4. Real push. Re-verify the eight sha256 hashes against the manifest immediately before
   pushing — the arrays must be byte-identical to §7.
5. The run regenerates `RT-1258`/`RT-1259` from `FROZEN_GPU_CONFIG.json` (hash-enforced in
   `train()`, **not weakened**), **no parameter changed**, both learners complete before
   either is evaluated, and `_try_binding_replacement_test()` then finds the controls and
   computes the full `E0`/`E1`/`E2` battery **in-session**.
6. Capture the block between `=== GPU_TABULAR_OOF_RESULTS_BEGIN ===` and
   `=== GPU_TABULAR_OOF_RESULTS_END ===`. That is **H4**.

No new RT IDs. No tuning. No combination. Recorded in `RDOF_LEDGER.md` as a **technical
recovery**, not a new arm.

---

## 9. Answers to C1.1 as asked

> **Can the `RT-1258`/`RT-1259` model artifacts from submission `76357` / task
> `run-3e834e0f` be mounted or reused by a subsequent run?**

Not by a new submission. A new submission's model is exactly what `crunch push` uploads
from the local `resources/`, which is empty; there is no mount or inherit mechanism. They
would persist automatically into a re-run *of submission 76357 itself*, but that is
platform-scheduled and `train()` only executes under `force_first_train`. The remaining
route — downloading and re-pushing them — is prohibited.

> **If so, by what mechanism, and does the reused artifact land at a readable path?**

Not applicable for a new submission. For a same-submission re-run the mechanism is
`_download_files(self.run.model → model_directory_path)` and the artifacts would land at
`model_directory_path/gpu_tabular_oof/`, which is exactly where
`_persisted_output_root()` and `--resume` look.

> **Does `resources/` persist across runs, or is it rebuilt from the submitted tree each
> time?**

Neither, strictly. It is **not** rebuilt from the submitted code tree — the code walk
explicitly excludes it. It is populated from the server's per-run model registry and
written back at the end of each run, so it persists **within a submission's run chain**
and starts from whatever `crunch push` uploaded for a **new** submission.

> **Do not assume a new submission automatically inherits the previous submission's
> uploaded model artifacts.**

It does not. Confirmed in `push.py`. The optimistic assumption would have been wrong.

---

## 10. Status

- **C1.1: complete.** Branch A closed, Branch B authorized at 4.545 GPU-hours.
- **H1: received and independently verified. PASS.** C1 is unblocked.
- **Open before the run:** total submission size vs the server cap (§6) — resolved for
  free by `crunch push --dry`, and failing at push time rather than after the run.
- **No IDs allocated. No compute spent. No parameters changed. `STATUS.md` untouched.**
