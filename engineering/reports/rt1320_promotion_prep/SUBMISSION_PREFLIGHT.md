# RT-1320 submission pre-flight

Date: 2026-09-02. Purpose: prove the artifact runs clean before spending scarce
Crunch quota on it. Every check below was run locally.

## The artifact

Rebuilt from current HEAD `05a6b1c` so that **the shipped bytes are the tested
bytes**. This mattered: the copy sitting in `submissions/` was `19fb2d4d…` while
the one that passed the issue #20 Crunch test was `fd2a56b6…` — same byte count,
different content, differing only in the embedded `built_at`. Neither had been
tested in the state it was in.

```
entrypoint sha256   d6447e025a2b9dbcb45b2440c7705fd0e4180f86e1391ce8a144c10286bd9fda
source_zip_sha256   6a36e641b7ad276703f3af4d5165bc4eb5c4b79e5ac4decdbc37d6c5e43a137c
model_zip_sha256    2282ffeeed72f65d782b7a9697e7796b06062a08bea6ed01264de611aa35b0da
feature manifest    1646c3b9e09d8a7fb3c564483a1d1d999680caeefe848b092f907a6cac80cced
embedded_source_clean = true
```

Source and model zips match the issue #20 build exactly; only the entrypoint hash
moved, as it must with a new timestamp.

## The builder's warning, resolved

`build_submission.py` warns that the model was trained at `6b4fafac` while this
build is at `05a6b1c`, and that this is "NOT fine if src/sbr changed". It did
change. The full provenance chain:

| span | `src/sbr` files changed |
|---|---|
| `6b4fafac` (model manifest) → `e50098a4` | 0 |
| `e50098a4` → `f1912b6` | 34 |
| `f1912b6` → `05a6b1c` (today) | 0 |

The single moving leg is exactly the one RT-1257 gate item 1 measured on
2026-09-02: **0 changed predictions out of 89,706**, across the 67-series
packaged battery and the 100-series Crunch reduced set. So the warning is
answered by evidence rather than assumption.

## Checks

| # | check | result |
|---|---|---|
| 1 | Import on a read-only tree (crunch's `load_user_code`) | PASS, 0.4s |
| 2 | Embedded payload integrity | PASS — "source and model verified by sha256" |
| 3 | Payload unpacks outside the code tree | PASS — `tempfile.mkdtemp`, not `./_sbr_payload_<pid>` |
| 4 | `train()` installs the artifact | PASS — 8 model files: `model.cbm.0/4`, `model.txt.1/2/3/5/6/7` |
| 5 | `infer()` accepts a **generator** online stream | PASS |
| 6 | One score per online point, order preserved | PASS — 40/40, and 45/45 over three mixed-length series |
| 7 | All scores finite and in [0, 1] | PASS |
| 8 | Prefix invariance | PASS — first 10 scores identical whether 10 or 40 points follow |
| 9 | `INFER_PARALLELISM` | 1 |
| 10 | `crunch test` run 1 | PASS, 02:27, 2.17 GB |
| 11 | `crunch test` run 2 | PASS |
| 12 | CLI determinism check | passed on both runs |
| 13 | Full prediction replay, run 1 vs run 2 | **bitwise identical**, 0/50,983 changed |
| 14 | Predictions vs the issue #20 validated build | **identical**, sha `89d47345…` |

Checks 3 and 5 are the two failure modes that actually killed submissions in this
programme, so both were verified structurally *and* functionally:

- **Check 3** — an RT-1257 deployable built before `733c727` died at import with
  `PermissionError: '/context/code/_sbr_payload_110'` because `/context/code` is
  read-only. This build uses `tempfile.mkdtemp`.
- **Check 5** — submission #17 completed a 2h15m training run and then died with
  `TypeError: float() argument must be ... not 'generator'` because `infer` called
  `np.asarray` on the online stream. This build iterates it lazily.

## A real constraint, quantified

The streaming engine's emitted row width is **history-length dependent**:

```
n_hist <= 511  ->  486 columns   (members index to 499 -> IndexError)
n_hist >= 512  ->  500 columns
```

A series with fewer than 512 historical points would crash inference with
`IndexError: index 486 is out of bounds`. Checked against the actual data:

| dataset | series | min `n_hist` | below 512 |
|---|---|---|---|
| `X_test.reduced.parquet` | 100 | 1,002 | **0** |
| `X_train.parquet` | 10,000 | 1,000 | **0** |

Minimum observed history is 1,000, roughly a 2× margin over the threshold. Safe
on this data release. Worth re-checking if a future release changes the minimum
series length, because the failure mode is a hard crash mid-inference rather than
a degraded score.

## Status

The artifact is ready to submit. What it does **not** carry is an external score;
that is the point of submitting. Local evidence stands at four dev partitions
(mean E2−E1 +0.0017678, 4/4 positive) and one 100-series held-out read of
−0.0078, which the preregistered rule records as uninformative at roughly one
standard error.
