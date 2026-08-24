# novel_streams — one plugin API, one evaluation pack

Infrastructure for testing new causal mechanisms against RT-600 without
rebuilding the competition pipeline. Designed in
[`../../NEW_AVENUES_2026.md`](../../NEW_AVENUES_2026.md) §N.

**Nothing here has been run as an experiment.** No `RT-*` ID is consumed by
any file in this directory, and none of them appends to `research/RESULTS.csv`.
`d1`–`d6` are descriptive diagnostics whose results are in
`../../reports/new_avenues_2026_diagnostics.json`; their pre-declaration is in
[`PREDECLARE.md`](PREDECLARE.md).

## Setup

Importing needs nothing:

```python
import sys; sys.path.insert(0, "research/scripts/novel_streams")
import harness            # resolves its own checkout from __file__
```

Running a mechanism needs the competition caches, which are not redistributed.
Point `SBR_ROOT` at a tree carrying `cache/store`, `cache/features`,
`research/folds` and `research/oof`; code is still imported from the checkout
`harness.py` lives in, so the two can differ.

```bash
export SBR_ROOT=/path/to/a/worktree/with/caches
python research/scripts/novel_streams/m20_dwell_probe.py
```

The shared evaluation helpers (`wave8_common.ensemble_marginal`,
`wave8_common.pair_repair_stats`) are **already on `research/current`** at
`../wave8_common.py`. Do not copy them from a historical branch and do not
reimplement them. Wave 8 itself remains unmerged; only this function-only
infrastructure was carried forward, and using it does not make Wave 8's five
killed mechanisms part of the active research programme.

## Writing a mechanism

```python
class MyMechanism(harness.StreamingMechanism):
    name = "m21_mine"
    cols = ["a", "b"]
    def fit_history(self, hist):   ...   # per-series state, HISTORY ONLY
    def emit(self, hist, online):  ...   # (n_online, len(cols)) float32
```

Row `t` of `emit` may depend only on `hist` and `online[:t+1]`.

## The causal gate

`harness.verify(mech)` is bitwise prefix invariance at `atol=0.0`: rebuilding on
a truncated online segment must reproduce the surviving rows **exactly**,
including their NaN pattern. Rows before a window has filled must be `NaN` —
never `0.0`, never back-filled from history (`research/PROTOCOL.md` §3).

A mechanism that has not passed `verify()` may not be scored. Pass
`series=[(hist, online), ...]` to run it on synthetic data with no store, which
is how `tests/test_novel_streams_harness.py` exercises the sentinel.

## Pipeline

```
verify()              causal contract, atol=0.0
build()               cached (n_rows, k) float32 memmap -> cache/novel_streams/
diagnostic_pack()     whole fold / dominant cell / negative type / t- / age-buckets
                      + within-t rank correlation with the RT-600 blend
pair_flow_by_cell()   repairs / damage / net, split by cell and negative type
marginal()            RT600 vs RT600+seed-clone vs RT600+candidate   <- BINDING
```

`marginal()` returns `marginal_vs_clone`. **That** is the number that decides,
never the standalone delta and never the gain over RT-600 alone — see
`../../reports/wave7_t2_promotion_final.md` for what happens when the two are
confused (+0.00943 standalone became +0.00024 in the ensemble).

All artifacts land in `<SBR_ROOT>/cache/novel_streams/`, which `.gitignore`
covers. Nothing in this directory writes into the source tree.
