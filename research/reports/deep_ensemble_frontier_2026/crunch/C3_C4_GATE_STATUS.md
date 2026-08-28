# C3 AND C4 — GATE STATUS: BOTH DECLINED

**Lane:** CRUNCH. **Date:** 2026-08-28.
Written per `CODEX_CRUNCH_LANE_PROMPT.md` §7.9: *"C3 and C4 only if their gates open — and
if they do not, decline them in writing with the gate result that declined them."*

**No RT IDs allocated. `RT-1270`–`RT-1289` and `RT-1290`–`RT-1319` remain entirely
unallocated.**

---

## C3 — Cross-family slot mixing: **DECLINED, both conditions unmet**

C3 opens only if **both** hold.

### Condition 1 — LOCAL's best-`k` hybrid reaches ≥ +0.0015, ≥4/5 folds positive, dominant pair net > 0

**Not yet determined.** CSA-04 has not returned; **H2** has not arrived. Cannot be
evaluated.

### Condition 2 — a non-LightGBM, non-CatBoost family has a measured single-slot `marginal_vs_clone` ≥ +0.0010

**Fails now, and no in-flight work would satisfy it except C1.**

Searched `research/RESULTS.csv`, all ledgers, and all reports. Exactly one XGBoost result
exists in the entire repository:

```
RT-A09-FAM-xgb-B  2026-08-18  agent09  xgboost  FAM-xgb
  modules m00_core,m03_dyn   n_features 211
  mean_oof_ts_auc 0.5987583471719681
```

This is a **standalone TS-AUC** on a two-module 211-feature bank. `RESULTS.csv` has no
`marginal_vs_clone` column at all — it records standalone scores. **XGBoost has never been
evaluated as an ensemble slot replacement in this repository**, so there is no single-slot
marginal for it, let alone one ≥ +0.0010. It also ran on a different feature bank from the
500-column matched contract, so it is not comparable even as a prior.

The only live route to condition 2 is **C1** (TabM / RealMLP), which has not run.

### The open question I was assigned: is `research/xgb-gpu-2026` live or a stub?

**Neither — the branch does not exist.**

```
git branch -a --list "*xgb*"        -> empty
git ls-remote origin | grep -i xgb  -> empty
```

Not local, not remote, not in reachable history. `LANE_CRUNCH.md` §301 and
`PROGRAM_PLAN.md` §747 state it "appears in the remote snapshot"; `git ls-remote` against
`origin` does not confirm that. Per `AGENTS.md` — *"if a memory or prompt names a branch
that no longer exists, trust the repository over the prompt and say so explicitly"* — the
snapshot claim is **stale or mistaken**, and **C3 has no XGBoost artifacts to build on.**

The 27 branches that do exist on `origin` are listed by `git branch -r`; none is an
XGBoost arm.

**Consequence:** even if CSA-04 clears condition 1, C3 would still be blocked on condition
2 unless C1 delivers a qualifying neural single-slot marginal. Standing up an XGBoost arm
from scratch is **new work requiring its own preregistration and IDs** — it is not the
"reuse existing artifacts" path the brief anticipated.

**A note on scope, for whoever opens C3 later.** The brief's constraint that C3 be a
*fixed enumerated assignment, not a search*, is now better supported than when it was
written: C2 measured that slot family choice is free on deployment cost at every `k`
(3.0× headroom even at k=7). That removes a natural brake. Nothing but the multiplicity
argument and `RDOF_LEDGER.md` restrains a 7×4 = 16,384-configuration search, so the
enumerated-assignment constraint should be treated as load-bearing.

---

## C4 — Neural sequence models: **DECLINED, trigger absent**

C4 opens on **H3** — the LOCAL agent's arbitration probe finding a gating rule that
retains ≥50% of the prior neural arm's repairs at a <0.05 damage rate.

**H3 has not arrived.** `L3_ARBITRATION_PREREG.md` exists on the LOCAL branch, so the
probe is preregistered and presumably in flight, but no result exists.

Checking the other three triggers in the brief's table:

| Trigger | State |
|---|---|
| **H3 succeeds** — the main path | **Not returned.** No result. |
| **C1 returns TabM ≥ +0.0010** | **Not run.** C1.1 established it costs 4.545 GPU-h and it has not been launched. Explicitly *"not sufficient alone"* regardless. |
| **CSA-04 KILL across all four slots and `k* = 2`** — opens C4 by elimination | **Not returned.** |
| **C1 KILL and H3 fails** → **C4 should not be funded** | Cannot be evaluated; neither input exists. |

**No trigger is satisfied. C4 does not open.** Per §C4.4, generic TabM/RealMLP
hyperparameter search before C1's binding result exists is out of scope, so there is no
preparatory neural work available either.

I have started nothing neural, allocated nothing from `RT-1290`–`RT-1319`, and written no
neural preregistration — consistent with §8's *"do not start a neural model until H3 opens
C4."*

**One thing worth recording for whenever C4 is assessed.** §C4.1(6) says the real cost of
a neural arm is the preflight, not the training. C2 adds a second cost that was previously
unquantified: swapping merely *one tree library for another* — LightGBM to CatBoost, same
inputs, same single-row predict shape — costs **3.27× per slot**. A causal selective SSM
stepped at every online timestep is a different order of computation again. C4-1's
deployment profile should be estimated *before* the 24-gate preflight is built, not after,
because it is the cheaper thing to be wrong about.

---

## Status

- **C3: declined.** Condition 2 fails outright (no non-LGBM/non-CatBoost single-slot
  marginal exists anywhere); condition 1 undetermined pending H2. `research/xgb-gpu-2026`
  does not exist.
- **C4: declined.** No trigger satisfied; H3 not returned.
- Both remain re-openable — this records the gate state on 2026-08-28, not a permanent
  verdict.
- **`RT-1270`–`RT-1289` and `RT-1290`–`RT-1319`: unallocated.**
