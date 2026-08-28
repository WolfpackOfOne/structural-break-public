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

**FAILS — and this is now a measurement, not an absence.** C1 has reported: TabM
`+0.0000409`, RealMLP `−0.0032237`, both KILL, both far below `+0.0010`
(`H4_GPU_TABULAR_RESULT.md`). These are the first actual measurements of condition 2 for
any non-LightGBM, non-CatBoost family, and both fail.

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

The only live route to condition 2 was **C1**, and it has now run and failed.

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

## C4 — Neural sequence models: **DECLINED — main path now falsified, not merely absent**

> **UPDATE, 2026-08-28 (later same day): H3 has returned, and it FAILED.**
> LOCAL commit `5a116d9`, `local/L3_ARBITRATION_PROBE.md`, verdict
> **`FAIL_NO_RETENTION_MECHANISM`**. This changes C4's status from *"gate not yet
> evaluated"* to *"the main gate has been evaluated and it is shut."*

C4's main path opens only if the LOCAL arbitration probe finds a gating rule retaining
**≥50% of the prior neural arm's repairs at a <0.05 damage rate**. Fourteen rules were
tried. **None clears both thresholds**, and the table shows why the two constraints are in
direct tension:

| rule | retained repairs | mature-prebreak damage rate | why it fails |
|---|---:|---:|---|
| `confidence_q10` | 0.251 | 0.0221 | damage fine, retention less than half the bar |
| `confidence_q20` | 0.454 | 0.0484 | **closest** — both just miss |
| `dominant_confidence_q20` | 0.454 | 0.0484 | same, on the dominant cell |
| `confidence_q30` | 0.656 | 0.0751 | retention clears, damage 1.5× over |
| `three_way_abstain_q20` | 0.826 | 0.1461 | retention clears, damage ~3× over |
| `unconditional_equal` | 1.000 | 0.1170 | keep everything, damage 2.3× over |

Retention and damage move together across every rule family tried — confidence
thresholds, RT-600 boundary distance, agreement direction, dominant-cell restriction, and
three-way abstention. **No rule separates the repairs from the damage.** The probe's own
conclusion: *"direct evidence against opening another neural detector without a new
retention mechanism."*

This is fold-0 only and descriptive — it cannot promote a model — but it is being used as
a *negative* gate, which is exactly what a descriptive probe can legitimately do.

Trigger table, current state:

| Trigger | State |
|---|---|
| **H3 succeeds** — the main path | **FAILED.** `FAIL_NO_RETENTION_MECHANISM`, 0/14 rules. **Main path closed.** |
| **C1 returns TabM ≥ +0.0010** | **FAILED.** TabM returned **+0.0000409**, ~24× short. RealMLP **−0.0032237**. Both KILL. |
| **CSA-04 KILL across all four slots and `k* = 2`** — opens C4 by elimination | **Not returned.** CSA-04 is mid-training (`CSA04_TRAIN_LOG.json`, CAT-411 arm). |
| **C1 KILL and H3 fails** → **C4 should not be funded** | **SATISFIED IN FULL.** H3 failed; C1 returned KILL on both learners. The brief's instruction for this state: **do not fund C4; redirect to deployment robustness.** |

**C4 does not open.** Nothing neural started, no prereg written, `RT-1290`–`RT-1319`
untouched.

**RESOLVED 2026-08-28: C4 should not be funded.** Both independent lines have now
reported and both are negative — a tabular-neural family test at the binding endpoint
(`H4_GPU_TABULAR_RESULT.md`) and a fourteen-rule retention probe. This is the exact state
the brief names, and its instruction is explicit: **redirect to deployment robustness.**

**The sharpest single finding is TabM's.** `LANE_CRUNCH` §C4.1(1) records the prior neural
program reaching **0.59276**, *"still below the `0.600` necessary condition"* — which left
open the reading that clearing 0.600 was the blocker. **TabM cleared it at 0.60065 and
returned `marginal_vs_clone = +0.00004`**, 1/59th of `RT-1257`. It also had ρ 0.585 against
the clone, genuinely more decorrelated than the standalone/ρ regression predicts. It was
both good enough and different enough, and it still contributed nothing. "Raise standalone
past 0.600" is therefore closed as a live hypothesis, not merely unproven.

The remaining theoretical route — CSA-04 returning KILL on all four slots with `k* = 2`,
opening C4 *by elimination* — is now **outweighed by an explicit instruction not to fund
it**. If anyone reopens C4 on that basis, they are overriding a satisfied stop condition,
and the brief requires that be **said explicitly**: opened by elimination, against
evidence, not by it.

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
- **C4: DO NOT FUND.** Decided, not pending. H3 returned `FAIL_NO_RETENTION_MECHANISM`
  (0 of 14 rules) and C1 returned KILL on both learners (TabM +0.0000409, RealMLP
  −0.0032237). The brief's "C1 both KILL and H3 fails" stop condition is satisfied in
  full: redirect to deployment robustness.
- Both remain re-openable — this records the gate state on 2026-08-28, not a permanent
  verdict.
- **`RT-1270`–`RT-1289` and `RT-1290`–`RT-1319`: unallocated.**
