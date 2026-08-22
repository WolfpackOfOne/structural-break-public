# C2-E — HARD-NEGATIVE TRAINING PROTOCOL

**Pre-registered 2026-08-22, branch `research/wave5-alpha`, BEFORE any Wave-5
5-fold outcome exists and before the competition store is present in the
environment. No number below was chosen with knowledge of a score.**

Code: `research/scripts/wave5_hard_negative.py`
Gates: `tests/test_wave5_hard_negative.py` (11 tests, no data required)

---

## 1. THIS IS NOT A FEATURE MODULE

There is no `mXX_hardnegative` and there will not be one. Hard-negative mining
changes the **training distribution**, not the feature vector. It composes with
every module rather than competing with them, and it is therefore evaluated as
a training-protocol arm, not as a stream.

## 2. WHAT A NEGATIVE ACTUALLY IS HERE

The row label is `t >= tau`. So negatives come in **two structurally different
kinds**, and conflating them would be an error:

| kind | rows | why it is hard |
|---|---|---|
| **never-break** | every row of a series with no break | a transient in a stable series that mimics a break |
| **pre-break** | `t < tau` of a series that *does* break | the series really will break; a model that anticipates evidence it cannot yet legally see scores these high |

The second kind exists only because the metric is time-stratified, and it is the
more interesting one: it is where a causally-legal model and a leaky one differ
most sharply.

### Negative taxonomy to report (descriptive, not used for selection)

transient spike · burst · temporary level shift · volatility burst · local trend ·
dependence fluctuation · tail cluster · strong-but-temporary pseudo-break ·
pre-break rows of true positives

## 3. THE FOLD-PURE PROCEDURE

For each outer fold `k` of the canonical 5-fold partition:

```
inner = folds != k                                  # 6,400 series
1. produce OOF scores for inner series by an inner CV that never touches fold k
2. hardness(row) = rank-percentile of the OOF score among INNER NEGATIVES only
3. weights / oversampling from hardness, using the constants pre-registered below
4. train the candidate on inner with those weights
5. score fold k, once
```

**Fold purity is enforced, not asserted.** `mine_fold_pure()` masks fold `k`
before hardness is computed, ranks among inner negatives only, and returns
weight 1.0 for fold-k rows. `test_mining_is_fold_pure` corrupts fold `k`
completely — new random scores and every label flipped — and requires the mined
weights and hardness to be **bitwise identical**. A leak here would invalidate
every number produced under this protocol and would be entirely invisible in a
score.

## 4. PRE-REGISTERED CONSTANTS — FIXED NOW

| constant | value | meaning |
|---|---|---|
| `LAMBDA` | **1.0** | weight arm: `w = clip(1 + LAMBDA * hardness_pct, 1, 2)` |
| `TOP_Q` | **0.20** | oversample arm: duplicate the hardest 20% of negatives once |
| `MIN_W, MAX_W` | **1.0, 2.0** | weights bounded so no row can dominate a split |
| positives | **weight 1.0, always** | never reweighted — see §6 |

Three arms, declared now, no others:

| arm | id | what |
|---|---|---|
| uniform control | `RT-540` | `LAMBDA = 0`, exactly uniform (a test asserts this) |
| hard-negative weighting | `RT-541` | the weight arm |
| hard-negative oversampling | `RT-542` | the oversample arm |

No weight search, no `TOP_Q` sweep, no per-fold tuning. If these values are
wrong the experiment reports that they are wrong.

## 5. FALSIFICATION

H1 is rejected unless, over the five canonical folds, the best hard-negative arm
beats `RT-540` by **> +0.0010 mean** *and* is positive on **≥ 4 of 5 folds**.

**Seed-clone bar applies unchanged.** A training-distribution change is subject
to the same rule as a feature family: if a seed clone of `RT-540` produces the
same blend gain, the arm has demonstrated nothing.

## 6. THE FAILURE MODE THIS PROTOCOL IS BUILT AROUND

The likely way hard-negative training "wins" is by **suppressing scores
everywhere near the decision boundary**, which improves the no-break tail and
destroys early true-break sensitivity. Aggregate TS-AUC can rise while the
mechanism that matters gets worse — and the young buckets are where this
project's weakness already is (TS-AUC 0.513 at age 0–5 against 0.646 at 100+).

Two structural defences, both fixed in advance:

1. **Positives are never reweighted.** Only negatives carry hardness weight, so
   the procedure cannot directly down-weight young positives. A test enforces it.
2. **Mandatory reporting, whatever the aggregate says.** Every arm reports:
   * the **no-break high-score tail** (what fraction of top-scoring rows are
     never-break series), and
   * TS-AUC by post-break age bucket — **0–5, 5–10, 10–20, 20–50, 50–100, 100+**
     — with baseline AUC, candidate AUC, delta and sample count per bucket.

**Pre-committed rejection rule.** An arm that gains aggregate TS-AUC while the
**0–5 and 5–10 buckets do not improve** is **REJECTED**, regardless of the
aggregate. This is the same rule already binding on `m10_perm`, and it exists
because `m09_back` gained in aggregate while making young breaks worse — the
exact inversion of its stated mechanism. Reading such a result as a success once
is a mistake; reading it as a success twice would be a policy.

An arm may only be promoted on an aggregate gain accompanied by young-bucket
gains, or on a young-bucket gain that the official weighted metric also rewards
and whose trade-off is explicitly characterised.

## 7. STATUS

Protocol pre-registered. Code written and gated by 11 passing tests. **Not run —
it requires the competition store, which is absent.** Zero training runs, zero
TS-AUC values.
