# STATE OF RESEARCH — V5

**2026 ADIA Lab / CrunchDAO Structural Break Challenge (Real-Time Edition)**
Written 2026-08-21. Branch `research/wave5-alpha`, parent `research/wave3-integration` @ `17bb5df`.

`VALIDATION_V2.md` and `research/WAVE5_PREREG.md` are binding. Where this file
and the repository disagree, **the repository wins**.

---

## 0. THE HEADLINE

Wave 5 found new causal information, and it is not enough to submit.

Three new feature blocks were built and gated; two of them beat a same-strength
seed clone, which no new feature family in this project had ever done before —
`m09_back` failed that test in wave 3 and the 13-way union failed it in wave 4.
The strongest, `m12_rdep`, adds **+0.00479** standalone on 4/5 folds and beats
its seed-clone blend by **+0.00141**. That is real and it is roughly half of what
the §4 promotion bar demands.

Two things were *disproved* that mattered more than another 0.001:

* **The metric does not reward early detection the way this project assumed.**
  Ages 0–20 carry **11%** of the official pair weight; age 100+ carries **57%**.
  Two of the brief's own premises rest on the opposite belief.
* **Matching the training objective to the metric's pair weighting makes the
  model worse** (−0.00147). So does a hinge (−0.00476). So does any admixture of
  bagging into the specialist blend, at every weight tested.

And the most consequential number in the repository was not produced by wave 5
at all. `codex/oracle-information-frontier-2026` measured that **the legal causal
model already matches or beats a model told where the break is, at every horizon
through h = 150.** The 2025 reproduction that was supposed to calibrate against a
strong teacher **failed its own gate** — its dependencies were never installed
and every rung of its ladder is `blocked`. Teacher distillation is not justified,
and the reason is not lack of time.

---

## 1. WHERE THE EXTERNAL SCORE STANDS

| | |
|---|---|
| LB-001, Crunch public | **0.6268** |
| RT-600 development architecture, canonical partition | 0.62581 |
| internal → external transfer | **flat to slightly positive** |
| leaderboard snapshot supplied | #1 65.10% · #10 64.10% · #25 63.60% · #50 62.91% |
| approximate standing | ~rank 59 |

The validation framework passed its first real external test. That is why wave 5
spent its compute on alpha and not on rebuilding CV.

---

## 2. WHAT IS IMMUTABLE, AND VERIFIED SO

| ref | SHA | state |
|---|---|---|
| `research/wave3-integration` | `17bb5df` | unmoved |
| `claude/rt600-baseline-submission` | `9aaa9b0` | unmoved, worktree clean |
| `codex/reproduce-2025-public-solution` | `422e4b2` | unmoved, read only |
| `codex/oracle-information-frontier-2026` | `5a3b8a0` | unmoved, read only |

`research/wave5-alpha` is strictly ahead of its parent and behind it by nothing.
The 10 GB feature cache is shared **read-only**: the wave-3 worktree still holds
exactly its original eight modules, every wave-5 artifact is local to the wave-5
worktree, and no `RT-7xx` or `wave5_*` file exists in the wave-3 OOF directory.
Wave 5 does not merge itself.

---

## 3. BASELINES, RE-ESTABLISHED ON THIS PLATFORM

Same session, same folds, same scorer.

| arm | id | TS-AUC | matches |
|---|---|---|---|
| A single control | `RT-300` | **0.61605** | V4 §3 exactly |
| B seven seed clones | `RT-421` | **0.62164** | V4 §5 exactly |
| C seven specialists (**RT-600's architecture**) | `RT-420` | **0.62581** | V4 §4 exactly |

Bagging **+0.00559**, specialisation **+0.00417**. Paired bootstrap of
specialists − seed clones: **+0.00409, CI [+0.00199, +0.00614], 200/200
positive** — W4-E1's three digits, reproduced independently.

---

*(Sections 4 onward — experiments, diagnostics, verdict — are consolidated from
`research/WAVE5_STATUS.md`, which carries the full working record and every
number as it landed.)*
