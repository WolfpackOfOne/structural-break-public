# C2-D — `m07_bayes` REDUNDANCY AUDIT: ABSORBING-STATE BOCPD

**2026-08-22, branch `research/wave5-alpha`. No competition data. No TS-AUC.**

**VERDICT: REJECTED — the proposed family already exists, in full.**

---

## 1. THE PROPOSAL

C2-D proposed that textbook BOCPD answers the wrong question — it models
"change now / run length", while the competition asks *has a persistent break
already occurred?* — and that the right formulation is a hidden state

    PRE  ->  POST      (absorbing, no transition back)

with outputs `P(post by t)`, posterior change age, posterior entropy,
hazard-weighted evidence, posterior persistence, predictive likelihood ratio,
max posterior tau, and concentration of the tau posterior.

## 2. WHAT `m07_bayes` ALREADY IS

The module's own first line:

> `m07_bayes` -- generative / sequential Bayesian evidence for an **ABSORBING** break.

And its first construction bullet:

> the latent state NOT-BROKEN → BROKEN is **ABSORBING**, so the exact filtering
> recursion for `P(broken at t | x_1:t)` collapses to one log-space accumulator
> per alternative parameter value:
>
>     L_t(th) = logaddexp(L_{t-1}(th), log h) + llr_t(th) - log(1-h)
>
> **There is no dyadic approximation and no candidate-changepoint scan: the
> absorbing structure integrates over every changepoint exactly**, in O(1) per
> step per mixture component.

**50 columns across five families:**

| prefix | n | what |
|---|---|---|
| `ab` | 18 | the **absorbing** PRE→POST posterior, plus its calibrations, family decomposition, peak, slope, fast/slow contrast |
| `abz` | 3 | absorbing recursion on the deliberately un-Gaussianised raw channel |
| `bo` | 10 | Adams–MacKay BOCPD run-length posterior — mass below k, mode, mean, entropy, no-change odds |
| `ev` | 12 | e-values / test martingales (GRAPA plug-in, mixture, Vovk power martingale) |
| `bf0` | 5 | sequential Bayes factors for "break at the first online point" |
| `xb` | 2 | cross-channel maxima |

## 3. PROPOSAL → EXISTING COLUMN MAPPING

| proposed output | already in `m07_bayes`? | column(s) |
|---|---|---|
| `P(post by t)` | **YES** | `ab_lpo` — log posterior odds of BROKEN under the absorbing model; `ab_lpo_z`, `ab_lpo_sur` calibrated |
| posterior change age | **YES** | `bo_mean_rel` (posterior mean run length / t), and the mode channel |
| posterior entropy | **YES** | `bo_ent`, `bo_ent_z` |
| hazard-weighted evidence | **YES** | the `log h` / `log(1-h)` hazard terms are in the absorbing recursion itself |
| posterior persistence | **YES** | `ab_post_lvr`, `ab_post_rho`, `ab_lpo_rel`, `ab_fast_slow` |
| predictive likelihood ratio | **YES** | `bf0_all`, `bf0_var`, and the `ev_*` e-process family |
| max posterior tau | **YES, and strictly better** | the absorbing recursion **integrates over every tau exactly**; a max is an approximation to that |
| concentration of tau posterior | **YES** | `bo_p_lt10`, `bo_p_lt25`, `bo_ent` |
| robust predictive model (Student-t / variance-adaptive) | **YES, by a different and arguably stronger route** | the primary stream is AR(6)-whitened then **normal-scored against the historical residual ECDF**, so the per-series null is exactly N(0,1) *by construction* — a rank-based robustification that removes the tail-heaviness bias a Student-t likelihood would only soften |

**Nine of nine proposed outputs already exist.** One of them (integration over
tau) exists in a *stronger* form than proposed.

## 4. THE DISTINCTION THE PROPOSAL ASSUMED WAS MISSING IS ALREADY DRAWN

The proposal's premise was that BOCPD's resettable run-length state is the wrong
model for an absorbing break. `m07_bayes` makes exactly that argument itself, and
resolves it by carrying **both** models deliberately:

> Unlike the absorbing model it allows repeated resets, **so the two disagree
> exactly on transients.**

The disagreement between the absorbing arm and the BOCPD arm is the transient-vs-
permanent signal, and both arms are already emitted so a tree can use the
contrast. Building `m14_absorb` would re-derive a distinction the module was
designed around.

## 5. VERDICT AND COST

**REJECTED before any implementation.** No `m14_absorb`. Rebuilding it would be
the precise failure mode that produced `tl_` (−0.00410) and `rk_` (−0.00273):
re-stating an existing channel under a new name, paying the split-search variance
cost, and gaining nothing.

Cost of this decision: one file read. Zero training runs, zero compute, zero
degrees of freedom.

## 6. WHAT WOULD REOPEN IT

Not a new Bayesian detector. The only genuinely open items in this area are the
ablation `FAILED_EXPERIMENTS` already records as unspent (attribution of
`m07_bayes`'s 0.599 across its absorbing / BOCPD / e-process / stream
components), and that is an **ablation of an existing module**, not a new family.
It is out of scope for C2 because it requires competition data to score.
