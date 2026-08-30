# SOURCES — Causal Representation Frontier

External sources actually used to make a design decision in
`CAUSAL_REPRESENTATION_FRONTIER.md` and `CRF_PROGRAM_PREREG.md`. Nothing here is
decoration: each entry names the mechanism borrowed, the legal input it maps to,
why the repository audit shows it is not already tested, and the failure mode it
addresses. No generic deep-learning surveys.

Accessed 2026-08-25.

---

## S1. van den Oord, Li, Vinyals — *Representation Learning with Contrastive Predictive Coding* (2018)

arXiv:1807.03748 · <https://arxiv.org/abs/1807.03748>

**Mechanism borrowed.** Learn a representation by predicting *in latent space*
with an autoregressive summary of the past, rather than reconstructing raw future
observations — the observation that reconstructing the future forces the encoder
to model nuisance detail that carries no task-relevant information.

**Maps to.** The diagnosis of why Wave 8's CFEP lost to its own BCE control:
CFEP's target was a *future handcrafted summary* (`SST` channels at `t+200`)
whose present value is already one of the 500 columns the downstream model
received. CPC's argument predicts exactly that outcome — the target was easy and
label-irrelevant.

**Legal input.** No future observation reaches inference; this influences only how
a pretraining target is *chosen*, and CRF-02 uses it in the negative direction —
by pretraining on the **conditional null of the present** rather than on any
future quantity at all.

**Not already tested.** CFEP is the only self-supervised pretraining ever scored
here, and it used a future point target with a frozen encoder appended to the
saturated bank (`wave8_cfep.py`, `wave8_cfep_classify.py`). No contrastive or
latent-space predictive objective appears anywhere in `RESULTS.csv`.

**Failure mode addressed.** "Self-supervised objective is easy but orthogonal to
the label" — the CFEP failure this program must not repeat.

---

## S2. Yue et al. — *TS2Vec: Towards Universal Representation of Time Series* (AAAI 2022)

arXiv:2106.10466 · <https://arxiv.org/abs/2106.10466>

**Mechanism borrowed.** Timestamp-level (not instance-level) contextual
representations, learned hierarchically so that every time index carries its own
embedding usable for downstream per-timestamp tasks.

**Maps to.** This competition scores **one prediction per online time step**, so
any representation must be per-timestamp, not per-series. It is the direct
argument for a per-timestep head over a sequence encoder (CRF-01's design) rather
than a series-level embedding followed by broadcasting — the latter being
structurally what D3R Arm C did illegally and what `m05_ctx` did harmfully.

**Legal input.** Per-timestamp representations computed left-to-right are exactly
prefix-invariant; TS2Vec's own hierarchical contrasting is *not* adopted, because
it samples overlapping subseries and would require future context.

**Not already tested.** No timestamp-level representation-learning objective
exists in the ledger; `RT-970/971` are per-timestep but supervised by BCE only.

**Failure mode addressed.** Series-level embeddings that must be broadcast — the
`m05_ctx` memorisation failure (deranged control −0.057).

---

## S3. Gu, Goel, Ré — *Efficiently Modeling Long Sequences with Structured State Spaces (S4)* (ICLR 2022)

arXiv:2111.00396 · <https://arxiv.org/abs/2111.00396>

## S4. Gu, Dao — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)

arXiv:2312.00752 · <https://arxiv.org/abs/2312.00752>

**Mechanism borrowed.** A **constant-size recurrent state** during autoregressive
decoding — linear scaling in sequence length with no cache that grows with
context — as the property that makes a sequence model deployable in a streaming
setting.

**Maps to.** §O's streaming-cost analysis and §N's architecture rejection. This is
the criterion on which a compact causal Transformer is rejected: its KV cache grows
with `t`, so per-series state is unbounded over a 999-point online segment across
~2,000 concurrent series. It is also why SSMs are named as *the* follow-up
architecture if CRF-01's receptive field of 127 is shown to bind.

**Legal input.** Recurrent-mode inference is inherently causal and per-series.

**Not already tested.** No SSM has been run; `research/STATE_OF_RESEARCH.md`
MOONSHOT 3 names "State-space / Mamba-style causal sequence model … trained with a
within-timestep ranking loss" and it has never been opened. Deliberately **not**
selected now: unverified determinism against the repository's ≤1e-8 Gate 5, and a
new dependency, both of which would confound the §M ablation.

**Failure mode addressed.** Designing a mechanism that cannot fit the 15 h /
1.734 ms-per-point deployment envelope.

---

## S5. Moustakides & Basioti — *Training Neural Networks for Likelihood/Density Ratio Estimation* (2019), with density-ratio change detection

arXiv:1911.00405 · <https://arxiv.org/abs/1911.00405>
See also *Deep density ratio estimation for change point detection*,
arXiv:1905.09876 · <https://arxiv.org/abs/1905.09876>

**Mechanism borrowed.** Estimate the **ratio** of densities between regimes
directly, rather than estimating each density and dividing — and the associated
result that cross-entropy / symmetric-KL trained ratio estimators give strong
AUC for change scoring.

**Maps to.** `STATE_OF_RESEARCH.md` MOONSHOT 1 states the theoretically correct
object for this metric is a likelihood ratio between NOT-YET-BROKEN and
ALREADY-BROKEN latent states, and that *"nothing we built approximates it
directly"*. CRF-01's same-`t` pairwise logistic objective is, up to a monotone
transform, a density-ratio estimator between the same-`t` positive and negative
populations — which is the formal reason to prefer it over rowwise BCE here, and
not merely a metric-alignment argument.

**Legal input.** The ratio is learned from training pairs; inference emits one
scalar per series from its own prefix. Cross-sectional ranks are never used at
inference.

**Not already tested.** All seven prior ranking experiments were LightGBM on the
static 500-vector (§G); none learned a representation jointly.

**Failure mode addressed.** Rowwise BCE optimising an elapsed-time channel that
TS-AUC deletes within a timestep (`RT-960`: training BCE 0.077, dev TS-AUC 0.5706).

---

## S6. Sun, Zhang, Zhang, Ren, Cai — *Enhancing Personalized Ranking with Differentiable Group AUC Optimization* (PDAOM) (2023)

arXiv:2304.09176 · <https://arxiv.org/abs/2304.09176>

**Mechanism borrowed.** Optimise AUC **within groups** rather than globally, using
a differentiable pairwise surrogate over the score difference, because the
deployed metric is a *grouped* AUC and global AUC is a different objective.

**Maps to.** TS-AUC is precisely a group AUC with the group being the online index
`t`. This supplies the grouping rule (group = `t`, pairs drawn inside the group)
and the surrogate choice (logistic on the score difference). It also supports the
preregistered decision to use **uniform** pair weighting rather than the metric's
own `n_pos(t)·n_neg(t)` weighting — `RT-700` tested metric-shaped weighting on
trees and lost 0.00147, and group-AUC work generally optimises groups uniformly.

**Legal input.** Grouping by `t` is a *training-time* construction from training
labels; at inference the model scores each series independently and never sees the
current cross-section.

**Not already tested in this form.** `RT-A09-OBJ-lambdarank` and `-xendcg` used
LightGBM's listwise objectives on the static bank and both lost; no differentiable
grouped-AUC surrogate has been trained end-to-end with a representation.

**Failure mode addressed.** Optimising row-level separation instead of same-`t`
ordering.

---

## S7. Rothfuss, Ferreira, Walther, Ulrich — *Conditional Density Estimation with Neural Networks: Best Practices and Benchmarks* (2019)

arXiv:1903.00954 · <https://arxiv.org/abs/1903.00954>

**Mechanism borrowed.** Practical guidance for conditional density estimation on
heavy-tailed financial-style series: noise regularisation stabilises training, and
mixture/kernel networks estimate higher moments and quantiles more reliably than
classical KDE — together with the observation that MDNs are the fragile end of the
family.

**Maps to.** CRF-02's head choice. The program selects a **21-knot monotone
quantile head under pinball loss** over a mixture-density network, precisely
because this program forbids post-score tuning and an MDN's instability would
force it. The quantile head also yields the predictive PIT directly by
interpolation, which is the object the sequential statistics consume.

**Legal input.** The density is conditioned on `H_i` and `x_<t` only.

**Not already tested.** The repository's null models are AR(2) shared context,
per-series frozen AR(2)-state Kalman (`RT-1214`), delay-16 rank-4 Hankel-DMD
(`RT-1215`), GARCH(1,1) (scored 0.50012 — literally zero signal because the filter
adapts on the same timescale as the break), and fixed historical-window `NullCal`.
None is a learned conditional **distribution**.

**Failure mode addressed.** GARCH's documented failure — a per-series adaptive
filter that absorbs the break it is meant to reveal. An amortized global null
fitted on **break-free histories only** cannot adapt to a break it never saw.

---

## S8. Downey, Hefny, Li, Boots, Gordon — *Predictive State Recurrent Neural Networks* (NeurIPS 2017)

arXiv:1705.09353 · <https://arxiv.org/abs/1705.09353>
Foundational: Littman, Sutton, Singh, *Predictive Representations of State*
(NeurIPS 2001) ·
<https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html>

**Mechanism borrowed.** Represent state as a set of **predictions about future
observables** rather than as a latent code with no observable grounding — giving a
state that is statistically identifiable and cheap to update recursively.

**Maps to.** CRF-02's online state is a predictive object by construction
(predictive quantiles, PIT, log score, cumulative discrepancy) rather than an
uninterpretable embedding. That matters here for a specific repository reason:
`m05_ctx` proved that an unconstrained series-level code becomes a series
*identifier*, and a predictive-state parameterisation is grounded in observables
the model must actually get right, which is a structural bottleneck rather than a
regularisation hyperparameter.

**Legal input.** The predictive state at `t` is a function of `H_i` and
`x_<t` only. Predictions about the future are *emitted*, never *read*.

**Not already tested.** No predictive-state or latent-filtering representation
appears in `RESULTS.csv`; a repository-wide search for "predictive state" returns
zero hits outside aspirational text.

**Failure mode addressed.** Learned history embeddings collapsing into series
identity (`m05_ctx` −0.0177 at full scale, deranged control −0.057) — which is
also why the derangement control on `h_i` is a mandatory CRF-02 gate rather than
an optional diagnostic.

---

## Sources deliberately NOT used

- Time-Frequency Consistency (TF-C) and other augmentation-based contrastive
  time-series methods: they require augmentations whose invariances are assumed,
  and this data's break signal *is* a scale/dependence perturbation — the standard
  augmentation set would destroy the label.
- SimTS (arXiv:2303.18205) and forecasting-oriented contrastive frameworks: the
  task here is same-`t` ranking, not forecasting, and CFEP already measured that a
  forecasting-shaped objective loses to BCE on this data.
- Generic deep-learning-for-time-series surveys: excluded by the brief's own rule
  and by this program's rule that a citation must change a design decision.
