# 2026 Data Forensics / Break Taxonomy — AGENT 02

**Data used:** `cache/store` restricted to `fold >= 0` in `research/folds/folds.parquet`
= **8,000 dev series** (3,975 break / 4,025 no-break). The lockbox (`fold == -1`) was
never loaded; `X_test.reduced.parquet` was never read. No new split was made.

**Script:** `/home/claude/sb/scripts/agent02_taxonomy.py` (stages `compute` / `analyze` / `resid`).
**Artifact:** `/home/claude/sb/research/artifacts/break_taxonomy.parquet` (8,000 rows x 560 cols:
raw segment stats for HIST / PRE / POST / whole-online, effect sizes vs both references,
calibrated family surprises, `break_class`, `lead_family`, horizon statistics, AR-residual stats).
Seed `20260818`. The intermediate `_taxo_raw.parquet` was deleted after use, so a
reproduction must run `--stage compute` before `--stage analyze` and `--stage resid`.
No file in `src/sbr` was touched. Runtime: 342 s compute + 30 s analyze + 94 s resid,
single process, full 8,000 dev series (no subsampling — the taxonomy rests on all 3,975 break series).

---

## EXECUTIVE SUMMARY (the eight things that matter)

1. **Every historical segment is exactly standardised** — `mean = 0`, `sd = 1.00000` to 5 dp,
   for all 8,000 dev series. The raw competition data ships this way (`scripts/build_store.py`
   does no scaling). Absolute level and absolute scale of `hist` carry zero information;
   only the *online* segment's position relative to (0, 1) is meaningful.
2. **Location breaks essentially do not exist.** AUC of |post-vs-hist mean shift| for
   break vs no-break is **0.4998**; median |mean shift| is 0.054 sd for break series and
   0.053 sd for no-break placebo splits. Any feature family whose story is "the mean moved"
   is fighting for nothing.
3. **Scale is the dominant family, and it is much stronger after AR whitening.**
   Raw log-sd ratio AUC 0.549 (vs hist) / 0.559 (vs pre); **AR(6)-residual log-sd ratio
   0.596 (vs hist) / 0.603 (vs pre)** on a matched subsample. Whitening buys ~+3.5 AUC
   points; going from AR(2) (the library default) to AR(6) buys ~+0.9 more.
4. **The metric is decided between elapsed 20 and 320.** 46 % of all post-break rows lie
   within 160 steps of tau and 73 % within 320; the realistic prefix AUC there is only
   0.53–0.56. Before elapsed 20 (7 % of positive rows) nothing works (AUC ≤ 0.518).
5. **~92 % of break series are individually indistinguishable** from a no-break placebo
   split at p < 0.01 on any of the five families. The excess over the placebo null is only
   3.9 pp of break series. The signal is not a detectable minority plus noise — it is a
   *tiny distributional nudge spread over almost every break series*.
6. **17.0 % of no-break series contain a break-lookalike transient** in their online
   segment (isolated outlier / vol burst / mean-reverting excursion) — but **15.6 % of
   equally long, break-free historical windows contain one too**. Transients are a property
   of the DGP, not of the online period. A naive detector tuned to 5 % FPR recovers only
   ~6 % of breaks.
7. **Generator artifact — the raw variance ratio is biased and the bias tracks tail
   heaviness.** For no-break series the median online/hist log-sd ratio is **−0.018**, and
   it goes from −0.004 (lightest-tail quartile of history) to **−0.063** (heaviest). This is
   a length-mismatch bias (n_hist ≈ 3,000 vs n_online ≈ 500 under heavy tails), not a break.
   Uncalibrated variance features will mis-rank the cross-section.
8. **Generator artifact — tau is *almost* uniform but not quite**: KS vs U(0,1)
   D = 0.0263, p = 0.008; deciles run 10.5 %, 10.6 %, 11.0 % in the first 30 % of the online
   segment against 9.3–9.4 % in the last 30 %. The apparent correlation between effect size
   and tau (rho ≈ +0.23 to +0.33) is **entirely a sample-size artifact** — the placebo
   series show the same correlation (+0.23 to +0.37).

---

## Method note — which comparison was used where

For every series the online segment is split at a cut point: `tau_index` for break series,
and for no-break series a **placebo cut** drawn from the empirical relative-tau distribution
of the break series (seed 20260818). Every statistic is then computed identically for both
groups, so the no-break population is a matched null for everything reported below.

Two comparisons are reported everywhere:
* `e_*` = **POST vs REF**, where `REF = PRE` if `n_pre >= 60` else `REF = HIST`
  (this is the mission's rule; the `ref_used` column records which was taken — 22.9 % of
  break series fall back to HIST);
* `eh_*` = **POST vs HIST** always (available for every series, longer reference).

Where the two differ materially it is stated. For the scale family the PRE reference is
*better* when it exists (0.575 vs 0.561 AUC raw, 0.603 vs 0.596 whitened) — the pre-segment
is closer in length to the post-segment, so the length-mismatch bias of item 7 partly cancels.

---

## A. Sample, reference availability, comparison used

- dev series **8000** (folds 0-4): break 3975 (49.69%), no-break 4025.

- no-break series get a PLACEBO cut drawn from the empirical rel-tau distribution of break series (seed 20260818); every statistic below is computed identically for both groups.

| group | n_pre==0 | n_pre<24 | n_pre<60 -> ref=HIST | n_post<24 | n_post<60 |
|---|---|---|---|---|---|
| break | 0.006 | 0.110 | 0.229 | 0.100 | 0.212 |
| no-break (placebo cut) | 0.007 | 0.110 | 0.218 | 0.099 | 0.224 |


## B. The historical segment is pre-standardised (generator artifact)

|  | hist_mu | hist_sd(ddof=1) | hist_mad/sd | H_kurt | H_acf1 | H_aacf1 | H_volvol | H_pe3 |
|---|---|---|---|---|---|---|---|---|
| count | 8000.00000 | 8000.00000 | 8000.00000 | 8000.00000 | 8000.00000 | 8000.00000 | 8000.00000 | 8000.00000 |
| mean | -0.00000 | 1.00020 | 0.94816 | 9.39085 | 0.07026 | 0.11893 | 0.34308 | 0.99169 |
| std | 0.00000 | 0.00010 | 0.09804 | 102.54676 | 0.28684 | 0.16164 | 0.18421 | 0.02225 |
| min | -0.00000 | 1.00010 | 0.05501 | -1.56011 | -0.98746 | -0.33832 | 0.12537 | 0.70301 |
| 25% | -0.00000 | 1.00012 | 0.92002 | -0.00686 | -0.04015 | 0.01649 | 0.26103 | 0.99403 |
| 50% | -0.00000 | 1.00017 | 0.97640 | 0.22673 | 0.00981 | 0.06402 | 0.29931 | 0.99909 |
| 75% | 0.00000 | 1.00025 | 1.00214 | 1.49646 | 0.18360 | 0.17069 | 0.37785 | 0.99971 |
| max | 0.00000 | 1.00050 | 1.49747 | 4781.78966 | 0.99800 | 0.99389 | 9.74888 | 0.99999 |


**Is a no-break online segment distributionally identical to its own history?**

| whole online segment vs hist | no-break online | no-break online (median) | break online (median) | historical value |
|---|---|---|---|---|
| F_mean_z | 0.0052 | 0.0000 | -0.0011 | 0.0000 |
| F_log_sd_r | -0.0323 | -0.0178 | -0.0086 | 0.0000 |
| F_log_mad_r | -0.0248 | -0.0206 | -0.0074 | 0.0000 |
| F_acf1 | 0.0613 | 0.0141 | 0.0167 | 0.0098 |
| F_kurt | 2.1644 | 0.1024 | 0.1502 | 0.2267 |
| F_skew | 0.0764 | 0.0305 | 0.0257 | 0.0249 |
| F_occ_tail05 | 0.0989 | 0.0953 | 0.0987 | 0.1000 |
| F_occ_center | 0.5072 | 0.5067 | 0.5006 | 0.5000 |
| F_slope_sd | 0.0141 | 0.0071 | 0.0065 | -0.0076 |


## C. Which family carries the signal (whole post-cut segment)


**Series-level AUC (break vs no-break placebo), whole post segment:**

| statistic | |POST vs REF| | |POST vs HIST| |
|---|---|---|
| loc: mean_z | 0.5012 | 0.4998 |
| scale: log_sd_r | 0.5594 | 0.5486 |
| scale2: log_mad_r | 0.5484 | 0.5389 |
| dep: acf1 | 0.5343 | 0.5379 |
| dep2: aacf1 | 0.5225 | 0.5279 |
| shape: kurt | 0.5129 | 0.5219 |
| shape2: skew | 0.5110 | 0.5187 |
| shape3: dOHs_ks | 0.5083 | 0.5083 |
| trend: slope_sd | 0.5177 | 0.5177 |
| trend2: halfdiff_sd | 0.5235 | 0.5235 |
| dist: dOH_ks |  | 0.5278 |
| dist: dOH_w1 |  | 0.5469 |
| dist: dOH_energy |  | 0.5392 |
| dist: dOH_js |  | 0.5258 |
| dist: dOH_hell |  | 0.5258 |
| dist: dOH_cvm |  | 0.5488 |
| dist: dOP_ks | 0.5337 |  |
| dist: dOP_w1 | 0.5583 |  |
| dist: dOP_energy | 0.5481 |  |
| dist: dOP_js | 0.5396 |  |
| dist: dOP_cvm | 0.5452 |  |
| dist: dPH_ks |  | 0.5150 |


**Same AUC stratified by post-segment length:**

| family (vs HIST) | n_post<50 | 50-99 | 100-199 | 200-399 | >=400 |
|---|---|---|---|---|---|
| loc | 0.5125 | 0.5052 | 0.4997 | 0.4914 | 0.5149 |
| scale | 0.5045 | 0.5189 | 0.5355 | 0.5627 | 0.6084 |
| scale-mad | 0.5053 | 0.5250 | 0.5307 | 0.5536 | 0.5846 |
| dep | 0.5107 | 0.5057 | 0.5341 | 0.5394 | 0.5814 |
| shape-kurt | 0.5190 | 0.4686 | 0.5126 | 0.5453 | 0.5407 |
| shape-ks | 0.4919 | 0.5413 | 0.5166 | 0.5201 | 0.5387 |
| trend | 0.5231 | 0.5361 | 0.5109 | 0.5196 | 0.5221 |
| dist-w1 | 0.5249 | 0.5419 | 0.5394 | 0.5643 | 0.6162 |
| dist-ks | 0.5161 | 0.5446 | 0.5275 | 0.5440 | 0.5832 |


## D. Taxonomy

| class (threshold rule, -log10p >= 2.0) | break | no-break | break_% | placebo_% | excess_% |
|---|---|---|---|---|---|
| dep_dominant | 32.00 | 22.00 | 0.81 | 0.55 | 0.26 |
| loc_dominant | 6.00 | 7.00 | 0.15 | 0.17 | -0.02 |
| mixed | 163.00 | 70.00 | 4.10 | 1.74 | 2.36 |
| scale_dominant | 11.00 | 5.00 | 0.28 | 0.12 | 0.15 |
| shape_dominant | 35.00 | 12.00 | 0.88 | 0.30 | 0.58 |
| trend_dominant | 55.00 | 32.00 | 1.38 | 0.80 | 0.59 |
| weak_unclassified | 3673.00 | 3877.00 | 92.40 | 96.32 | -3.92 |


**Leading family for every series (no threshold) - the placebo column is the null composition:**

| leading family (argmax, no threshold) | break | no-break | excess_pp |
|---|---|---|---|
| dep | 21.23 | 20.00 | 1.23 |
| loc | 20.00 | 24.00 | -4.00 |
| scale | 23.75 | 20.42 | 3.33 |
| shape | 15.75 | 14.16 | 1.59 |
| trend | 19.27 | 21.42 | -2.15 |


**Excess-over-placebo composition of the DETECTABLE breaks (total detectable excess = 3.9% of break series):**

| class | share_of_detectable_% |
|---|---|
| dep_dominant | 6.6 |
| loc_dominant | 0.0 |
| mixed | 59.9 |
| scale_dominant | 3.9 |
| shape_dominant | 14.8 |
| trend_dominant | 14.9 |


**Validation - family surprise profile by class (break series, mean stage-2 -log10 p, cap 3.2):**

| class | S_loc | S_scale | S_dep | S_shape | S_trend | n |
|---|---|---|---|---|---|---|
| dep_dominant | 0.527 | 0.646 | 2.512 | 0.733 | 0.420 | 32.000 |
| loc_dominant | 2.275 | 0.528 | 0.773 | 0.974 | 0.744 | 6.000 |
| mixed | 0.918 | 2.035 | 1.045 | 2.233 | 0.868 | 163.000 |
| scale_dominant | 0.455 | 2.331 | 0.469 | 0.977 | 0.263 | 11.000 |
| shape_dominant | 0.719 | 0.780 | 0.598 | 2.789 | 0.378 | 35.000 |
| trend_dominant | 1.304 | 1.499 | 0.804 | 1.674 | 2.486 | 55.000 |
| weak_unclassified | 0.419 | 0.470 | 0.458 | 0.464 | 0.446 | 3673.000 |


**Validation - median |raw effect size| by class (break series):**

| class | e_mean_z | e_log_sd_r | e_log_mad_r | e_acf1 | e_aacf1 | e_kurt | e_skew | e_dOHs_ks | e_slope_sd | e_halfdiff_sd | dOH_ks | dOH_w1 | dOH_js | n_post_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dep_dominant | 0.1018 | 0.2018 | 0.1519 | 0.2231 | 0.2432 | 0.6267 | 0.1810 | 0.0548 | 0.2503 | 0.1372 | 0.0955 | 0.1620 | 0.0271 | 140.5000 |
| loc_dominant | 0.8260 | 0.1640 | 0.1304 | 0.1314 | 0.0506 | 0.4388 | 0.1870 | 0.0492 | 0.1093 | 0.0969 | 0.1855 | 0.2770 | 0.0406 | 360.5000 |
| mixed | 0.1142 | 0.8094 | 0.7906 | 0.1237 | 0.1067 | 0.7864 | 0.2631 | 0.0470 | 0.2745 | 0.1575 | 0.1761 | 0.4397 | 0.0807 | 266.0000 |
| scale_dominant | 0.0988 | 0.7925 | 0.7252 | 0.0701 | 0.0927 | 2.0678 | 0.6470 | 0.0472 | 0.0566 | 0.0427 | 0.0967 | 0.3208 | 0.0359 | 60.0000 |
| shape_dominant | 0.0642 | 0.1552 | 0.1941 | 0.0462 | 0.0854 | 0.3334 | 0.2412 | 0.0817 | 0.1273 | 0.0560 | 0.1190 | 0.2014 | 0.0380 | 420.0000 |
| trend_dominant | 0.2837 | 0.4066 | 0.2528 | 0.0975 | 0.0804 | 0.7356 | 0.2524 | 0.0647 | 1.1739 | 0.6494 | 0.1886 | 0.5546 | 0.0583 | 220.0000 |
| weak_unclassified | 0.0656 | 0.1067 | 0.1146 | 0.0736 | 0.0778 | 0.4920 | 0.2093 | 0.0513 | 0.1636 | 0.0927 | 0.0752 | 0.1327 | 0.0220 | 197.0000 |


**Same for the PLACEBO (no-break) series assigned to each class - this is what the class looks like when there is no break:**

| class (placebo/no-break) | e_mean_z | e_log_sd_r | e_acf1 | e_kurt | e_slope_sd | dOH_w1 |
|---|---|---|---|---|---|---|
| dep_dominant | 0.0889 | 0.1968 | 0.1618 | 0.7892 | 0.1423 | 0.1004 |
| loc_dominant | 0.6652 | 0.1256 | 0.0641 | 0.1870 | 0.4733 | 0.6066 |
| mixed | 0.2700 | 0.5407 | 0.0963 | 0.9294 | 0.3111 | 0.3184 |
| scale_dominant | 0.2527 | 1.0233 | 0.1942 | 19.9918 | 0.4475 | 0.3965 |
| shape_dominant | 0.0576 | 0.1777 | 0.0573 | 3.1005 | 0.0571 | 0.1798 |
| trend_dominant | 0.2042 | 0.1415 | 0.0691 | 0.3858 | 1.4021 | 0.3393 |
| weak_unclassified | 0.0674 | 0.0843 | 0.0651 | 0.4803 | 0.1514 | 0.1225 |


## E. Detectability curve


**AUC, ORACLE WINDOW: statistic on online[tau : tau+h] only (needs to know tau; upper bound):**

| horizon h | tmean | atmean | trmean | tvar | tmad | tks | ttail | tacf1 | tabsacf1 | tslope | BEST | n_series |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.4971 | 0.5129 | 0.5129 | 0.4973 | 0.5020 | 0.5147 | 0.5088 |  |  |  | 0.5147 | 7847.0000 |
| 10 | 0.4961 | 0.5094 | 0.5183 | 0.5016 | 0.5101 | 0.5210 | 0.5182 |  |  |  | 0.5210 | 7670.0000 |
| 20 | 0.5007 | 0.5159 | 0.5094 | 0.5149 | 0.5120 | 0.5153 | 0.5211 | 0.4903 | 0.4986 | 0.5085 | 0.5211 | 7329.0000 |
| 40 | 0.4941 | 0.5224 | 0.5172 | 0.5296 | 0.5178 | 0.5365 | 0.5364 | 0.4921 | 0.4973 | 0.5241 | 0.5365 | 6766.0000 |
| 80 | 0.4954 | 0.5221 | 0.5242 | 0.5314 | 0.5214 | 0.5435 | 0.5379 | 0.4923 | 0.5001 | 0.5237 | 0.5435 | 5833.0000 |
| 160 | 0.4981 | 0.5212 | 0.5304 | 0.5598 | 0.5548 | 0.5574 | 0.5601 | 0.4955 | 0.4914 | 0.5072 | 0.5601 | 4477.0000 |
| 320 | 0.4714 | 0.5069 | 0.5308 | 0.5806 | 0.5629 | 0.5748 | 0.5934 | 0.4937 | 0.5096 | 0.5430 | 0.5934 | 2638.0000 |
| 640 | 0.5061 | 0.5324 | 0.5302 | 0.6231 | 0.5871 | 0.6003 | 0.6187 | 0.4436 | 0.4691 | 0.5562 | 0.6231 | 627.0000 |


**AUC, REALISTIC PREFIX: statistic on online[0 : tau+h], i.e. what an online detector actually sees at elapsed h:**

| horizon h | tmean | atmean | trmean | tvar | tmad | tks | ttail | tacf1 | tabsacf1 | tslope | BEST | n_series |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.5031 | 0.5148 | 0.5119 | 0.5041 | 0.5092 | 0.5120 | 0.5059 | 0.5050 | 0.5092 | 0.5108 | 0.5148 | 7847.0000 |
| 10 | 0.5032 | 0.5180 | 0.5113 | 0.5047 | 0.5094 | 0.5118 | 0.5064 | 0.5031 | 0.5122 | 0.5152 | 0.5180 | 7670.0000 |
| 20 | 0.5033 | 0.5177 | 0.5105 | 0.5094 | 0.5148 | 0.5137 | 0.5141 | 0.5048 | 0.5098 | 0.5157 | 0.5177 | 7329.0000 |
| 40 | 0.4993 | 0.5226 | 0.5157 | 0.5111 | 0.5192 | 0.5210 | 0.5178 | 0.5028 | 0.5174 | 0.5236 | 0.5236 | 6766.0000 |
| 80 | 0.4964 | 0.5186 | 0.5157 | 0.5154 | 0.5107 | 0.5199 | 0.5205 | 0.4976 | 0.5212 | 0.5197 | 0.5212 | 5833.0000 |
| 160 | 0.4964 | 0.5184 | 0.5070 | 0.5312 | 0.5288 | 0.5302 | 0.5305 | 0.4925 | 0.5178 | 0.5124 | 0.5312 | 4477.0000 |
| 320 | 0.4795 | 0.5207 | 0.5147 | 0.5556 | 0.5413 | 0.5527 | 0.5649 | 0.5012 | 0.5227 | 0.5237 | 0.5649 | 2638.0000 |
| 640 | 0.5010 | 0.5394 | 0.5221 | 0.6037 | 0.5687 | 0.5843 | 0.6131 | 0.4496 | 0.4728 | 0.5469 | 0.6131 | 627.0000 |


**Oracle-window AUC inside a FIXED cohort (n_post >= 160, 2244 break / 2233 no-break):**

| horizon h | tmean | atmean | trmean | tvar | tmad | tks | ttail | tacf1 | tabsacf1 | tslope | BEST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.4875 | 0.5230 | 0.5173 | 0.4975 | 0.5011 | 0.5266 | 0.5147 |  |  |  | 0.5266 |
| 10 | 0.4877 | 0.5176 | 0.5286 | 0.5079 | 0.5103 | 0.5267 | 0.5244 |  |  |  | 0.5286 |
| 20 | 0.4941 | 0.5206 | 0.5096 | 0.5247 | 0.5181 | 0.5205 | 0.5322 | 0.4907 | 0.4944 | 0.5158 | 0.5322 |
| 40 | 0.4914 | 0.5213 | 0.5192 | 0.5338 | 0.5281 | 0.5389 | 0.5439 | 0.4936 | 0.4911 | 0.5301 | 0.5439 |
| 80 | 0.4942 | 0.5213 | 0.5268 | 0.5361 | 0.5290 | 0.5477 | 0.5418 | 0.4946 | 0.5016 | 0.5229 | 0.5477 |
| 160 | 0.4981 | 0.5212 | 0.5304 | 0.5598 | 0.5548 | 0.5574 | 0.5601 | 0.4955 | 0.4914 | 0.5072 | 0.5601 |


**Best-single-statistic oracle-window AUC by class and horizon, vs ALL no-break (oracle class assignment -> optimistic):**

| class | 5 | 10 | 20 | 40 | 80 | 160 | 320 | 640 | n_pos | n_neg |
|---|---|---|---|---|---|---|---|---|---|---|
| dep_dominant | 0.527 | 0.589 | 0.615 | 0.714 | 0.643 | 0.673 | 0.733 |  | 32.000 | 4025.000 |
| mixed | 0.671 | 0.739 | 0.754 | 0.790 | 0.813 | 0.841 | 0.884 | 0.922 | 163.000 | 4025.000 |
| shape_dominant | 0.626 | 0.688 | 0.676 | 0.650 | 0.738 | 0.773 | 0.788 | 0.972 | 35.000 | 4025.000 |
| trend_dominant | 0.745 | 0.797 | 0.776 | 0.823 | 0.790 | 0.771 | 0.820 |  | 55.000 | 4025.000 |
| weak_unclassified | 0.506 | 0.509 | 0.510 | 0.522 | 0.524 | 0.543 | 0.579 | 0.609 | 3673.000 | 4025.000 |


**Best-single-statistic oracle-window AUC by class and horizon, vs class-MATCHED placebo (honest):**

| class | 5 | 10 | 20 | 40 | 80 | 160 | 320 | 640 | n_pos | n_neg |
|---|---|---|---|---|---|---|---|---|---|---|
| dep_dominant | 0.544 | 0.613 | 0.668 | 0.577 | 0.598 | 0.776 | 0.787 |  | 32.000 | 22.000 |
| mixed | 0.583 | 0.597 | 0.610 | 0.634 | 0.609 | 0.690 | 0.674 | 0.772 | 163.000 | 70.000 |
| trend_dominant | 0.534 | 0.519 | 0.565 | 0.585 | 0.506 | 0.521 | 0.579 |  | 55.000 | 32.000 |
| weak_unclassified | 0.509 | 0.514 | 0.513 | 0.527 | 0.533 | 0.550 | 0.585 | 0.620 | 3673.000 | 3877.000 |


**Where the metric is decided: share of all post-break online rows within h steps of tau, next to the prefix AUC available there:**

| elapsed <= h | cum_share_of_positive_rows | prefix_BEST_AUC |
|---|---|---|
| 5 | 0.0190 | 0.5148 |
| 10 | 0.0376 | 0.5180 |
| 20 | 0.0735 | 0.5177 |
| 40 | 0.1411 | 0.5236 |
| 80 | 0.2623 | 0.5212 |
| 160 | 0.4603 | 0.5312 |
| 320 | 0.7326 | 0.5649 |
| 640 | 0.9625 | 0.6131 |


## F. No-break series: transients that look like breaks

| transient (rate per series) | no-break online | matched-length hist window (no-break) | pre-tau region (break series) | whole online (break series) |
|---|---|---|---|---|
| isolated outlier |z_rob|>5 with quiet neighbours | 0.1632 | 0.1429 | 0.1245 | 0.2099 |
| transient vol burst | 0.0390 | 0.0407 | 0.0220 | 0.0497 |
| mean-reverting excursion | 0.0092 | 0.0099 | 0.0043 | 0.0121 |
| any |z_rob|>5 point | 0.1774 | 0.1632 | 0.1205 | 0.2216 |
| ANY of the three | 0.1702 | 0.1563 | 0.1132 | 0.2156 |


**Naive online detector peaks (max over t of the expanding-prefix statistic):**

| quantile | tr_on_det_mean | tr_on_det_rmean | tr_on_det_vol | tr_on_det_cusum |
|---|---|---|---|---|
| no-break q50 | 1.771 | 1.564 | 2.984 | 2.057 |
| no-break q90 | 3.619 | 3.254 | 8.154 | 4.028 |
| no-break q95 | 4.597 | 4.208 | 11.099 | 5.054 |
| no-break q99 | 8.514 | 8.168 | 21.707 | 8.236 |
| break q50 | 1.864 | 1.663 | 3.241 | 2.180 |
| break q90 | 3.879 | 3.563 | 9.123 | 4.372 |
| break q95 | 4.924 | 4.612 | 13.659 | 5.529 |
| break q99 | 9.020 | 8.375 | 30.899 | 9.978 |
| break pre-tau q50 | 1.638 | 1.442 | 2.582 | 1.927 |
| break pre-tau q90 | 3.477 | 3.070 | 6.900 | 3.801 |
| break pre-tau q95 | 4.415 | 3.936 | 9.139 | 4.725 |
| break pre-tau q99 | 7.442 | 6.322 | 18.842 | 7.376 |


**Naive series-level detectors: operating points:**

| naive detector | thr_at_50pct_break_recall | nobreak_fire_rate | series_auc | break_recall_at_5pct_FPR |
|---|---|---|---|---|
| tr_on_det_mean | 1.8640 | 0.4532 | 0.5261 | 0.0606 |
| tr_on_det_rmean | 1.6635 | 0.4502 | 0.5288 | 0.0619 |
| tr_on_det_vol | 3.2408 | 0.4427 | 0.5340 | 0.0684 |
| tr_on_det_cusum | 2.1796 | 0.4378 | 0.5339 | 0.0619 |


## G. Interaction with the historical DGP


**Mean leading-family surprise by historical-property tertile (break vs placebo - the difference is the real effect):**

| hist property | break low | break mid | break high | placebo low | placebo mid | placebo high | break-placebo (high) | break-placebo (low) |
|---|---|---|---|---|---|---|---|---|
| H_kurt | 0.966 | 0.909 | 1.052 | 0.809 | 0.799 | 0.919 | 0.134 | 0.157 |
| H_acf1 | 0.873 | 0.960 | 1.094 | 0.737 | 0.811 | 0.981 | 0.113 | 0.136 |
| H_aacf1 | 0.857 | 0.901 | 1.167 | 0.720 | 0.792 | 1.017 | 0.151 | 0.136 |
| H_volvol | 0.840 | 0.901 | 1.179 | 0.689 | 0.832 | 1.013 | 0.167 | 0.152 |
| H_pe3 | 1.054 | 0.922 | 0.955 | 0.914 | 0.824 | 0.786 | 0.169 | 0.140 |
| H_spec_ent | 1.088 | 0.911 | 0.929 | 0.985 | 0.760 | 0.784 | 0.145 | 0.103 |
| H_qspacing | 0.975 | 0.893 | 1.060 | 0.827 | 0.798 | 0.901 | 0.159 | 0.149 |
| H_bp_lo | 0.864 | 0.964 | 1.099 | 0.726 | 0.805 | 0.996 | 0.103 | 0.138 |
| H_hj_mob | 1.094 | 0.958 | 0.875 | 0.981 | 0.812 | 0.736 | 0.139 | 0.113 |


**Leading-family composition EXCESS over placebo (pp) by H_kurt tertile:**

| H_kurt tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | 1.6 | -4.4 | 1.5 | 2.8 | -1.4 |
| mid | 2.2 | -2.8 | 3.9 | 1.0 | -4.2 |
| high | 0.2 | -4.4 | 4.1 | 0.5 | -0.4 |


**Leading-family composition EXCESS over placebo (pp) by H_acf1 tertile:**

| H_acf1 tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | 0.9 | -2.2 | 0.4 | 1.8 | -0.9 |
| mid | 1.4 | -5.8 | 6.1 | 1.3 | -3.0 |
| high | 1.8 | -4.4 | 4.0 | 1.7 | -3.1 |


**Leading-family composition EXCESS over placebo (pp) by H_aacf1 tertile:**

| H_aacf1 tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | -1.2 | -3.0 | 5.2 | 3.1 | -4.2 |
| mid | 2.3 | -4.5 | 2.5 | 0.6 | -0.8 |
| high | 2.5 | -4.5 | 2.4 | 1.2 | -1.6 |


**Leading-family composition EXCESS over placebo (pp) by H_volvol tertile:**

| H_volvol tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | -2.2 | -3.0 | 3.3 | 4.4 | -2.6 |
| mid | 2.3 | -3.7 | 4.7 | 0.0 | -3.3 |
| high | 3.7 | -4.8 | 1.2 | -0.2 | 0.1 |


**Break rate by historical-property tertile (does hist alone leak the label?):**

| hist property | low | mid | high |
|---|---|---|---|
| H_kurt | 0.4931 | 0.4872 | 0.5103 |
| H_acf1 | 0.4874 | 0.4955 | 0.5077 |
| H_aacf1 | 0.4848 | 0.5000 | 0.5058 |
| H_volvol | 0.4811 | 0.4981 | 0.5114 |
| H_pe3 | 0.4976 | 0.5008 | 0.4923 |
| H_spec_ent | 0.5096 | 0.4936 | 0.4874 |
| H_qspacing | 0.4848 | 0.4940 | 0.5118 |
| H_bp_lo | 0.4916 | 0.4929 | 0.5062 |
| H_hj_mob | 0.5077 | 0.4959 | 0.4871 |
| H_n | 0.5028 | 0.4942 | 0.4936 |


**Series-level AUC of purely historical statistics against has_break (0.50 = no leakage):**

|  | AUC(hist stat -> has_break) |
|---|---|
| H_kurt | 0.5079 |
| H_acf1 | 0.5083 |
| H_aacf1 | 0.5094 |
| H_volvol | 0.5151 |
| H_pe3 | 0.4974 |
| H_spec_ent | 0.4903 |
| H_qspacing | 0.5091 |
| H_bp_lo | 0.5057 |
| H_hj_mob | 0.4917 |


## H. tau, lengths, generator artifacts

- rel_tau = tau/n_online on break series: mean 0.4872, sd 0.2903, KS vs U(0,1) D=0.0263 p=0.00813

- rel_tau decile shares (%): {'0.0-0.1': 10.49, '0.1-0.2': 10.64, '0.2-0.3': 10.97, '0.3-0.4': 10.11, '0.4-0.5': 9.61, '0.5-0.6': 9.84, '0.6-0.7': 9.43, '0.7-0.8': 9.31, '0.8-0.9': 10.16, '0.9-1.0': 9.43}

- Spearman rho: (tau_index, n_online) 0.6432; (rel_tau, n_online) 0.0025; (rel_tau, n_hist) 0.0072; (n_hist, n_online) all-series 0.0019

- break rate by n_online tertile: {'low': 0.491, 'mid': 0.4959, 'high': 0.5038}

- break rate by n_hist tertile: {'low': 0.5028, 'mid': 0.4942, 'high': 0.4936}


**Spearman correlation of effect size with tau and lengths:**

| |effect size| vs | break: tau_index | break: rel_tau | break: n_online | break: n_hist | break: n_post | break: n_pre | placebo: n_post | placebo: rel_tau |
|---|---|---|---|---|---|---|---|---|
| eh_mean_z | -0.0245 | 0.2307 | -0.2519 | -0.0391 | -0.3672 | -0.0245 | -0.3900 | 0.2252 |
| eh_log_sd_r | -0.0058 | 0.1387 | -0.1254 | 0.0087 | -0.2038 | -0.0058 | -0.3234 | 0.1816 |
| eh_acf1 | 0.0404 | 0.1754 | -0.1386 | -0.0289 | -0.2561 | 0.0404 | -0.3420 | 0.2097 |
| eh_kurt | 0.0043 | 0.0911 | -0.0959 | 0.0270 | -0.1578 | 0.0043 | -0.2359 | 0.1434 |
| e_slope_sd | 0.0171 | 0.2186 | -0.2347 | -0.0072 | -0.3594 | 0.0171 | -0.3638 | 0.2181 |
| dOH_w1 | -0.0062 | 0.3298 | -0.3312 | -0.0301 | -0.5227 | -0.0062 | -0.6246 | 0.3668 |
| lead_score | -0.0133 | -0.0394 | 0.0069 | -0.0127 | 0.0611 | -0.0133 | -0.0377 | -0.0255 |


**Leading-family EXCESS over placebo (pp) by n_hist tertile:**

| n_hist tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | -1.7 | -4.1 | 2.4 | 4.8 | -1.4 |
| mid | 2.7 | -3.3 | 6.0 | -2.0 | -3.3 |
| high | 2.7 | -4.6 | 1.6 | 2.0 | -1.7 |


**Leading-family EXCESS over placebo (pp) by n_online tertile:**

| n_online tertile | dep | loc | scale | shape | trend |
|---|---|---|---|---|---|
| low | 0.2 | -1.0 | -0.6 | 0.4 | 1.0 |
| mid | 2.4 | -6.0 | 5.5 | 2.5 | -4.4 |
| high | 1.1 | -4.8 | 5.0 | 1.9 | -3.3 |

## I. AR-residual probe: is the break in the innovations?

| residual statistic | AUC | n |
|---|---|---|
| r2_logsd | 0.5780 | 7582.0000 |
| r2_mean | 0.4931 | 7582.0000 |
| r2_logmad | 0.5598 | 7582.0000 |
| r2_logsd_pre | 0.5958 | 5981.0000 |
| r2_acf1 | 0.5344 | 7044.0000 |
| r2_absacf1 | 0.5050 | 7044.0000 |
| r2_kurt | 0.5163 | 7044.0000 |
| r2_ks | 0.5409 | 7044.0000 |
| r2_w1 | 0.5739 | 7044.0000 |
| r6_logsd | 0.5854 | 7582.0000 |
| r6_mean | 0.4925 | 7582.0000 |
| r6_logmad | 0.5692 | 7582.0000 |
| r6_logsd_pre | 0.6008 | 5981.0000 |
| r6_acf1 | 0.5402 | 7044.0000 |
| r6_absacf1 | 0.5090 | 7044.0000 |
| r6_kurt | 0.5172 | 7044.0000 |
| r6_ks | 0.5458 | 7044.0000 |
| r6_w1 | 0.5829 | 7044.0000 |
| diff_logsd | 0.5612 | 7582.0000 |
| (raw) eh_log_sd_r | 0.5486 | 7847.0000 |
| (raw) eh_log_mad_r | 0.5389 | 7847.0000 |
| (raw) e_log_sd_r | 0.5594 | 7847.0000 |
| (raw) dOH_w1 | 0.5469 | 7735.0000 |
| (raw) eh_acf1 | 0.5379 | 7201.0000 |
| [matched n=5648] e_log_sd_r | 0.5754 | 5648.0000 |
| [matched n=5648] eh_log_sd_r | 0.5610 | 5648.0000 |
| [matched n=5648] r2_logsd | 0.5875 | 5648.0000 |
| [matched n=5648] r6_logsd | 0.5964 | 5648.0000 |
| [matched n=5648] r2_logsd_pre | 0.5970 | 5648.0000 |
| [matched n=5648] r6_logsd_pre | 0.6029 | 5648.0000 |
| [matched n=5648] r6_w1 | 0.5896 | 5648.0000 |
| [matched n=5648] dOH_w1 | 0.5611 | 5648.0000 |
| [matched n=5648] diff_logsd | 0.5666 | 5648.0000 |
| [matched n=5648] r6_acf1 | 0.5443 | 5648.0000 |

---

## J. Reading of the tables

### J.1 Which break families dominate

On the whole post-cut segment (section C), the ranking of series-level AUC is:

| family | best raw statistic | AUC | after AR(6) whitening |
|---|---|---|---|
| SCALE | `log(sd_post/sd_ref)` | 0.559 (vs PRE) | **0.603** |
| DISTANCE (mixed loc+scale+shape) | Wasserstein-1 post-vs-pre | 0.558 | 0.590 |
| DEPENDENCE | `acf1` shift | 0.538 | 0.544 |
| SHAPE | kurtosis shift / shape-only KS | 0.522 / 0.508 | 0.517 |
| TREND | within-post slope, half-difference | 0.518 / 0.524 | — |
| **LOCATION** | mean shift | **0.4998** | 0.493 |

Location is *exactly* dead — including the robust and trimmed variants and the residual
mean (0.493). This is the single most important negative result in this report: it kills
CUSUM-of-mean, EWMA-of-mean and every level-shift detector as a *primary* signal.
(They may still contribute as disambiguators — see §J.5.)

The dependence family is real but modest, and it is partly *aliased into* the scale family:
an AR-coefficient change moves the unconditional variance, which is why AR whitening
(which removes the dependence channel from the scale statistic) still *improves* scale AUC —
the whitened statistic is measuring innovation variance, which is the cleaner target.

Shape (kurtosis, tail occupancy, entropy, quantile spacing) contributes on top of scale but
mostly at long horizons (see the `ttail` column: AUC 0.593 at h = 320 and 0.619 at h = 640,
comparable to `tvar` and *better* than it in the prefix table).

### J.2 Where the metric is won or lost

Combining section E with the row-share column:

| elapsed since tau | share of positive rows | prefix AUC available |
|---|---|---|
| 1–20 | 7.4 % | 0.515 – 0.518 |
| 21–80 | 18.9 % | 0.521 – 0.524 |
| 81–320 | 47.0 % | 0.521 – 0.565 |
| 321+ | 26.7 % | 0.565 – 0.613 |

**~74 % of the positive rows sit at elapsed ≤ 320 where the best single statistic gives
0.52–0.56.** The high-AUC regime (elapsed > 320, AUC 0.61) covers only ~27 % of positive
rows and is available only in series with long online segments. So TS-AUC is won in the
mid-band (elapsed 40–320) — that is where feature work pays. Very-early detection
(elapsed < 20) is essentially impossible with univariate statistics and should not be
optimised for; it is 7 % of the positive rows at AUC 0.515.

The gap between the ORACLE-WINDOW table (statistic computed on `online[tau:tau+h]` only,
which requires knowing tau) and the REALISTIC-PREFIX table (statistic on `online[0:tau+h]`)
is the **dilution cost of not knowing tau**: at h = 160 it is 0.560 vs 0.531, at h = 320
0.593 vs 0.565. **Roughly 3 AUC points are recoverable by any mechanism that localises the
change point** (multi-scale trailing windows, change-point-weighted statistics, CUSUM-style
max-over-split rather than expanding means). That is the largest single structural
opportunity found in this analysis.

### J.3 Taxonomy — frequency, effect size, detectable horizon

The interpretable rule (section D): each family's effect sizes are converted to a
`-log10 p` surprise against the **no-break placebo null within the same post-length
bucket** (buckets `<50 / 50-99 / 100-199 / 200-399 / >=400`); the family score is the max
over its members, itself re-calibrated against the placebo null so the five families have
identical null distributions; class = leader if leader `>= 2.0` (p < 0.01), `mixed` if the
runner-up is `>= 0.6 x` the leader, `trend_dominant` given precedence when trend is within
20 % of the leader (a ramp mechanically produces a level shift), else `weak_unclassified`.

The class profiles in section D validate the rule: `dep_dominant` shows S_dep 2.51 with all
other families < 0.8; `scale_dominant` S_scale 2.33 with others < 1.0; `shape_dominant`
S_shape 2.79 with others < 1.0; `trend_dominant` S_trend 2.49 with median within-post slope
1.17 sd (vs 0.16 for the unclassified). Median raw effect sizes are coherent with the labels
(`scale_dominant` median |log sd ratio| 0.79 vs 0.11 for unclassified; `dep_dominant` median
|Δacf1| 0.22 vs 0.07).

| class | % of break series | % of placebo | excess pp | share of *detectable* breaks | honest AUC (class-matched, h=160) |
|---|---|---|---|---|---|
| mixed (scale+shape) | 4.10 | 1.74 | +2.36 | 60 % | 0.69 |
| trend_dominant | 1.38 | 0.80 | +0.59 | 15 % | 0.52 |
| shape_dominant | 0.88 | 0.30 | +0.58 | 15 % | n/a (n too small) |
| dep_dominant | 0.81 | 0.55 | +0.26 | 7 % | 0.78 |
| scale_dominant | 0.28 | 0.12 | +0.15 | 4 % | n/a |
| loc_dominant | 0.15 | 0.17 | −0.02 | 0 % | n/a |
| weak_unclassified | 92.40 | 96.32 | −3.92 | — | 0.55 |

Two things must be read carefully here:

* The **"vs ALL no-break" per-class AUC table is optimistic and should not be quoted** —
  the class label is assigned using post-tau data, so comparing a class against the *whole*
  no-break population is selection on the outcome. The class-MATCHED table (break series in
  class *c* vs placebo series the same rule put in class *c*) is the honest one. Under it,
  `trend_dominant` collapses from 0.77–0.82 to **0.51–0.58**: the trend class is mostly
  picking up *non-stationary series*, which exist equally in the no-break population, not
  post-break drift. Do not build a trend detector expecting the optimistic numbers.
* `weak_unclassified` — 92 % of break series — still scores **0.55 (h=160) / 0.62 (h=640)**
  against class-matched placebos. So the mass of breaks is real but sub-threshold at the
  series level. **A per-series classifier is the wrong frame; the value is in aggregating a
  weak, well-calibrated signal over many rows**, which is exactly what the TS-AUC metric
  rewards.

Detectable horizon by class (oracle window, class-matched): `dep_dominant` needs ~160 points
(0.60 at h=80, 0.78 at h=160); `mixed` is detectable from h=20 (0.61) and improves steadily
to 0.77 at h=640; `trend_dominant` never separates from its placebo; `weak_unclassified`
climbs slowly from 0.51 (h=5) to 0.62 (h=640).

### J.4 No-break series and the false-positive budget

The numbers our false-positive research has to beat (section F):

* **17.0 %** of no-break series contain at least one of the three break-lookalike transients
  in the online segment: 16.3 % an isolated `|z_rob| > 5` outlier with quiet neighbours,
  3.9 % a transient volatility burst (rolling-20 sd > 2x hist sd for 10–80 points then back
  under 1.25x), 0.9 % a mean-reverting excursion (rolling-30 mean |z| > 3.5 for ≥ 15 points
  then back inside 1.5).
* **The correct baseline is 15.6 %** — the same detectors run on a *matched-length window of
  the break-free history* fire almost as often. The online-segment excess is only +1.4 pp.
  **These transients are the DGP, not the break.** Any feature that fires on an isolated
  outlier or a short vol burst is buying a 15–17 % base-rate false-positive stream, and the
  disambiguator has to be *persistence*, not amplitude.
* Naive expanding-prefix detectors are close to useless at the series level: max prefix
  mean-z AUC 0.526, robust mean 0.529, prefix log-variance 0.534, max-split CUSUM 0.534.
  At a 5 % false-positive rate they recover 6.1–6.8 % of break series. The 95th percentile of
  the max-CUSUM statistic on *no-break* series is 5.05 — i.e. a break-free series routinely
  produces a 5-sigma-looking change point somewhere in its online segment. Any hard threshold
  must be calibrated per-series against its own historical null and against the number of
  looks taken (multiple-testing over t), which is precisely what `sbr.nullcal` provides.
* Break series' own pre-tau regions are *quieter* than no-break online segments
  (11.3 % vs 17.0 % transient rate) — but that is a length artifact (pre-tau regions are on
  average half as long), not evidence that pre-tau data is unusually calm.

### J.5 Signal / false-signal / disambiguator, per family

* **SCALE (the main family).** *Signal:* innovation variance changes at tau. *False signal:*
  (a) the systematic negative bias of the online/hist variance ratio under heavy tails
  (median −0.063 log-units in the top hist-kurtosis quartile — a fake *decrease*);
  (b) a transient vol burst (3.9 % of no-break series); (c) an AR-coefficient change
  aliasing into unconditional variance. *Disambiguator:* whiten with an AR filter fitted on
  history before measuring scale (that removes (c) and improves AUC by 3.5 points), compare
  against the PRE segment rather than history when `n_pre >= 60` (that cancels most of (a)),
  and require persistence over multiple window lengths (that removes (b)).
* **DEPENDENCE.** *Signal:* AR/ARCH structure changes. *False signal:* short windows give
  |acf1| estimates of order `1/sqrt(n)`, so any acf-shift feature is dominated by sample-size
  noise when `n_post < 100` (AUC 0.51 at `n_post < 50` versus 0.58 at `n_post >= 400`).
  *Disambiguator:* the null-calibration grid — an acf shift must be scored against the
  historical null *at the same window length*.
* **SHAPE/TAIL.** *Signal:* tail thickness and centre occupancy change. *False signal:* a
  single outlier moves kurtosis by an arbitrary amount; historical kurtosis in this data
  ranges to 4,782, so kurtosis differences are un-normalisable across the cross-section.
  *Disambiguator:* use bounded shape statistics (tail-occupancy fractions, PIT entropy,
  quantile spacing) rather than moments — `ttail` beats `tvar` in the realistic-prefix table
  at h = 320 and 640 (0.565/0.613 vs 0.556/0.604) and is far more robust.
* **TREND.** *Signal:* post-break drift. *False signal:* the history itself is
  near-non-stationary in a slice of the population (`H_acf1` up to 0.998), and those series
  drift in the online segment with or without a break. *Disambiguator:* the class-matched
  comparison shows trend features carry essentially nothing once you condition on the series
  being trendy; a trend feature must be normalised by the *historical* trend variability of
  that same series or it is a pure DGP indicator.
* **LOCATION.** *Signal:* none detectable. *False signal:* everything. Do not spend a
  feature budget here except as a disambiguator for the scale family (a mean jump plus a
  variance jump is a different animal from a variance jump alone).

### J.6 Interaction with the historical DGP

* Break-vs-placebo effect size is **larger in series whose history is more volatile-clustered**:
  mean leading-family surprise excess over placebo is +0.167 in the top `H_volvol` tertile and
  +0.152 in the bottom, +0.151 top vs +0.136 bottom for `H_aacf1`. The effect is small but
  consistently positive across all nine historical properties (+0.10 to +0.17), i.e. breaks
  are slightly easier to see in messier series, with no property showing a reversal.
* **Break-type composition does shift with the DGP** (leading-family excess over placebo, pp):
  scale-dominance excess is +4.1 in the top hist-kurtosis tertile vs +1.5 in the bottom, and
  +6.1 / +4.0 in the mid/high `H_acf1` tertiles vs +0.4 in the low tertile. Dependence-
  dominance excess rises with `H_volvol` (−2.2 → +2.3 → +3.7 across tertiles). Shape
  dominance runs the other way (+4.4 in the low-`H_volvol` tertile, −0.2 in the high).
  **Practical consequence: stratify any evaluation by historical tail/dependence tertile —
  a feature that helps on the quiet third of the population can be flat on the messy third.**
  The `H_*` columns in the artifact are there for exactly this.
* **No label leakage from history alone.** Break rate by historical-property tertile stays
  inside 0.481–0.515, and the series-level AUC of each historical statistic against
  `has_break` is 0.490–0.515. The largest is `H_volvol` at 0.5151 (2.3 sigma, not significant
  after testing nine properties). History-only features are a legitimate *conditioning*
  signal (they tell you how noisy the null is) but not a *prediction* signal.

### J.7 tau, lengths and generator artifacts

* `rel_tau = tau/n_online` has mean 0.4872, sd 0.2903 — close to uniform but **rejected**:
  KS D = 0.0263, p = 0.0081. Deciles: 10.49, 10.64, 10.97, 10.11, 9.61, 9.84, 9.43, 9.31,
  10.16, 9.43 %. There is a mild tilt toward early breaks (first 30 %: 32.1 % of mass;
  last 30 %: 28.9 %). **This is a generator artifact worth exploiting only as a weak prior**
  — a per-row prior on `t/n_online` is legitimate and causal only if `n_online` is known at
  inference time; in the real-time protocol it is *not*, so treat this as a diagnostic only.
* `corr(tau_index, n_online) = 0.643` is mechanical (tau = rel_tau x n_online).
  `corr(rel_tau, n_online) = 0.003` and `corr(rel_tau, n_hist) = 0.007` — **tau position is
  independent of both lengths**, and `corr(n_hist, n_online) = 0.002` — the two lengths are
  independently drawn. No generator coupling there.
* Break rate is flat in both lengths (n_online tertiles 0.491 / 0.496 / 0.504; n_hist
  tertiles 0.503 / 0.494 / 0.494). **No length-based label leakage.**
* **The apparent effect-size/tau relationship is fake.** |effect| correlates with `rel_tau`
  at +0.14 to +0.33 and with `n_post` at −0.16 to −0.52 in break series — but the *placebo*
  series show the same or stronger correlations (+0.14 to +0.37 with `rel_tau`, −0.24 to
  −0.62 with `n_post`). It is pure estimator noise scaling with `1/sqrt(n_post)`. After
  length-bucketed calibration the residual dependence collapses to +0.06 (break) / −0.04
  (placebo). **Any feature that is not calibrated at matched window length will encode
  `n_post` — i.e. the position of tau — instead of the break.** That is the most dangerous
  trap in this dataset.
* Break-type composition by length shows only weak structure (scale excess +6.0 pp in the
  middle `n_hist` tertile vs +2.4 / +1.6 in low / high; shape excess +4.8 pp in the low
  `n_hist` tertile). Given the ~1.5 pp sampling noise on these cells, only the `n_hist`-middle
  scale bump is suggestive, and it does not replicate in `n_online`. **No strong break-type /
  length coupling.**

---

## K. Recommendations to the org (ranked)

1. **Whiten before measuring scale, and use AR order > 2.** `HistParams` fixes `ar_order = 2`;
   AR(6) residual log-sd is worth +0.9 AUC points over AR(2) and +3.5 over raw. Someone
   should test AR(6)/AR(8) residual transforms as a module (this is a suggestion about a core
   file, not an edit — per PROTOCOL §1 I have not touched `src/sbr/transforms.py`).
2. **Add a PRE-referenced scale channel.** When `n_pre >= 60` (77 % of break series), the
   post-vs-pre variance ratio beats post-vs-hist (0.575 vs 0.561 raw; 0.603 vs 0.596
   whitened) because it cancels the heavy-tail length bias. NaN when `n_pre < 60` is the
   correct value and LightGBM handles it.
3. **Spend the feature budget on change-point localisation, not on longer windows.**
   The oracle-window vs realistic-prefix gap is ~3 AUC points at elapsed 160–320, which is
   where the bulk of the positive rows live.
4. **Prefer bounded tail statistics over moments.** Tail occupancy beats variance in the
   realistic prefix at long horizons and cannot be blown up by one outlier — critical given
   historical kurtosis reaching 4,782.
5. **Do not build a mean-shift feature family.** AUC 0.4998.
6. **Stratify every evaluation by `break_class` / `lead_family` and by hist-property tertile**
   using the artifact — mean TS-AUC hides that ~92 % of breaks are sub-threshold and that
   families concentrate differently across the DGP.

## L. Caveats

* The taxonomy threshold (p < 0.01 per family against the placebo null) is deliberately
  conservative; at that threshold only 3.9 pp of break series exceed the null rate. The
  `lead_family` column (argmax, no threshold, assigned to every series) plus the placebo
  composition column is the more robust statement of "which family dominates" and is what
  §J.1 is based on.
* Long-horizon rows in the detectability table use a shrinking, non-random cohort
  (h = 640 keeps only 627 series, all with long online segments). The FIXED-COHORT table
  (`n_post >= 160`, 2,244 break / 2,233 no-break) is the composition-controlled version and
  shows the same shape (0.527 at h=5 → 0.560 at h=160).
* Series-level AUC is *not* TS-AUC. It is used here only as a comparable effect-size scale
  across families and horizons. Nothing in this report is a model-selection decision.
* The placebo cut for no-break series is one draw per series (seed 20260818); re-drawing it
  changes third-decimal numbers, not conclusions.

---

## M. PROTOCOL §6 return block

```
AGENT NAME:            agent02 (data forensics / break taxonomy)
HYPOTHESIS:            The 2026 break population is a mixture of distinguishable
                       break types whose effect sizes and detectable horizons differ,
                       and the no-break population contains break-lookalike transients
                       at a quantifiable rate.
FALSIFICATION:         (a) if no family separated break from placebo above AUC 0.52 the
                       taxonomy would be vacuous; (b) if the class rule gave the same
                       composition on break and placebo series the taxonomy would be noise.
                       Neither happened: scale reaches 0.603 and the class composition
                       differs from placebo by +3.3 pp (scale) / -4.0 pp (loc).
FILES CHANGED:         scripts/agent02_taxonomy.py (new)
                       research/artifacts/break_taxonomy.parquet (new)
                       research/reports/break_taxonomy.md (new)
                       nothing in src/sbr was read-modified; no core file edited.
EXPERIMENT IDs:        none — no pipeline.run() call, no model was trained, nothing
                       was appended to RESULTS.csv (this is analysis, not an experiment).
DATA USED:             cache/store, folds 0-4 only (8,000 series). Lockbox untouched.
                       X_test.reduced.parquet never read.
FOLDS USED:            0,1,2,3,4 (pooled; no fold-wise selection was performed)
MODEL+FEATURES:        none (univariate statistics only)
TS-AUC:                n/a — series-level AUC is reported as an effect-size scale only
CAUSALITY CHECK:       n/a — no feature module was registered. Every statistic in this
                       report deliberately uses tau and/or the full post segment and is
                       therefore NOT causal; nothing here may be used as a feature as-is.
                       The artifact's columns are for stratification and diagnosis only.
LEAKAGE RISKS:         the artifact contains tau-derived columns. Any downstream use must
                       treat break_class / lead_family / effect sizes as LABELS, never as
                       model inputs. Flagged in the parquet by the column prefixes
                       (e_, eh_, S_, O_, P_, w*_, p*_, r2_, r6_ are all tau-aware).
CONCLUSION:            KEEP (diagnostic artifact + six actionable findings)
```
