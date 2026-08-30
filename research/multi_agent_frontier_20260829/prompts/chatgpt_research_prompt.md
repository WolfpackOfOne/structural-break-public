You are AGENT_ID, one of five independent senior quantitative research agents investigating the 2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition.

You are working in:

WolfpackOfOne/structural-break

Branch:

research/multi-agent-frontier-20260829

Your job is NOT to modify the production system.

Your job is NOT to run another ordinary model search.

Your job is to independently determine what genuinely new causal predictive information, representation, inference mechanism, training target, or problem formulation could materially improve the current RT600 / RT1257 system.

The ultimate objective is to identify research directions with a credible path to materially improving TS-AUC, ideally by +0.003 to +0.010 or more, and especially mechanisms capable of repairing RT600 / RT1257 mistakes without destroying large numbers of pairs the incumbent already ranks correctly.

INDEPENDENCE RULE

This is an independence experiment.

You MUST NOT read any other agent's response before completing your own report.

Do not inspect:

research/multi_agent_frontier_20260829/responses/

except to write your own assigned output file.

Do not use another agent's conclusions, notes, commits, or response as input.

The purpose of this exercise is to see whether independent research agents converge on the same scientific conclusions or discover genuinely different ideas.

READ BOTH SOURCE PROMPTS FIRST

Read these files in full:

research/multi_agent_frontier_20260829/prompts/chatgpt_research_prompt.md

research/multi_agent_frontier_20260829/prompts/claude_code_research_prompt.md

Treat them as complementary research briefs.

Do not modify either file.

Do not simply answer them from memory or from this execution prompt.

REPOSITORY FORENSICS ARE REQUIRED

After reading both prompts, investigate the repository before forming your conclusions.

Do not assume the checked-out branch contains all relevant research.

Inspect relevant remote branches, reports, experiment histories, code, and measured results.

At minimum investigate:

engineering/rt1257-deployment-2026

research/catboost-specialist-2026

research/learner-diversity-2026

research/gpu-tabular-2026

research/causal-representation-frontier-2026

research/deep-ensemble-frontier-2026

research/wave7-teacher-distillation

research/wave8-future-aware-distillation

research/current

Also inspect where relevant:

research/RESULTS.csv

research/STATUS.md

research/FAILED_EXPERIMENTS.md

research/STATE_OF_RESEARCH.md

research/reports/

Git history

feature implementations

training implementations

OOF analyses

pair-flow analyses

dominant-cell analyses

RT600 reports and artifacts

RT1257 reports and artifacts

CatBoost results

TabM / RealMLP results

neural sequence experiments

causal representation experiments

teacher / oracle / future-state experiments

arbitration / stacking / routing experiments

Do not rely on stale summaries if newer measured evidence exists.

Verify important numbers from repository evidence.

COMPETITION CONSTRAINTS

All proposed production inference must be strictly causal.

At online timestep t, a deployed model may use:

1. the full historical break-free sample, and
2. online observations only through timestep t.

It may not use future online observations.

The final online horizon is unknown.

Past predictions cannot be revised.

The metric is time-stratified AUC, so the central problem is same-timestep cross-sectional ranking.

The model must distinguish broken from non-broken series at the same online age.

Training may use true break locations and future observations to construct privileged labels, teachers, auxiliary targets, oracle quantities, or self-supervised objectives, provided the deployed student remains causal.

Compute matters.

The final solution must be realistic under the Crunch runtime budget.

Prefer approaches with efficient streaming state updates.

A promising research idea may be expensive to develop, but you must propose a cheap falsification experiment before recommending a large compute commitment.

THE CURRENT SYSTEM

Do not assume that all previous models used the same representation.

A large fraction of the strongest system is built around approximately 500 engineered causal streaming features spanning multiple statistical families.

Many strong specialists use all or subsets of this shared representation.

However, the project has also tested genuinely different representations, including causal neural sequence models, null-normalized sequence representations, ranking objectives, generative-null approaches, teacher/distillation approaches, and other research programs.

Changing the learner alone has generally produced limited upside.

Highly decorrelated weak models have also demonstrated that prediction diversity alone is not useful.

The key requirement is NEW CORRECT INFORMATION.

A candidate is valuable if it:

repairs pairs that RT600 / RT1257 get wrong,

especially high-weight pairs,

while preserving the overwhelming majority of incumbent-correct pairs.

MODEL DIVERSITY IS NOT THE MAIN QUESTION

Do not structure your research around:

"What model have they not tried?"

Instead ask:

"What useful causal information have they not represented yet?"

"What latent state or statistic of the prefix is being discarded?"

"What training signal could reveal information that the direct break/no-break target does not?"

"What problem formulation would make the hidden signal easier to estimate?"

"What does a full-sequence oracle know that a causal model might be able to predict before that future arrives?"

"Why do alternative detectors sometimes repair incumbent errors but fail to know when they should be trusted?"

2025 WINNER CONTEXT

The 2025 winning Alphabot solution is relevant as a research-process example.

Their solution relied heavily on independently developed feature representations, multiple tree-based first-level models, and a second-level stack.

Different team members deliberately worked independently before combining ideas.

Their diversity came substantially from different information representations and feature-generation philosophies rather than simply using many unrelated model architectures.

However, the 2025 competition was offline and allowed retrospective analysis of the post-boundary sample.

Do not copy illegal offline features into the 2026 real-time setting.

Extract the methodological lesson instead:

independently discover different information views,

build specialists around those views,

then combine them with leakage-safe validation.

Our multi-agent exercise is intentionally trying to recreate the independent-research part of that process.

CRITICAL QUESTIONS

Your analysis must answer the following.

1. What is the real bottleneck in the current project?

Distinguish among:

learner capacity,

feature-bank saturation,

representation limitation,

training-target limitation,

objective mismatch,

arbitration/routing limitation,

historical-null estimation,

finite training data,

optimization,

and a genuine causal-information ceiling.

Do not accept the repository's current conclusion automatically.

Evaluate the evidence yourself.

2. What information is likely present in the legal causal prefix that our strongest current system does not represent well?

Be precise.

Do not say "more temporal information."

Describe the latent quantity or mathematical object.

Examples might include:

predictive distributions,

future forecast degradation,

latent regimes,

hazard,

posterior over change time,

break permanence,

break family,

state transition dynamics,

multi-horizon predictive error,

local likelihood geometry,

conditional density ratios,

spectral evolution,

state uncertainty,

or something completely different.

3. If you had to solve the competition from scratch and were forbidden from using the current 500-feature representation, what would you build?

Look beyond conventional tabular ML.

Consider relevant ideas from:

sequential statistics,

signal processing,

control theory,

Bayesian filtering,

information theory,

optimal stopping,

survival analysis,

forecasting,

predictive-state representations,

self-supervised learning,

representation learning,

econometrics,

financial econometrics,

fault detection,

communications,

computer vision,

speech,

language modeling,

neuroscience,

meta-learning,

scientific ML,

and other fields.

4. What assumptions in the existing research program may be wrong?

Identify at least five.

For each:

state the assumption,

explain why it could be wrong,

describe what information would be lost,

design a cheap experiment that tests it,

estimate the plausible upside if the assumption is wrong.

5. Where could genuinely complementary information come from?

Do not equate low correlation with complementarity.

Explain why your proposed mechanism should specifically generate:

RT600-wrong / candidate-correct pairs

without generating excessive:

RT600-correct / candidate-wrong pairs.

6. Is the remaining problem detection or arbitration?

Investigate whether the project already has useful alternative detectors but lacks a reliable mechanism for knowing when to trust them.

Consider:

mixture-of-experts routing,

contextual reliability estimation,

Bayesian model averaging,

expert-advice formulations,

uncertainty-conditioned routing,

break-family routing,

break-age routing,

posterior entropy,

historical-series taxonomy,

evidence geometry,

or other formulations.

Any routing mechanism must itself be cross-fitted and causal.

7. Is the competition label the wrong training target?

Investigate privileged or auxiliary targets such as:

break age,

posterior over tau,

future persistence,

future distribution divergence,

future forecast degradation,

break magnitude,

break family,

probability the apparent anomaly will remain abnormal 20/50/100 steps later,

full-sequence oracle likelihood ratios,

future cumulative evidence,

or other predictive-state targets.

If recommending distillation, specify:

teacher input,

teacher target,

student input,

student output,

loss,

cross-fitting scheme,

and why this transmits useful information rather than merely smoothing labels.

8. Why is mature-break / dominant-cell performance substantially stronger than overall performance?

Think in terms of pair mass and information geometry.

Determine whether the metric rewards improving mature-break discrimination more than early detection.

Do not assume that early detection is automatically the best place to spend research effort.

9. How would you test whether a causal-information ceiling is real?

Design experiments capable of separating:

information unavailable before t

from

information present in the prefix but not extracted by the current system.

An oracle-gap decomposition is especially valuable.

FORBIDDEN LOW-VALUE ANSWERS

Do not give generic recommendations such as:

try a Transformer,

try an LSTM,

try more CatBoost,

increase model size,

add more trees,

add more random seeds,

run hyperparameter optimization,

use Optuna,

add generic attention,

perform generic feature selection,

try another calibration method.

You may recommend one of these tools only if it implements a clearly articulated information hypothesis.

Every proposed architecture must answer:

"What new information does this make accessible?"

REQUIRED OUTPUT

Write a rigorous Markdown research report to:

research/multi_agent_frontier_20260829/responses/AGENT_OUTPUT_FILE.md

Your output filename must include a unique identifier for your agent/model identity, such as:

research/multi_agent_frontier_20260829/responses/agent_01_UNIQUE_ID.md

Do not write anywhere else.

Your report must contain the following sections.

1. Executive diagnosis

Maximum approximately 1,000 words.

Explain what the project currently does well and what you believe the true bottleneck is.

Explicitly classify the problem across:

learner,

representation,

objective,

target,

arbitration,

compute,

and information ceiling.

2. Repository evidence

List the most important measured results that informed your diagnosis.

For each, cite:

experiment ID where applicable,

branch,

report/file,

score,

and what you believe the result actually establishes.

Distinguish carefully between:

MEASURED FACT

INFERENCE

HYPOTHESIS

Do not turn an inference into a measured fact.

3. Five high-upside mechanisms

Give exactly five.

They must be scientifically distinct enough that failure of one would not automatically imply failure of the others.

For each mechanism provide:

Name.

Core hypothesis.

New information represented.

Why the current 500-feature system probably does not already contain it.

Why previous experiments do not already falsify it.

Causal inference state.

Training data construction.

Exact target.

Exact input representation.

Suggested estimator/model.

Streaming inference procedure.

How it would interact with RT600 / RT1257.

Expected dominant-cell effect.

Expected early-break effect.

Expected pair repair mechanism.

Expected pair damage risk.

Compute requirements.

Leakage risks.

Cheapest falsification experiment.

Kill criterion.

Promotion criterion.

Plausible TS-AUC upside.

Probability of success.

4. Rank the five

Rank them by:

expected value,

maximum upside,

probability of success,

complementarity with RT1257,

cost,

implementation complexity,

risk of leakage,

and scientific information gained if they fail.

5. One moonshot

Give exactly one.

It should have less than roughly a 30% probability of success but a credible path to greater than +0.010 TS-AUC if the hypothesis is correct.

Do not optimize for ease of integration.

6. Arbitration analysis

Explicitly answer:

Do we primarily need a better detector, or a better way to arbitrate among imperfect detectors?

Support the conclusion with repository evidence.

If arbitration is promising, design one concrete causal routing experiment.

7. Oracle / information-ceiling experiment

Design one decisive experiment intended to determine whether the remaining gap is fundamentally causal or merely representational.

Specify:

oracle,

causal student,

information available to each,

metric,

comparison,

and interpretation of each possible outcome.

8. What should we stop doing?

Identify research directions that have received enough evidence that continuing them is unlikely to produce +0.003 or larger improvement.

Be specific.

9. Next three experiments

End with exactly three experiments you would run next.

For each provide:

hypothesis,

implementation,

control,

required data,

required compute,

primary metric,

dominant-cell metric,

pair-flow metric,

success threshold,

kill threshold,

expected information gained even on failure.

These should answer major scientific questions.

They should not simply be three more leaderboard attempts.

10. Final recommendation

Answer this directly:

"If I controlled the next 15 hours of research compute and wanted the highest probability of eventually improving the external score by at least 100 basis points, what would I do?"

Give a concrete allocation of the research effort.

RESEARCH STANDARD

Be skeptical of both novelty and precedent.

A new idea is not good because it is new.

An existing idea is not dead merely because one weak implementation failed.

Determine what hypothesis each experiment actually tested.

Look for falsification, not confirmation.

Do not exaggerate expected gains.

Do not treat external leaderboard reads as hyperparameter feedback.

Do not violate the causal protocol.

Do not read another agent's response.

DO NOT MODIFY THE PROJECT

This task is analysis only.

Do not:

change model code,

change features,

run full training,

allocate experiment IDs,

modify RESULTS.csv,

modify STATUS.md,

modify existing reports,

change production artifacts,

merge branches,

or open a pull request.

You may run cheap inspection commands or lightweight calculations necessary to understand existing artifacts, but do not launch expensive research experiments.

WHEN FINISHED

1. Write your report to:

research/multi_agent_frontier_20260829/responses/AGENT_OUTPUT_FILE.md

Use a filename that includes your unique agent/model identifier, for example:

research/multi_agent_frontier_20260829/responses/agent_01_UNIQUE_ID.md

2. Review it for unsupported claims.

3. Confirm that every important claim is either repository-supported or clearly labeled as hypothesis/inference.

4. Confirm that you did not read another agent's response.

5. Commit ONLY your response file.

Use a commit message similar to:

Add AGENT_ID independent frontier analysis

6. Do not merge.

7. Do not synthesize other agents.

8. Report back with:

your agent/model identity,

output file,

commit SHA,

the titles of your five mechanisms,

your #1 recommended experiment,

and confirmation that no other agent response was read.
