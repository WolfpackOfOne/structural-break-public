# Multi-Agent Structural Break Research Frontier — 2026-08-29

## Purpose

This directory coordinates an independent multi-agent research exercise intended to identify genuinely new causal information representations or inference mechanisms for the 2026 ADIA Lab / CrunchDAO Structural Break Challenge — Real-Time Edition.

Independence is mandatory.

## Source prompts

Every agent must read both:

- prompts/chatgpt_research_prompt.md
- prompts/claude_code_research_prompt.md

## Repository investigation

Agents must not assume the checked-out branch contains every relevant experiment.

They should inspect relevant remote branches and historical research, including at minimum:

- engineering/rt1257-deployment-2026
- research/catboost-specialist-2026
- research/learner-diversity-2026
- research/gpu-tabular-2026
- research/causal-representation-frontier-2026
- research/deep-ensemble-frontier-2026
- research/wave7-teacher-distillation
- research/wave8-future-aware-distillation
- research/current

They should also inspect, where relevant:

- research/RESULTS.csv
- research/STATUS.md
- research/FAILED_EXPERIMENTS.md
- research/STATE_OF_RESEARCH.md
- research/reports/
- Git history
- relevant source code
- OOF analyses
- pair-flow analyses
- dominant-cell analyses
- RT600 and RT1257 artifacts and reports

## Rules for agents

- Do not modify model code during this stage.
- Do not run expensive training.
- Do not allocate RT experiment IDs.
- Do not modify research/RESULTS.csv.
- Do not modify either source prompt.
- This stage is research analysis only.
- Do not read any other agent response before completing and committing your own report.
- Clearly distinguish measured facts, inference, hypotheses, and proposed experiments.
- Cite repository evidence wherever possible.

## Response convention

Each agent will eventually write exactly one Markdown report under:

research/multi_agent_frontier_20260829/responses/

Suggested filenames:

agent_01.md
agent_02.md
agent_03.md
agent_04.md
agent_05.md

Agents must not modify another agent's response.

## Synthesis

No synthesis should occur until all independent responses have been completed and frozen.

The eventual synthesis should be written under:

research/multi_agent_frontier_20260829/synthesis/final_synthesis.md
