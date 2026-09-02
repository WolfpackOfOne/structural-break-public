# RT-1320 Promotion Prep

Generated: `2026-09-02T00:54:56Z`
Branch: `engineering/rt1320-promotion-prep`
Git SHA: `9d2e6c962ac6a6a6fa9d376811632f0f775a90d2`

Status: **PROMOTION_BLOCKED**

RT-1320 remains the best live post-RT-1257 engineering target, but promotion is blocked until the missing gates are closed.

## Current Decision

Work on RT-1320, not another GPU tabular family. RT-1320 is the only current research-alive candidate with a champion-relative addition endpoint above the measured noise floor. It is not deployable yet.

## Artifact Contract

- Experiment ID: `RT-1320`
- Members: `8` total, RT-1257 plus one residual student
- Student model file: `model.txt.7`
- Student objective: `regression`
- Student score scale: member-local `SCDF_NSEEN` calibration before the equal-weight blend
- Feature bank: existing 500 causal columns, unchanged RT-600 manifest SHA
- Final fit partition: `folds_final10k`, 10,000 series

## Gate Status

| gate | status | note |
|---|---|---|
| `champion_relative_endpoint` | `passed` | champion-relative addition endpoint clears the current noise floor |
| `nested_teacher_student_report` | `passed` | nested teacher/student report gates are present and internally clean |
| `alternate_partition_leg` | `passed` | four-partition leg complete; plan §4 rule evaluates to PASS (mean +0.0017678, 4/4 positive, worst +0.0014825). This is the frozen arithmetic, not the owner's promotion sign-off. |
| `student_artifact_causality` | `passed` | artifact-level causality tests ran and passed |
| `final10k_target_and_fit` | `passed` | model manifest records an 8-member folds_final10k fit |
| `production_artifact_manifest` | `passed` | model manifest matches the RT-1320 8-member production contract |
| `crunch_test` | `passed` | Crunch test passed |
| `external_score` | `missing` | RT-1320 has no official external score as an 8-member system |

## Next Server Work

Prepare the folds-final10k Arm-C residual target and train the 8th-member student. The target must be regenerated under the final 10,000-series population; the existing nested teacher labels only cover the 8,000 dev series and cannot be reused as a final artifact.

After a model directory exists, run:

```bash
python research/scripts/rt1320_promotion_prep.py validate-artifact --model-dir models/rt1320_final
python research/scripts/rt1320_promotion_prep.py audit --model-dir models/rt1320_final
```

Promotion remains blocked until the artifact causality report, alternate partition leg, Crunch test, and external score are all recorded.
