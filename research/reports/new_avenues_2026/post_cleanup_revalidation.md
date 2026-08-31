# POST-CLEANUP REVALIDATION -- PILOTS 1-4

Cleanup SHA: `aca2c4f9b68ad6315956f7c499ebefb2ae311f7b`.

Merge commit: `81aa56c`.

## Tests

| check | result |
|---|---|
| `pytest -q tests/test_novel_streams_harness.py` | 11 passed |
| `research/scripts/check_research_hygiene.py` | OK: 215 experiment rows, no duplicate IDs |

## Reproduction Tolerance

Floating metrics were compared with absolute tolerance `1e-12`. Pair-flow
repairs, damage, and net counts were required to match exactly.

## Results

| item | old marginal | clean marginal | changed | final verdict |
|---|---:|---:|---|---|
| Pilot 1 diagnostic | n/a | n/a | no affected path | WEAK |
| `RT-1200` relay score-state | -0.000206633 | -0.000206633 | no | KILL |
| `RT-1201` IM2+dwell scalar | +0.000301470 | +0.000301470 | no | KILL |
| `RT-1202` trajectory geometry | -0.002586857 | -0.002586857 | no | KILL |
| `RT-1203` shuffled-order control | -0.003041629 | -0.003041629 | no | negative control |

All cleaned-harness values reproduce the original pilot values exactly under
the preregistered definitions. No new RT IDs were allocated, no duplicate
`RESULTS.csv` rows were appended, and no lockbox/test data was touched.

## Production Stream Parity Note

The pre-existing production parity issue in
`m07_bayes::bo_p_lt25_z` does not invalidate these pilot screens. Pilots 1-4
used committed RT-600 OOF arrays and/or raw-history candidate transforms; none
recomputed production `m07_bayes` stream-engine features or read
`m07_bayes::bo_p_lt25_z` directly. The parity issue remains separate work.

## Final Status

The pre-cleanup provisional labels are retired in the reports. Pilots 1-4 are
final screen results, and no candidate cleared the `+0.001` marginal-vs-clone
screen.
