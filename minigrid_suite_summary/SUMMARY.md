# MiniGrid External Benchmark Suite

All values are recomputed from the raw episode logs. Rates are percentages; uncertainty after +/- is the sample SD across five independently trained models.

## MiniGrid-MemoryS13Random-v0

| Model | Per-seed success | Mean +/- SD | Q4 minus model (paired 95% CI) |
|---|---:|---:|---:|
| q4 | 100.0, 98.8, 100.0, 100.0, 100.0 | 99.76 +/- 0.54 | reference |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00 | -0.24 pp [-0.80, 0.00] |
| learned_retrieval | 46.6, 100.0, 87.0, 92.0, 100.0 | 85.12 +/- 22.23 | 14.64 pp [1.20, 34.64] |
| q1 | 53.0, 47.0, 53.0, 53.0, 53.0 | 51.80 +/- 2.68 | 47.96 pp [45.48, 50.64] |
| window | 47.0, 47.0, 53.0, 53.0, 53.0 | 50.60 +/- 3.29 | 49.16 pp [46.16, 52.28] |
| observation_reconstruction | 53.0, 47.0, 47.0, 53.0, 47.0 | 49.40 +/- 3.29 | 50.36 pp [47.20, 53.48] |
| discrete_policy | 53.0, 47.0, 53.0, 47.0, 53.0 | 50.60 +/- 3.29 | 49.16 pp [46.20, 52.28] |

## MiniGrid-MemoryS11-v0

| Model | Per-seed success | Mean +/- SD | Q4 minus model (paired 95% CI) |
|---|---:|---:|---:|
| q4 | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00 | reference |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00 | 0.00 pp [0.00, 0.00] |
| learned_retrieval | 49.2, 100.0, 87.6, 95.0, 100.0 | 86.36 +/- 21.38 | 13.64 pp [1.04, 32.80] |
| q1 | 49.6, 50.4, 49.6, 49.6, 49.6 | 49.76 +/- 0.36 | 50.24 pp [48.28, 52.20] |
| window | 50.4, 50.4, 49.6, 49.6, 49.6 | 49.92 +/- 0.44 | 50.08 pp [48.08, 52.04] |
| observation_reconstruction | 49.6, 50.4, 50.4, 49.6, 50.4 | 50.08 +/- 0.44 | 49.92 pp [47.96, 51.92] |
| discrete_policy | 49.6, 50.4, 49.6, 50.4, 49.6 | 49.92 +/- 0.44 | 50.08 pp [48.08, 52.08] |

## MiniGrid-MemoryS17Random-v0

| Model | Per-seed success | Mean +/- SD | Q4 minus model (paired 95% CI) |
|---|---:|---:|---:|
| q4 | 100.0, 93.8, 97.2, 99.2, 100.0 | 98.04 +/- 2.63 | reference |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00 | -1.96 pp [-4.32, -0.16] |
| learned_retrieval | 47.6, 99.2, 89.0, 93.6, 100.0 | 85.88 +/- 21.86 | 12.16 pp [-1.36, 32.96] |
| q1 | 53.0, 47.0, 53.0, 53.0, 53.0 | 51.80 +/- 2.68 | 46.24 pp [43.92, 48.44] |
| window | 47.0, 47.0, 53.0, 53.0, 53.0 | 50.60 +/- 3.29 | 47.44 pp [44.28, 50.88] |
| observation_reconstruction | 53.0, 47.0, 47.0, 53.0, 47.0 | 49.40 +/- 3.29 | 48.64 pp [45.64, 51.76] |
| discrete_policy | 53.0, 47.0, 53.0, 47.0, 53.0 | 50.60 +/- 3.29 | 47.44 pp [44.48, 50.64] |

## Capacity

| Codes | Per-seed success | Mean +/- SD |
|---:|---:|---:|
| 1 | 53.0, 47.0, 53.0, 53.0, 53.0 | 51.80 +/- 2.68 |
| 3 | 77.2, 73.4, 100.0, 100.0, 100.0 | 90.12 +/- 13.60 |
| 4 | 100.0, 98.8, 100.0, 100.0, 100.0 | 99.76 +/- 0.54 |
| 8 | 98.8, 100.0, 100.0, 100.0, 100.0 | 99.76 +/- 0.54 |

## Nested Data Efficiency

| Training trajectories | Per-seed success | Mean +/- SD |
|---:|---:|---:|
| 250 | 49.2, 27.0, 27.0, 75.0, 53.0 | 46.24 +/- 20.14 |
| 500 | 100.0, 100.0, 100.0, 49.4, 99.2 | 89.72 +/- 22.54 |
| 1000 | 98.8, 98.8, 100.0, 100.0, 99.2 | 99.36 +/- 0.61 |
| 2000 | 100.0, 98.8, 98.0, 98.8, 99.6 | 99.04 +/- 0.78 |

## Audit Notes

- Q4 uses three of four available codes in every seed: pre-cue, key cue, and ball cue.
- The learned retrieval baseline can solve the task but is unstable across seeds; this is not evidence that retrieval is intrinsically incapable.
- Two action-target-only optimization attempts stayed at 47.0% and 47.2%; the positive Q4 result therefore requires explicit policy-class/Q-equivalence supervision.
- Evaluation uses only standard partial observations and the fixed {left, right, forward} action space. Environment internals are used only to generate training demonstrations and sufficient-class labels.
- Recorded suite runtime, including one repeated policy-only control: 42.3 wall-clock minutes on one RTX 4090.
