# MiniGrid Q-Memory Result Summary

- Training: 2000 expert trajectories per seed, 5 seeds.
- Evaluation: 500 held-out environment seeds per condition.
- Runtime: 21.8 wall-clock minutes on cuda.
- Intervals below compare Q4 and each baseline with a hierarchical paired bootstrap over training seeds and test tasks.

## MiniGrid-MemoryS13Random-v0

| Model | Seed rates | Mean +/- seed SD |
|---|---:|---:|
| q4 | 100.0, 98.8, 100.0, 100.0, 100.0 | 99.76 +/- 0.54% |
| q1 | 53.0, 47.0, 53.0, 53.0, 53.0 | 51.80 +/- 2.68% |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00% |
| window | 47.0, 47.0, 53.0, 53.0, 53.0 | 50.60 +/- 3.29% |
| retrieval | 30.8, 40.0, 36.2, 33.6, 25.8 | 33.28 +/- 5.38% |
| reconstruction | 53.0, 47.0, 47.0, 53.0, 47.0 | 49.40 +/- 3.29% |

| Q4 comparison | Mean difference | Hierarchical paired 95% CI | Positive seeds |
|---|---:|---:|---:|
| vs q1 | 47.96 pp | [45.48, 50.64] pp | 5/5 |
| vs continuous | -0.24 pp | [-0.80, 0.00] pp | 0/1 |
| vs window | 49.16 pp | [46.16, 52.28] pp | 5/5 |
| vs retrieval | 66.48 pp | [61.56, 71.28] pp | 5/5 |
| vs reconstruction | 50.36 pp | [47.20, 53.48] pp | 5/5 |

## MiniGrid-MemoryS11-v0

| Model | Seed rates | Mean +/- seed SD |
|---|---:|---:|
| q4 | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00% |
| q1 | 49.6, 50.4, 49.6, 49.6, 49.6 | 49.76 +/- 0.36% |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00% |
| window | 50.4, 50.4, 49.6, 49.6, 49.6 | 49.92 +/- 0.44% |
| retrieval | 27.6, 23.2, 27.4, 26.2, 15.0 | 23.88 +/- 5.27% |
| reconstruction | 49.6, 50.4, 50.4, 49.6, 50.4 | 50.08 +/- 0.44% |

| Q4 comparison | Mean difference | Hierarchical paired 95% CI | Positive seeds |
|---|---:|---:|---:|
| vs q1 | 50.24 pp | [48.28, 52.20] pp | 5/5 |
| vs continuous | 0.00 pp | [0.00, 0.00] pp | 0/0 |
| vs window | 50.08 pp | [48.08, 52.04] pp | 5/5 |
| vs retrieval | 76.12 pp | [72.24, 80.92] pp | 5/5 |
| vs reconstruction | 49.92 pp | [47.96, 51.92] pp | 5/5 |

## MiniGrid-MemoryS17Random-v0

| Model | Seed rates | Mean +/- seed SD |
|---|---:|---:|
| q4 | 100.0, 93.8, 97.2, 99.2, 100.0 | 98.04 +/- 2.63% |
| q1 | 53.0, 47.0, 53.0, 53.0, 53.0 | 51.80 +/- 2.68% |
| continuous | 100.0, 100.0, 100.0, 100.0, 100.0 | 100.00 +/- 0.00% |
| window | 47.0, 47.0, 53.0, 53.0, 53.0 | 50.60 +/- 3.29% |
| retrieval | 33.6, 39.4, 38.8, 33.8, 27.8 | 34.68 +/- 4.70% |
| reconstruction | 53.0, 47.0, 47.0, 53.0, 47.0 | 49.40 +/- 3.29% |

| Q4 comparison | Mean difference | Hierarchical paired 95% CI | Positive seeds |
|---|---:|---:|---:|
| vs q1 | 46.24 pp | [43.92, 48.44] pp | 5/5 |
| vs continuous | -1.96 pp | [-4.32, -0.16] pp | 0/3 |
| vs window | 47.44 pp | [44.28, 50.88] pp | 5/5 |
| vs retrieval | 63.36 pp | [57.52, 69.12] pp | 5/5 |
| vs reconstruction | 48.64 pp | [45.64, 51.76] pp | 5/5 |

