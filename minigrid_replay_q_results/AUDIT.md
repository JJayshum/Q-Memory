# Replay-Q Supervision Audit

- Replay labels agree with explicit policy classes on 2000/2000 training episodes.
- Training receives replay-derived anonymous classes, not environment state labels.
- The evaluated policy receives only standard partial observations and the fixed left/right/forward interface.

| Environment | Replay-Q mean +/- seed SD | Outcome matches | Action matches | Code matches |
|---|---:|---:|---:|---:|
| MiniGrid-MemoryS13Random-v0 | 99.76% +/- 0.54% | 2500/2500 | 2500/2500 | 2500/2500 |
| MiniGrid-MemoryS11-v0 | 100.00% +/- 0.00% | 2500/2500 | 2500/2500 | 2500/2500 |
| MiniGrid-MemoryS17Random-v0 | 98.04% +/- 2.63% | 2500/2500 | 2500/2500 | 2500/2500 |

The replay-Q and explicit-label Q4 runs are episode-wise identical under matched training and evaluation seeds.
This establishes distillation from counterfactual replay values and terminal standard observations; it does not establish end-to-end discovery from a single sparse terminal reward.
