# MiniGrid External Benchmark Conclusion

## Protocol

- Environment: public `minigrid==2.3.1`, trained on `MiniGrid-MemoryS13Random-v0`.
- Evaluation interface: standard 7x7 partial observations and the fixed non-oracle action set `left`, `right`, `forward`.
- Training supervision: shortest-path expert demonstrations plus either (i) explicit three-class policy-equivalence targets or (ii) anonymous classes derived from counterfactual branch replay rewards and terminal standard observations. Environment internals are never available to the evaluated policy.
- Main sample: five independently initialized models, 2,000 training trajectories per seed, and 500 shared held-out environment seeds per condition.
- Transfer: zero-shot evaluation on `MiniGrid-MemoryS11-v0` and the longer `MiniGrid-MemoryS17Random-v0`.
- Uncertainty: seed SD for optimization variation and a hierarchical paired bootstrap over training seeds and held-out tasks for model differences.
- Independent environment audit: expert trajectories solve 500/500 evaluation seeds in each of S13, S11, and S17. Cue counts are 246/254, 262/238, and 246/254, respectively.

## Main result

| Model | S13 Random | S11 | S17 Random |
|---|---:|---:|---:|
| Four-code policy-class memory | 99.76% +/- 0.54% | 100.00% +/- 0.00% | 98.04% +/- 2.63% |
| Continuous GRU state | 100.00% +/- 0.00% | 100.00% +/- 0.00% | 100.00% +/- 0.00% |
| Learned full-history retrieval | 85.12% +/- 22.23% | 86.36% +/- 21.38% | 85.88% +/- 21.86% |
| One-code control | 51.80% +/- 2.68% | 49.76% +/- 0.36% | 51.80% +/- 2.68% |
| Four-step recent window | 50.60% +/- 3.29% | 49.92% +/- 0.44% | 50.60% +/- 3.29% |
| Observation reconstruction | 49.40% +/- 3.29% | 50.08% +/- 0.44% | 49.40% +/- 3.29% |
| Four-code policy-only training | 50.60% +/- 3.29% | 49.92% +/- 0.44% | 50.60% +/- 3.29% |

On the primary S13 task, four-code memory exceeds the one-code control by 47.96 percentage points (hierarchical paired 95% CI 45.48--50.64), the recent window by 49.16 points (46.16--52.28), observation reconstruction by 50.36 points (47.20--53.48), and policy-only discrete memory by 49.16 points (46.20--52.28). It is within 0.24 points of the uncompressed continuous state (-0.80--0.00).

The learned retrieval model is a meaningful strong baseline rather than a strawman: three seeds reach 87--100%, but one seed remains at 46.6%. Four-code memory improves mean primary-task success by 14.64 points (1.20--34.64), mainly by eliminating this optimization failure mode. On the longer S17 transfer, the corresponding interval includes zero (-1.36--32.96), so no universal superiority over learned retrieval is claimed.

## Capacity and data efficiency

One code is insufficient. Three codes are theoretically sufficient for the pre-cue/key/ball classes but optimize unreliably (90.12% +/- 13.60%). Four and eight codes both reach 99.76% +/- 0.54%. Every four-code seed uses exactly three codes; the fourth is unused redundancy rather than hidden extra information.

With nested training sets, four-code success changes from 46.24% +/- 20.14% at 250 trajectories, to 89.72% +/- 22.54% at 500, 99.36% +/- 0.61% at 1,000, and 99.04% +/- 0.78% at 2,000. Reliable training therefore appears between 500 and 1,000 demonstrations for this implementation; the non-monotonic final means are within small seed variation and should not be interpreted as degradation from more data.

## Negative controls and boundary

Two attempts to learn the code using only the single downstream branch action target achieved 47.0% and 47.2%. Inspection confirmed that the policy executed valid trajectories and produced left/right codes, but failed to bind the early cue reliably to the later decision. Structured replay-Q supervision solves this temporal credit-assignment problem; a single sparse action or terminal-reward target does not in this implementation.

## Replay-Q supervision

For each training trajectory, the replay-Q procedure resets the environment to the branch snapshot by replaying the observed action prefix, evaluates both branch actions, and identifies the positive-value branch from terminal reward. It reads the anonymous object category only from the successful terminal standard observation and propagates that equivalence class back to the cue. The memory learner never reads `success_pos`, key/ball names, or an explicit cue-state label.

Across 2,000 independently audited training seeds, replay-derived classes agree with the explicit policy classes on 2,000/2,000 episodes, with balanced class counts of 1,002/998. Five replay-Q-trained Q4 models achieve 99.76% +/- 0.54% on S13, 100.00% +/- 0.00% on S11, and 98.04% +/- 2.63% on S17. Under matched training and evaluation seeds, all 7,500 paired evaluation outcomes, action sequences, and code sequences are identical to the explicit-label models.

This closes the explicit-label objection for the following scoped claim:

> Policy-equivalence classes can be obtained from counterfactual replay values and standard observations, then distilled into a low-rate memory without exposing privileged state labels to the learner or policy.

It does not establish end-to-end discovery from a single sparse terminal reward: expert trajectories, controlled counterfactual branch evaluation, and temporal propagation of the derived class remain part of supervision.

This experiment closes the previous external-environment and non-oracle-action gap for the scoped mechanism claim. It supports:

> Explicit supervision of a low-rate policy-equivalence memory preserves the information needed for delayed decisions in a public partially observable control environment, matching continuous recurrent state and outperforming recent-window, reconstruction, policy-only, and unstable learned-retrieval alternatives.

It does not establish broad LLM-agent gains, learned-memory improvement on open TextWorld, or end-to-end discovery from a single sparse terminal reward. The first two require a competitive policy on an open language-agent benchmark; the last requires substantially weaker supervision than the replay-Q protocol used here.

## Reproducibility

- Runner: `minigrid_qmemory_benchmark.py`
- Statistical scripts: `analyze_minigrid_results.py`, `summarize_minigrid_suite.py`
- Machine-generated suite table: `minigrid_suite_summary/SUMMARY.md`
- Raw per-episode logs: `minigrid_qmemory_results/results.json` and the other `minigrid_*_results` / `minigrid_nested*` directories.
- Environment audit: `minigrid_oracle_audit.py` and `minigrid_oracle_audit.json`.
- Replay-Q audit: `minigrid_replay_q_audit.py`, `minigrid_replay_q_audit.json`, `analyze_replay_q_results.py`, and `minigrid_replay_q_results/AUDIT.md`.
- Hardware: one RTX 4090 with 49,140 MiB reported VRAM; PyTorch 2.5.1+cu124, CUDA 12.4, Gymnasium 1.3.0, MiniGrid 2.3.1.
- Recorded formal-suite runtime, including one repeated policy-only control: 42.3 wall-clock minutes.
