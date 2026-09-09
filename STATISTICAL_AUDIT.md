# Statistical Audit

All reported intervals are 95% Wilson intervals for independent episode-level decisions unless otherwise noted. Seeds are reported separately from episodes; seed variation is never treated as additional sample size.

## Core mechanism

The binary action-ambiguous benchmark uses 2,000 held-out episodes per seed and five seeds. The 1-code model averages 50.10% (seed SD 0.42%), consistent with chance. The 2-code model averages 100.00%; the exact two-sided binomial test against 50% has p < 10^-300 for every seed. Four- and eight-code models average 99.97% and 99.96%, respectively.

## Natural-language objective comparison

The positional-attention benchmark uses 2,000 held-out paraphrase episodes per seed and five seeds with the same 2-code capacity. Mean accuracies are 98.47% +/- 3.06% for the Q objective, 51.11% +/- 3.24% for ordered reconstruction, and 29.81% +/- 18.31% for scalar reward prediction. The action-only/random baseline is 50%.

## Real Qwen interface

The Qwen compressed-state interface benchmark uses 100 episodes. Full history and explicitly formatted compressed state both achieve 100/100; last-12-token truncation achieves 64/100. For 100/100, the Wilson lower bound is 96.3%; for 64/100, it is 54.2%. These intervals are descriptive because episodes share a generated task family.

## Real TextWorld agent

The environment-level benchmark uses 20 independent generated games. Native walkthrough success is 20/20. Qwen full-history success is 2/20 (10.0%, Wilson 2.8--30.1%); truncated-history success is 1/20 (5.0%, Wilson 0.9--23.6%). This sample is a baseline diagnostic, not a powered claim of general agent performance.

Candidate-action log-likelihood ranking produced 3/20 full-history successes (15%) and 6/20 truncated-history successes (30%). This confirms that the free-form generation parser was not the sole cause of the low TextWorld rate.

The learned action-family hint variant produced 3/20 full-history successes (15%), 3/20 full-history-plus-hint successes (15%), and 0/20 hint-only successes. Its memory classifier training accuracy was 99.17%; the gap is therefore in Qwen grounding/planning, not label fitting.

Qwen2.5-7B on the same TextWorld protocol achieved 1/20 full-history successes (5%) and 3/20 truncated-history successes (15%). The scale control does not support attributing the 3B failure solely to model size.

## Multi-seed Qwen compression

The end-to-end learned-memory/Qwen integration used three independently trained memory seeds and 50 held-out episodes per seed (150 total). Full-history and learned-compressed conditions both achieved 150/150; the Wilson 95% lower bound is 97.5%. Truncated history achieved 68/150 (45.3%, Wilson 37.6--53.3%). Because full and learned conditions tie at the ceiling, this supports retention/equivalence rather than a claim of improvement.

## Interpretation boundary

The strongest statistically supported claim is mechanism-level: a two-state/two-code memory preserves delayed decision information in action-ambiguous natural-language tasks, and decision-target training strongly outperforms text reconstruction on that task family. The evidence does not establish broad performance across LLM agent benchmarks; the TextWorld baseline is low and the environment-level sample is intentionally small.

## External non-oracle MiniGrid control

The public MiniGrid Memory experiment uses five independently trained models, 2,000 expert trajectories per training seed, and the same 500 held-out environment seeds for every model. Test policies receive only standard partial observations and the fixed `left/right/forward` action space. Optimization uncertainty is reported as the SD across training seeds. Model differences use a hierarchical paired bootstrap that resamples both training seeds and paired held-out tasks; repeated evaluation of the same 500 tasks is not treated as 2,500 independent environments.

An independent expert audit solved 500/500 evaluation seeds in each of S13 Random, S11, and S17 Random. The two cue classes occurred 246/254, 262/238, and 246/254 times, respectively, ruling out environment failure and material class imbalance as explanations of model differences.

On `MiniGrid-MemoryS13Random-v0`, explicit four-code policy-equivalence memory achieves 99.76% +/- 0.54%, compared with 100.00% +/- 0.00% for continuous recurrent state, 85.12% +/- 22.23% for learned full-history retrieval, 51.80% +/- 2.68% for one code, 50.60% +/- 3.29% for a four-step window, 49.40% +/- 3.29% for current-observation reconstruction, and 50.60% +/- 3.29% for four-code policy-only training.

The paired Q4 improvements are 47.96 percentage points versus one code (95% CI 45.48--50.64), 49.16 versus recent window (46.16--52.28), 50.36 versus reconstruction (47.20--53.48), and 49.16 versus policy-only discrete memory (46.20--52.28). Q4 is 0.24 points below continuous state (-0.80--0.00) and 14.64 points above learned retrieval (1.20--34.64). The latter comparison reflects retrieval seed instability, not an assertion that retrieval cannot solve the task.

Zero-shot Q4 success is 100.00% +/- 0.00% on S11 and 98.04% +/- 2.63% on longer S17 Random. The S17 difference versus learned retrieval is 12.16 points with an interval crossing zero (-1.36--32.96), so superiority over retrieval is not claimed for length transfer.

The nested data sweep gives 46.24% +/- 20.14%, 89.72% +/- 22.54%, 99.36% +/- 0.61%, and 99.04% +/- 0.78% for 250, 500, 1,000, and 2,000 trajectories. Two sparse downstream decision-target attempts achieved 47.0% and 47.2%. These are negative optimization results and bound the positive result to explicit policy-equivalence supervision rather than automatic discovery from sparse action/reward targets.

## Replay-Q label and outcome audit

The replay-Q audit covers 2,000 independent training environment seeds. Counterfactual branch replay produced exactly one positive-reward branch in every episode, and the anonymous class read from the successful terminal standard observation agreed with the explicit policy-equivalence class in 2,000/2,000 cases. Replay class counts were 1,002 and 998.

Five Q4 models trained from replay-derived labels achieved 99.76% +/- 0.54% on S13 Random, 100.00% +/- 0.00% on S11, and 98.04% +/- 2.63% on S17 Random. For every environment, the same 500 held-out task seeds were evaluated for each training seed. All 2,500 paired outcomes, action sequences, and code sequences per environment matched the explicit-label runs exactly (7,500/7,500 overall). This exact match is a paired reproducibility result, not 7,500 independent samples.

The inference is limited to structured replay/value-derived supervision. It does not imply that the representation was discovered end to end from one sparse terminal reward, because expert trajectories, counterfactual branching, and backward propagation of the recovered class are supplied by the procedure.

## Formal ALFWorld external pilot

The formal QLoRA policy used Qwen2.5-7B-Instruct, 180 balanced training games, 154 solved expert trajectories, 2,729 action examples, two epochs, and 342 updates. The fixed `valid_unseen` evaluation used 30 paired games (selection seed 20260827), with no admissible commands, expert plans, facts, or policy commands exposed at test time. Full history solved 22/30 (73.3%, Wilson 55.6--85.8%), a four-turn window solved 18/30 (60.0%, 42.3--75.4%), and the original neural-memory condition solved 6/30 (20.0%, 9.5--37.3%).

Code audit found that the original memory updater used the action LoRA adapter for summary generation. The corrected run disabled the adapter only during summary generation and used the identical 30 games; corrected memory solved 12/30 (40.0%, 24.6--57.7%). Corrected versus original memory had six corrected-only wins and no original-only wins (exact McNemar p=0.03125; paired bootstrap delta +20.0 percentage points, 95% percentile interval +6.7--+36.7). Corrected memory remained below full history (11 full-only versus 1 corrected-only, p=0.00635; bootstrap delta -33.3 points, -53.3-- -13.3) and below the window (7 versus 1, p=0.0703; bootstrap delta -20.0 points, -36.7-- -3.3). All conditions had zero invalid-action markers. Corrected memory failed on 18/30 episodes, all reaching the 50-step cap.

This is a paired external stress test and bug-fix result, not evidence of practical parity or broad LLM-agent superiority. The detailed paired tables and task-type breakdown are in `alfworld_final_analysis.json` and `ALFWORLD_EXPERIMENT_CONCLUSION.md`.
