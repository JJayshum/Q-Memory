# Stage B Pilot Conclusion

## Completed run

- Environment: TextWorld 1.7.0 on the provisioned RTX 4080 SUPER host.
- Configuration: 2 deterministic games, 1 snapshot per game, 1 paired rollout per action, horizon 2.
- Candidate actions: oracle admissible commands (a privileged replay-estimator gate, not the confirmatory frozen-generator setting).
- Replay audit: reset match 1.0, replay transition match 1.0, copy-branch match 1.0.
- Candidate coverage: 6 actions per snapshot (minimum 6).
- Labels: 24 action/policy rows; oracle-success labels included both 0 and 1 (max 1.0), with `go east` selected as successful in the inspected game; random continuation remained 0 in this short horizon.
- Cost: 50.3 s wall time, 72 environment steps, 0 GPU hours. GPU is not the bottleneck for this stage.

## Interpretation

The Stage B replay mechanism passes its deterministic correctness gate and can produce non-degenerate counterfactual action labels when the continuation policy follows TextWorld's generated walkthrough. The initial lexicographic policy produced all-zero labels; that was a label-quality failure caused by unreachable success within the horizon, not evidence against the project. The walkthrough-based pilot fixes that issue for this gate.

This is not yet evidence that a learned memory policy improves task success. The candidate set is privileged (`admissible_commands`), the sample is intentionally small, and the oracle walkthrough is used only to make a sensitivity label. A larger train/test replay set and the learned-memory training/evaluation step are required before a confirmatory conclusion.

The attempted 8-game expansion was stopped after exceeding the practical runtime budget because repeated TextWorld environment copies made the configuration too slow; this is an engineering-cost constraint, not a negative scientific result.

## Medium replay and learned-memory pilot

An optimized 12-game run (8 train games, 4 held-out games) completed in 193--199 seconds. All deterministic replay gates remained at 1.0, with 5.75 candidate actions per snapshot. Walkthrough-aligned snapshot enumeration improved oracle-positive coverage from 7/36 to 10/36 snapshots, but labels remained sparse (10/207 oracle action rows nonzero).

The CUDA learned-memory pilot trained 2-, 4-, and 8-code GRU bottlenecks over these labels (three seeds). Mean held-out MSE improvements versus a train-mean constant baseline were -0.00894, -0.00004, and +0.00043 respectively. The 8-code result was positive in 2/3 seeds but is far too small and variable to support a learned-memory advantage claim.

**Stage B conclusion:** replay correctness is supported; learned-memory efficacy is not yet established. The next required experiment is a larger, less sparse label set (or a denser value target) before drawing a practical conclusion.

## Larger learned-memory check

The 24-game replay set was used for a five-seed CUDA run. Relative to the train-mean constant baseline, mean held-out MSE improvements were:

| Codes | Mean improvement | Positive seeds |
|---:|---:|---:|
| 2 | -0.00335 | 3/5 |
| 4 | +0.00469 | 5/5 |
| 8 | +0.00059 | 4/5 |

This is a more encouraging signal for a 4-code bottleneck than the 12-game pilot, but it remains a small value-prediction pilot with privileged admissible-action candidates and sparse nonzero labels (24/415 oracle rows). It supports proceeding to a denser-label design, not a final claim of project success.

## Dense centered-advantage follow-up

To test whether the preceding signal reflected genuine action ranking, the same 24-game data were retrained using the replay-estimated centered advantage (`Q - mean_action_Q`) as the target for every candidate action. Across five seeds, mean held-out improvements versus a constant baseline were -0.00196 (2 codes), -0.00006 (4 codes), and -0.00033 (8 codes); no seed improved. The model often used only one discrete code. This negative control weakens the claim that the earlier reward-mean improvement represents robust Q-sufficient memory.

**Updated Stage B conclusion:** replay infrastructure is correct, but learned-memory efficacy is unresolved and currently not supported under the denser advantage target. The practical project should not be declared successful without a better-conditioned value target and a non-privileged candidate-action evaluation.

## Strict action-ranking audit

A grouped action-ranking experiment removed action text from the recurrent memory input and evaluated only the 10 held-out snapshots with a positive oracle action. An initial model exposed an action-only scoring bypass, so the final audit removed that bypass and compared against an explicit one-code model.

| Codes | Mean top-1 | Mean regret | Codes used |
|---:|---:|---:|---:|
| 1 | 0.86 | 0.14 | 1 |
| 2 | 0.86 | 0.14 | 1 |
| 4 | 0.82 | 0.18 | 1 |
| 8 | 0.84 | 0.16 | 1 |

The lexicographic baseline achieved 0.30 top-1 and 0.70 regret. Although the learned action scorer is useful, every multi-code model collapsed to one code and none outperformed the one-code control. Thus the gain is attributable to regularities in action text, not to learned observation memory.

**Current project decision:** the Stage A controlled mechanism passes and Stage B replay engineering passes, but this TextWorld pilot does not provide evidence that additional discrete memory carries Q-relevant information. A new benchmark construction must force identical or ambiguous action sets to require history-dependent decisions before further scale-up is scientifically justified.

## Action-ambiguous recovery experiment

That benchmark was implemented directly: every example presents the same two action names, while the correct action is determined only by the latest state update in the observation history. This removes the action-text shortcut identified in TextWorld.

| Codes | Held-out accuracy | Seeds |
|---:|---:|---:|
| 1 | 50.10% +/- 0.42% | 5 |
| 2 | 100.00% +/- 0.00% | 5 |
| 4 | 99.97% +/- 0.06% | 5 |
| 8 | 99.96% +/- 0.04% | 5 |

The 1-code model is at chance, while a 2-code memory recovers the delayed decision boundary perfectly. This is strong evidence that the discrete bottleneck can preserve Q-relevant history when the task genuinely requires memory. It does not rescue the TextWorld claim: the benchmark is synthetic and deliberately action-ambiguous. The appropriate conclusion is that the mechanism is real, but the current TextWorld environment/task construction is not diagnostic of it.

## Practical natural-language attempt

A more natural-language variant added long distractor histories, symmetric door actions, and held-out paraphrase templates. The first run produced 50.0% for 1 code and only 54--55% for 2/4/8 codes. An in-distribution control remained at 54--55%, and a one-seed diagnostic had 50.35% training accuracy. This identifies a benchmark/model engineering failure: the current GRU setup is not learning the natural-language task at all. These numbers are not evidence against the memory mechanism and are excluded from the scientific claim.

The practical-evidence track therefore remains open. Before using this variant in a paper, fix the language-task optimization (first verify a continuous no-bottleneck encoder and a simple lexical baseline), then rerun the discrete bottleneck comparison.

## Practical benchmark repair and generalization

The implementation was repaired by reading the last non-padding GRU state and tuning away an overly strong code-balance penalty. A continuous GRU reached 100% train/test accuracy, and a 2-code discrete model reached 100% in 3/3 in-distribution seeds.

With held-out paraphrase templates and five tuned seeds, the 2-code model obtained accuracies 0.850, 0.504, 1.000, 0.508, and 0.966 (mean 0.765, SD 0.218). The three successful runs fit the training set perfectly; the two failures also failed on training, indicating optimization instability rather than a clean generalization failure.

This is preliminary practical evidence: the model can transfer the delayed decision rule to unseen natural-language templates, but training reliability is inadequate for a strong claim. A robust practical paper result requires reducing this seed sensitivity or reporting a principled failure-rate analysis.

## Practical optimization iteration

I tested deterministic softmax training, direct two-code supervision, and continuous-to-discrete pretraining. Continuous GRU training is reliable (100% train/test on the in-distribution control), but discrete training remains seed-sensitive on the long natural-language history. The best held-out-template runs so far are 0.850, 0.504, 1.000, 0.508, and 0.966 across five seeds; continuous-to-discrete pretraining produced 0.778, 0.499, 0.495, 0.953, and 0.674. These results are useful diagnostics but not yet a publication-grade practical claim.

The remaining engineering requirement is a stable quantization procedure (for example, staged codebook fitting or a rate-constrained distillation objective) with a pre-registered failure-rate metric. The paper should not claim practical robustness until that procedure is validated across independent language templates and seeds.

## Positional-attention practical result

Replacing the unstable long-sequence GRU optimizer with a positional self-attention pooling encoder produced a stable practical result while retaining a two-class discrete memory code. The model was trained on 6,000 natural-language histories using four templates and evaluated on 2,000 histories using three unseen paraphrase templates. Across five seeds, test accuracies were 0.924, 1.000, 1.000, 1.000, and 1.000 (mean 0.985, SD 0.031); all five training accuracies were 1.000. The action-only/random baseline is 0.50 because the two action strings are symmetric.

This is the first practical-language result with both high accuracy and reliable optimization. It supports a paper claim that a small discrete memory can transfer a delayed decision rule across natural-language paraphrases when action text is non-informative. It remains a controlled benchmark, not evidence of general LLM-agent performance; a final paper should add an independent environment and confidence intervals before claiming broad applicability.

## Objective-matched baselines

Using the same positional-attention encoder, two-code bottleneck, train/test examples, and five seeds, I compared the decision-targeted Q objective against ordered-text reconstruction and scalar reward prediction:

| Training objective | Mean test accuracy | SD |
|---|---:|---:|
| Q / decision target | 98.47% | 3.06% |
| Ordered-text reconstruction | 51.11% | 3.24% |
| Scalar reward target | 29.81% | 18.31% |

The action-only/random baseline is 50%. Thus the practical benchmark now provides an objective-matched result: the decision advantage is not explained by the encoder, memory capacity, or language templates alone. Training the same bottleneck to reconstruct text does not preserve the delayed decision boundary, while the Q objective does. The reward baseline is intentionally weak because a scalar terminal target discards the action-conditioned distinction; this is itself evidence for using action-value/advantage supervision rather than undifferentiated reward prediction.

## Real LLM backbone smoke benchmark

Qwen2.5-3B-Instruct was downloaded from ModelScope and loaded on an RTX 4090. On 100 generated natural-language episodes with two symmetric door actions, deterministic Qwen generation achieved 100% with the full history, 64% with only the last 12 tokens, and 50% with the first natural-language compressed-state prompt. This confirms that a real LLM depends on long history for this task, but the compressed prompt was not a learned memory module and is treated only as a diagnostic.

## Independent four-state extension

An independent four-state, four-action benchmark was added with unseen language templates and 8,000/3,000 train/test examples. The continuous attention encoder reached 100% in all five seeds. The initial learned code-to-action head collapsed (25--50%); direct supervision of the 4-way discrete code fixed this completely: all five seeds reached 100% train and 100% test accuracy using all four codes. This confirms the expected scaling from one-bit binary state to two-bit four-state memory, while exposing the need to separate code learning from code-to-action head optimization.

A rerun with an explicit compressed-memory instruction completed all 100 episodes and achieved 100% full-history, 64% truncated-history, and 100% compressed-state accuracy. This control shows that Qwen can use a correctly formatted one-step state summary; it does not establish that a learned memory module can produce that summary. The result is therefore practical evidence for state compression as an interface, not yet evidence of end-to-end learned Q-memory integration.

The log-likelihood action-ranking TextWorld baseline also completed: 3/20 games (15%) with full history and 6/20 (30%) with truncated history. Replacing free-form generation with candidate-action probability ranking did not recover success, so the low rate reflects the frozen 3B model's planning/grounding limitation rather than a parser-only artifact.

A learned action-family memory was trained on 40 TextWorld games (99.17% training accuracy) and evaluated on 20 independent games. Injecting its hint into full-history Qwen left success unchanged at 3/20 (15%); using only current observation plus the hint yielded 0/20. The memory therefore learned simulator action labels, but Qwen did not reliably ground the hint into executable commands. This isolates the remaining practical gap to the language-to-action interface and open-task planning.

## End-to-end learned memory with Qwen

The final integration trained a two-class attention memory on 6,000 histories, froze Qwen2.5-3B-Instruct, and evaluated 100 held-out episodes. After correcting the Qwen chat-template interface, full-history accuracy was 100%, truncated-history accuracy 47%, and learned-memory compressed-state accuracy 100%; the memory training accuracy was 99.98%. This demonstrates an end-to-end pipeline in which a learned discrete state preserves the Qwen decision on the constructed task. The benchmark remains controlled and uses an oracle-aligned synthetic state, so it is evidence of feasibility rather than broad agent generalization.

## Real Qwen TextWorld agent

Qwen2.5-3B-Instruct was also evaluated as an environment-level agent selecting from actual TextWorld admissible commands in 20 generated games. The native walkthrough solved 20/20 games, confirming environment validity. Qwen solved 2/20 (10%, Wilson 95% CI 2.8--30.1%) with full history and 1/20 (5%, CI 0.9--23.6%) with truncated history. This is a realistic negative baseline: the frozen small model is weak at open TextWorld planning, so learned memory should next be tested with a stronger action-policy interface rather than summary injection into zero-shot planning.

## Multi-seed end-to-end Qwen integration

## Backbone scale control

Qwen2.5-7B-Instruct was evaluated on the same 20-game TextWorld free-generation protocol. It solved 1/20 games (5%) with full history and 3/20 (15%) with truncated history, versus 2/20 and 1/20 for the earlier 3B run. The larger backbone did not improve open-task success, reinforcing that the dominant practical limitation is planning/grounding and candidate-action interface rather than parameter count alone.

The learned-memory integration was repeated with three independently trained memory seeds and 50 new held-out episodes per seed (150 total). Full-history Qwen and learned-compressed-state Qwen both solved 150/150 episodes (Wilson lower bound 97.5%). Truncated-history Qwen solved 68/150 (45.3%, Wilson 37.6--53.3%). Memory training was 99.98% accurate. This confirms that the learned two-state summary preserves the decision on this benchmark across memory seeds, while the claim remains bounded to the constructed task family.

## Public MiniGrid non-oracle extension

The external-environment track was continued on `MiniGrid-MemoryS13Random-v0`, where the evaluated policy receives only standard partial observations and selects from the fixed `left/right/forward` action space. Five four-code models trained with explicit pre-cue/key/ball policy-equivalence supervision achieved 99.76% +/- 0.54% across 500 held-out tasks per seed. Continuous recurrent state reached 100%; one code, a four-step window, current-observation reconstruction, and four-code policy-only training remained at approximately 49--52%. A learned full-history retrieval model averaged 85.12% but varied from 46.6% to 100% across seeds.

Four-code memory transferred without retraining at 100% on S11 and 98.04% +/- 2.63% on longer S17 Random. Capacity and nested-data sweeps show that three codes optimize unreliably, four and eight codes are stable, and approximately 1,000 demonstrations are required for reliable training in this implementation.

This closes the public partially observable environment and non-oracle action-interface gap for a scoped mechanism claim. It does not close the broad LLM-agent gap. The positive result also requires explicit policy-equivalence labels: two attempts using only the sparse downstream branch action target remained at 47.0--47.2%. The defensible conclusion is therefore explicit low-rate sufficient-class distillation, not automatic discovery from terminal reward.
