# Decision-Preserving Discrete Memory via Policy-Class Q-Distortion

**Manuscript type:** Mechanism and methods paper  
**Status:** Evidence-grounded manuscript draft  
**Scope:** Controlled language decision tasks and public MiniGrid Memory; open-ended agent benchmarks are intentionally excluded.

## Abstract

Long-horizon decision systems must compress interaction histories, but the usual objectives for compression—text reconstruction, semantic similarity, retrieval, or end-task reward—do not specify which historical distinctions must survive for future action selection. We study a narrower and testable objective: preserve the action-value differences induced by a declared policy class and candidate-action set. We introduce policy-class Q-sufficient memory, a rate-constrained discrete state trained from action-centered advantages estimated by counterfactual replay. Centering the target removes history-specific return offsets that do not change a decision, while the declared policy class makes the sufficiency claim explicit rather than universal. Across action-ambiguous binary language tasks, a two-code memory reached 100.00% held-out accuracy over five seeds, whereas a one-code control remained at chance (50.10%). With natural-language paraphrases and the same two-code bottleneck, Q-target training reached 98.47% ± 3.06%, compared with 51.11% ± 3.24% for ordered-text reconstruction and 29.81% ± 18.31% for scalar reward prediction. A four-state extension reached 100% across five seeds. When coupled to a frozen Qwen2.5-3B decision interface, learned compressed state preserved 150/150 decisions, matching full history and exceeding truncated history (68/150). In the public MiniGrid Memory environment, four-code policy-class memory reached 99.76% ± 0.54% on S13 Random, 100% on S11, and 98.04% ± 2.63% on the longer S17 Random, while recent-window, observation-reconstruction, policy-only, and one-code controls remained near chance. Counterfactual replay labels agreed with explicit policy classes on 2,000/2,000 audited training episodes, and all 7,500 paired evaluation trajectories matched exactly. These results support a bounded mechanism claim: under a declared policy class and replay-derived supervision, a small discrete memory can preserve delayed decision information more reliably than text-oriented compression. The evidence does not establish universal state sufficiency, automatic discovery from a single sparse reward, or parity with open-ended language-agent systems.

## Introduction

Interactive decision systems repeatedly face a rate constraint. A long history contains observations, actions, feedback, and linguistic detail, but a deployed policy often cannot retain the full sequence because of memory, latency, or context-window limits. The central question is therefore not whether a history can be summarized, but which distinctions in that history must remain available when a later action is selected.

Most history-compression objectives define fidelity in terms of the history itself. Reconstruction preserves the ability to reproduce text; retrieval preserves similarity to stored episodes; and end-task reward provides a single downstream signal. These objectives can be useful, but they do not distinguish two cases that are critical for sequential decisions. Fifty sentences may differ substantially while inducing the same action under the policy being deployed. Conversely, a one-token change such as “locked” versus “unlocked” may reverse the best action even when the two histories are semantically close in embedding space. A memory optimized for textual fidelity can therefore allocate capacity to nondiscriminative detail and discard a small decision boundary.

Value-aware representation learning offers a principled alternative. DeepMDP connects representation quality to reward and transition structure, and the value-equivalence principle formalizes when abstractions preserve Bellman-relevant quantities [1,2]. Prompt-compression work has also made rate and distortion explicit for language-model interfaces [11,12]. The unresolved methodological question is how to operationalize a decision-relative distortion for recurrent language histories without exposing privileged simulator state at deployment. In particular, a useful protocol must specify whose decisions are preserved, which actions are covered, how counterfactual values are estimated, and how replay cost and uncertainty are accounted for.

Here we introduce **policy-class Q-sufficient memory (Q-memory)**. Before training, we freeze a finite reference policy class and a candidate-action generator. For replayable histories, we estimate the value of each covered first action under each reference policy, center those values into action advantages, and train a rate-constrained discrete memory to decode the resulting vectors. The memory is therefore judged by whether histories mapped to the same code induce similar policy-relative action preferences, not by whether their text is similar. We separate explicit policy-class labels from labels reconstructed by counterfactual replay, and we evaluate both synthetic language tasks and a public partially observable control environment with a non-oracle action interface.

The paper makes four bounded contributions. First, it defines an implementable rate-distortion objective for recurrent memory that is relative to a declared policy class and candidate-action set. Second, it provides an action-ambiguous language benchmark in which action strings are symmetric, so memory—not action naming—must carry the delayed decision. Third, it demonstrates transfer from controlled language tasks to a public MiniGrid Memory environment, including capacity, data-efficiency, and strong-control comparisons. Fourth, it closes the explicit-label objection with a replay-Q audit: labels derived from counterfactual branch rewards and standard observations reproduce the explicit-label result exactly under matched seeds. The resulting claim is intentionally narrow: Q-memory preserves decision information under the declared supervision and environment assumptions. It is not a claim of universal state abstraction or automatic discovery from sparse terminal reward.

## Related Work

### Value-aware abstraction

Representation-learning methods in reinforcement learning have long distinguished task-relevant information from reconstruction fidelity. DeepMDP uses reward and transition structure to constrain latent representations [1], while the value-equivalence principle studies abstractions that preserve value-relevant model updates [2]. Our method adopts the same decision-relative perspective but changes the object being compressed: the input is a recurrent language history, the output is a low-rate discrete code, and the supervision is obtained by replaying candidate actions from history-consistent snapshots. The policy class is explicit, so the method does not claim that one code is sufficient for every future policy.

### Memory and history compression

Recent agent-memory systems organize workflows, linked memories, reasoning traces, or compact internal states [3–7]. Fixed-length history compression and rate-distortion analyses for prompt interfaces further motivate measuring the cost of retained information [11,12]. These approaches differ in what they preserve. Q-memory does not attempt to reconstruct the history or store a generally reusable episode. It preserves action-conditioned differences for a declared policy class and reports the candidate-action coverage and replay cost needed to construct that target.

### Position of this work

The contribution is an operational bridge between value-aware abstraction and sequential language-history compression. The novelty claim is bounded to the protocol and evidence reported here: counterfactual replay estimates action advantages, a discrete bottleneck is trained against those advantages, and the resulting code is tested under action-ambiguous language and a public partially observable control interface. The experiments do not establish a universal sufficient statistic, a benchmark-independent policy, or an end-to-end sparse-reward discovery procedure.

## Method

### Problem formulation

Let an environment produce a history before the current observation,

\[
h_t^-=(o_1,a_1,r_1,\ldots,a_{t-1},r_{t-1}),
\]

and let a recurrent encoder update a persistent state

\[
m_t=e_\phi(m_{t-1},o_{t-1},a_{t-1},r_{t-1})=E_\phi(h_t^-).
\]

The deployed memory is a discrete code with rate \(R(m_t)\), measured both as code bits and as serialized prompt tokens when the code is presented to a language model. We freeze a finite reference policy class \(\Pi_{\mathrm{ref}}=\{\pi_1,\ldots,\pi_K\}\) and a covered candidate set \(A_{\mathrm{cov}}(o_t)\). The candidate set is generated from information visible to the policy at deployment; privileged admissible-action lists are not used in the primary MiniGrid evaluation.

For each policy and covered action, replay estimates

\[
\hat Q^\pi(h_t^-,o_t,a)=\widehat{\mathbb E}[G_t\mid h_t^-,o_t,a_t=a,\pi\ \mathrm{thereafter}],
\]

under a fixed horizon and reward normalization. We use centered advantages

\[
\hat A^\pi(h_t^-,o_t,a)=\hat Q^\pi(h_t^-,o_t,a)-
\frac{1}{|A_{\mathrm{cov}}|}\sum_{b\in A_{\mathrm{cov}}}\hat Q^\pi(h_t^-,o_t,b).
\]

Centering removes history-specific return offsets that do not change the action ranking. The policy-class distortion between two histories sharing the current observation and candidate set is

\[
d_Q(h^-,h'^-\mid o)=\max_{\pi\in\Pi_{\mathrm{ref}},a\in A_{\mathrm{cov}}}
\left|A^\pi(h^-,o,a)-A^\pi(h'^-,o,a)\right|.
\]

A code is called \(\varepsilon\)-Q-sufficient for the declared class when same-code histories have distortion at most \(\varepsilon\) on the evaluated support. This is a conditional, policy-relative definition; it is not a universal Markov-state claim.

### Replay-Q label construction

For each training history, the environment is reset or replayed to the snapshot defined by the observed action prefix and current observation. Each covered candidate action is executed from the same snapshot with paired random seeds for variance reduction. The frozen continuation policy then supplies the remaining actions. Branch returns are aggregated into \(\hat Q\), confidence estimates are stored, and the centered vector \(\hat A\) becomes the training target. The memory encoder never receives simulator coordinates, hidden object identities, or future observations.

The protocol separates three sources of potential leakage. First, train/test splits are made by base episode, world seed, and language template before branching. Second, future observations are not included in the encoded prefix or label input. Third, the environment must be history-identifiable or provide a history-consistent belief-state sampling procedure; otherwise snapshot values can silently become privileged latent-state values. The experiments reported here use deterministic or replay-audited settings satisfying this requirement.

### Rate-constrained decoder

The decoder \(D_\psi(m,o,\pi,a)\) predicts the replay-estimated advantage from the persistent code and current observation. The training objective is

\[
\mathcal L(\phi,\psi)=
\mathbb E\left[w\,\rho\left(D_\psi(E_\phi(h^-),o,\pi,a)-\hat A^\pi(h^-,o,a)\right)\right]
 +\beta\,\mathbb E[R(E_\phi(h^-))],
\]

where \(\rho\) is a Huber loss and \(w\) is a clipped inverse-variance weight from replay uncertainty. The rate coefficient \(\beta\) traces a rate-distortion frontier. In the controlled experiments, the bottleneck is implemented as a small recurrent or positional-attention encoder followed by a discrete codebook. The architecture is deliberately conventional so that comparisons isolate the supervision target rather than additional retrieval or planning modules.

At deployment, action selection receives only the code, current observation, and the declared policy/action interface. Branch returns and simulator state are unavailable. If action selection is greedy with respect to the decoder, a uniform decoder error \(\varepsilon_M\) and replay error \(\varepsilon_Q\) imply a one-step covered-action regret bound of at most \(2(\varepsilon_M+\varepsilon_Q)\): the selected and optimal covered actions each incur at most the sum of the two errors. The bound is conditional on coverage, the declared policy class, and simultaneous—not merely pointwise—error control. Our empirical results test these conditions rather than claiming a universal theorem.

### Experimental controls

Every comparison uses the same current observation interface, candidate action names, train/test split, memory capacity where applicable, and evaluation seed set. Controls include one-code memory, recent windows, current-observation reconstruction, ordered-text reconstruction, scalar reward prediction, policy-only discrete training, continuous recurrent state, and learned full-history retrieval. The primary language benchmark makes action names symmetric; the primary MiniGrid evaluation exposes only the standard partial observation and fixed `left/right/forward` actions.

## Results

### Action ambiguity isolates decision-relevant memory

We first tested whether a discrete code can preserve a delayed decision when action text provides no shortcut. Each example contains the same two action names; the correct action is determined by an earlier cue and a later state update. The one-code model remained at chance across five seeds (50.10% ± 0.42%), whereas two-code memory reached 100.00% on 2,000 held-out episodes per seed. Four- and eight-code models reached 99.97% ± 0.06% and 99.96% ± 0.04%, respectively. The result establishes the diagnostic property of the benchmark: capacity sufficient to separate the two policy classes preserves the delayed decision, while one code cannot.

### Q supervision outperforms text and reward objectives

We next replaced symbolic histories with natural-language paraphrases, long distractor contexts, and held-out templates. A positional-attention encoder and two-code bottleneck were trained on 6,000 histories and evaluated on 2,000 held-out histories across five seeds. The Q/decision objective achieved 98.47% ± 3.06% test accuracy. Ordered-text reconstruction achieved 51.11% ± 3.24%, statistically indistinguishable from the symmetric-action baseline, while scalar reward prediction achieved 29.81% ± 18.31%. Because encoder, code capacity, templates, and data splits were held constant, the comparison isolates the supervision target. The result supports decision distortion as the relevant training signal for this task family; it does not show that text reconstruction is generally inferior for all downstream uses.

### Four-state scaling and frozen-backbone integration

An independent four-state language benchmark used 8,000 training and 3,000 test examples with four action classes and unseen templates. Direct four-way code supervision reached 100% train and test accuracy in all five seeds. A frozen Qwen2.5-3B decision interface then evaluated 150 held-out episodes under three conditions. Full history and learned compressed state both preserved 150/150 decisions, whereas truncating the history to the final segment preserved 68/150. This experiment tests whether a learned discrete state can serve as a compact interface to a fixed language model; it remains a controlled decision task rather than an open-ended agent benchmark.

### MiniGrid validates the mechanism with non-oracle actions

We evaluated the method on `MiniGrid-MemoryS13Random-v0`, where the policy receives only standard 7×7 partial observations and selects `left`, `right`, or `forward`. Five independent models used 2,000 training trajectories per seed and 500 shared held-out environment seeds. An independent expert audit solved all 500 held-out seeds, with balanced cue classes (246/254), ruling out environment failure or material class imbalance.

| Model | S13 Random | S11 | S17 Random |
|---|---:|---:|---:|
| Four-code policy-class Q-memory | **99.76% ± 0.54%** | **100.00% ± 0.00%** | **98.04% ± 2.63%** |
| Continuous recurrent state | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| Learned full-history retrieval | 85.12% ± 22.23% | 86.36% ± 21.38% | 85.88% ± 21.86% |
| One-code control | 51.80% ± 2.68% | 49.76% ± 0.36% | 51.80% ± 2.68% |
| Four-step recent window | 50.60% ± 3.29% | 49.92% ± 0.44% | 50.60% ± 3.29% |
| Current-observation reconstruction | 49.40% ± 3.29% | 50.08% ± 0.44% | 49.40% ± 3.29% |
| Four-code policy-only training | 50.60% ± 3.29% | 49.92% ± 0.44% | 50.60% ± 3.29% |

On S13, Q-memory exceeded the one-code control by 47.96 percentage points, the recent window by 49.16 points, observation reconstruction by 50.36 points, and policy-only training by 49.16 points under hierarchical paired bootstrap intervals reported in the statistical audit. It was within 0.24 points of the continuous recurrent state. The learned retrieval baseline was meaningful but seed-unstable: its mean was 85.12%, with one seed at 46.6%. Q-memory therefore demonstrates near-continuous-state performance while using a small discrete code; superiority over retrieval is not claimed as a universal property, especially on S17 where the paired interval crosses zero.

### Capacity and data efficiency

The capacity sweep showed that one code is insufficient and that three codes, although theoretically sufficient for the task classes, optimize unreliably (90.12% ± 13.60%). Four and eight codes both reached 99.76% ± 0.54%, with four-code models using only three codes in practice. In a nested data sweep, success increased from 46.24% ± 20.14% with 250 trajectories to 89.72% ± 22.54% with 500, 99.36% ± 0.61% with 1,000, and 99.04% ± 0.78% with 2,000. Reliable optimization therefore emerged between 500 and 1,000 demonstrations in this implementation.

### Replay-Q labels reproduce explicit-label results

The final audit replaced explicit pre-cue/key/ball class names with anonymous classes derived from counterfactual branch replay rewards and the successful terminal standard observation. Across 2,000 independent training seeds, replay-derived labels agreed with explicit policy-equivalence classes on 2,000/2,000 episodes; class counts were 1,002 and 998. Five replay-Q-trained four-code models achieved the same S13, S11, and S17 rates as the explicit-label models. Under matched training and evaluation seeds, all 7,500 paired outcomes, action sequences, and code sequences were identical.

This exact match closes a specific objection: the positive MiniGrid result is not dependent on the human-readable names of the policy classes. It does not remove all supervision. Expert trajectories, controlled branch evaluation, and temporal propagation of the replay-derived class remain part of the protocol.

### Sparse decision targets do not automatically discover the code

Two negative controls trained the MiniGrid bottleneck using only the sparse downstream branch action target. They reached 47.0% and 47.2%, near chance, despite valid environment interaction. This failure separates the mechanism from an end-to-end sparse-reward discovery claim. The successful result requires structured policy-equivalence supervision obtained through replay-Q labels; a single terminal outcome or downstream action target was insufficient in this implementation.

## Discussion

The central result is that a very small discrete state can preserve delayed decisions when it is trained against the action differences that the declared policy class actually uses. The evidence is strongest where the benchmark prevents action-text shortcuts. In the binary language task, identical action names force the code to carry the decision boundary. In MiniGrid, the evaluated policy sees only partial observations and a fixed non-oracle action interface, yet four-code memory approaches the continuous recurrent-state reference and transfers to a longer task.

The objective comparison identifies why this behavior is not a generic consequence of having a bottleneck. The same positional-attention encoder and two-code capacity failed under ordered-text reconstruction and scalar reward prediction. Reconstruction preserves distinctions in the observation sequence, while scalar reward removes the action-conditioned contrast. Centered action advantages retain precisely the information needed to compare candidate actions while discarding return offsets that do not affect the choice.

The replay-Q audit also clarifies the role of supervision. The method does not require semantic labels such as “key” or “ball” at learning time, but it does require a structured replay procedure that contrasts candidate actions and propagates the resulting policy-equivalence class. This is a practical middle ground between explicit privileged state labels and a single sparse terminal reward. It is also the source of a real cost: branch generation can be expensive, and the benefit of a compact online state must be evaluated against that offline cost.

Several limitations define the interpretation boundary. First, sufficiency is relative to the frozen policy class and covered actions. A changed policy can require distinctions that the code discarded. Second, replay estimates are only meaningful when snapshots are history-consistent and candidate coverage is adequate. Third, discrete optimization is data-sensitive: four-code memory was reliable in the reported MiniGrid regime, but smaller data sets and three-code configurations were unstable. Fourth, the strongest language and Qwen results use controlled delayed-decision tasks rather than open-ended planning. The paper therefore supports a mechanism and methods claim, not a general claim that compact memory improves every language agent.

The results suggest a concrete design rule for future systems: define the downstream decision family before compressing history, estimate action-conditioned value differences under that family, and report rate, coverage, replay uncertainty, and cost together. Extending the method to larger action spaces or changing policies will require held-out policy classes, better uncertainty calibration, and a candidate generator whose coverage can be audited. Those are natural next tests, but they are outside the evidence presented here.

## Methods Details

### Controlled language tasks

The binary action-ambiguous benchmark generated histories containing an early cue, intervening state updates, distractor descriptions, and a final decision between two symmetric action strings. Training and test histories were generated independently, and all reported rates average five training seeds with 2,000 held-out episodes per seed. The natural-language benchmark used four training templates and three held-out paraphrase templates; histories were split before model fitting. The four-state benchmark used four action classes, unseen language templates, 8,000 training examples, and 3,000 test examples.

### MiniGrid protocol

The environment was `minigrid==2.3.1`, trained on `MiniGrid-MemoryS13Random-v0` and evaluated zero-shot on S11 and S17 Random. Each model received the standard partial observation and fixed action set. Five independently initialized models were trained with 2,000 trajectories per seed and evaluated on the same 500 held-out environment seeds. Reported uncertainty is seed variation; model comparisons use hierarchical paired bootstrap resampling over training seeds and paired held-out tasks. The replay-Q audit used the same protocol and matched seeds as the explicit-label models.

### Statistical reporting

Episode-level success intervals are 95% Wilson intervals. Seed standard deviations describe optimization variation and are not treated as additional episode samples. Paired differences use the hierarchical bootstrap described in `STATISTICAL_AUDIT.md`. Exact trajectory identity is reported as a paired reproducibility result, not as 7,500 independent environments.

### Reproducibility and resources

The experiments were run with Python/PyTorch on a single RTX 4090 for MiniGrid and Qwen integration. The Qwen controlled interface used Qwen2.5-3B-Instruct; the external MiniGrid experiment did not require a large language model. Core scripts, raw episode logs, checkpoints, statistical summaries, replay-Q audits, and the terminology/claim boundary are stored in the project directory. The central analysis can be regenerated with `alfworld_final_analysis.py` for the excluded open-agent pilot, but that pilot is not part of this manuscript's evidence or claims.

## Data and Code Availability

The project archive contains the MiniGrid raw results, replay-Q audit, language-task results, model checkpoints, training scripts, and statistical summaries. Before submission, the authors should publish a stable repository with environment lockfiles, generated split manifests, seeds, exact checkpoint hashes, and scripts that regenerate every table. Synthetic task generators and MiniGrid episode logs can be released directly; any third-party model weights should be referenced through their official distribution mechanism.

## Author Contributions

To be completed by the authors.

## Competing Interests

The authors declare no competing interests.

## References

1. Gelada, C., Kumar, S., Buckman, J., Nachum, O. & Bellemare, M. G. DeepMDP: Learning Continuous Latent Space Models for Representation Learning. *ICML* (2019).
2. Grimm, C., Barreto, A., Singh, S. & Silver, D. The Value Equivalence Principle for Model-Based Reinforcement Learning. *NeurIPS* (2020).
3. Wang, Z. Z., Mao, J., Fried, D. & Neubig, G. Agent Workflow Memory. *ICML* (2025).
4. Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J. & Zhang, Y. A-Mem: Agentic Memory for LLM Agents. *NeurIPS* (2025).
5. Ouyang, S. et al. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. *ICLR* (2026).
6. Zhou, Z. et al. MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. *ICLR* (2026).
7. Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W. & Yu, D. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. *ICLR* (2025).
8. Cote, M.-A. et al. TextWorld: A Learning Environment for Text-based Games. arXiv:1806.11532 (2018).
9. Shridhar, M. et al. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. *ICLR* (2021).
10. Yao, S. et al. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. *NeurIPS* (2022).
11. Zhu, X., Tang, P., Liao, H. & Appalaraju, S. Turbocharging Web Automation: The Impact of Compressed History States. *Findings of ACL* (2025).
12. Nagle, A., Girish, A., Bondaschi, M., Gastpar, M., Makkuva, A. V. & Kim, H. Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models. *NeurIPS* (2024).

## Evidence and Claim Boundary Notes

This manuscript intentionally excludes the project's open-ended TextWorld and ALFWorld agent experiments. They are not used as positive or negative evidence here. The paper's claims stop at controlled delayed-decision tasks, a frozen-backbone decision interface, and the public MiniGrid Memory environment under the declared policy-class and replay-Q assumptions. The phrase “Q-sufficient” is always policy-relative and empirical unless the stated replay and decoder-error conditions are satisfied.
