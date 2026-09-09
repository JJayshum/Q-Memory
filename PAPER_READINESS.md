# Publication Readiness Audit

## Supported claims

- Under a strict rate constraint, training a discrete memory on decision/Q targets preserves delayed decision information better than ordered-text reconstruction on the controlled benchmarks.
- In action-ambiguous binary and four-state tasks, one/two-bit memory reaches near-perfect held-out accuracy while a one-code or no-memory control is at chance.
- On natural-language paraphrase histories, a two-code positional-attention memory reaches 98.47% +/- 3.06% over five seeds, versus 51.11% +/- 3.24% for reconstruction.
- With a frozen Qwen2.5-3B backbone, a learned two-state summary preserved 150/150 held-out decisions across three memory seeds; truncated history achieved 68/150.
- TextWorld replay/reset/copy correctness is independently audited at 100%; native walkthrough solvability is 20/20.
- On public MiniGrid Memory with a fixed non-oracle action interface, explicit four-code policy-equivalence memory reaches 99.76% +/- 0.54% across five seeds and transfers at 98.04% +/- 2.63% to the longer S17 task.
- Replacing explicit class labels with classes derived from counterfactual replay rewards and terminal standard observations reproduces the same S13/S11/S17 rates and all 7,500 matched evaluation trajectories exactly; the label audit agrees on 2,000/2,000 training episodes.
- The MiniGrid result includes continuous-state, one-code, recent-window, learned-retrieval, observation-reconstruction, and policy-only controls, plus capacity and nested-data sweeps.

## Claims not supported

- Broad improvement on ALFWorld, WebShop, or other external LLM-agent benchmarks.
- Learned-memory improvement in open TextWorld planning: Qwen's full-history baseline solved only 2/20 games.
- General superiority over larger models, fine-tuned policies, or non-privileged candidate generators.
- Practical parity with full-history policies on ALFWorld: corrected memory improved from 6/30 to 12/30 after fixing adapter misuse, but full history remained 22/30 and the four-turn window 18/30 on the same paired games.
- End-to-end discovery of a Q-sufficient state from a single sparse terminal reward or downstream action target; both MiniGrid decision-target-only attempts remained near chance, while the successful replay-Q method uses structured counterfactual supervision.

## Required for a stronger CCF-A submission

1. For a broad LLM-agent claim, add an open language-agent benchmark whose full-history policy has a competitive non-floor success rate, then test learned memory without oracle action candidates.
2. If claiming discovery from terminal reward alone, add an end-to-end temporal credit-assignment method weaker than the completed structured replay-Q supervision.
3. Freeze a confirmatory protocol before any additional benchmark family; the completed MiniGrid suite followed exploratory debugging rather than prospective preregistration.
4. Publish scripts, exact model/environment versions, generated seeds, raw episode logs, and checkpoints in a stable repository.
5. Complete the manuscript-level novelty comparison, figures, tables, and claim-to-result audit.

The experimental artifact is now substantially stronger and sufficient for a scoped mechanism/proof-of-concept paper with a public partially observable control environment, including a transparent negative/diagnostic ALFWorld external pilot. It is not sufficient for a broad LLM-agent benchmark claim, practical memory parity, or a claim that Q-sufficient classes emerge automatically from sparse reward.
