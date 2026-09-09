# Pilot conclusion

Date: 7 August 2026

## Decision

**GO to the replay-estimator stage.** The controlled mechanism has not failed.

This decision is limited to the synthetic, history-identifiable setting with exact Q labels. It does not establish gains in TextWorld, ALFWorld, WebShop, or a deployed language agent.

## Audited protocol

- Exact pilot: 10,000 examples with a held-out 30% test split.
- Learned pilot: identical GRU encoders, discrete 1-, 2-, and 3-bit bottlenecks, 6,000 training and 2,000 held-out examples per seed, five seeds.
- Q-memory objective: Huber prediction of exact centered action advantages.
- Matched baseline: full ordered-token reconstruction.
- Baseline code-usage regularization prevents silent codebook collapse.
- Evaluation uses deterministic argmax codes, never evaluation-time Gumbel samples.
- A frozen train-derived majority probe measures decision information in each code.
- Negative control: Q labels permuted within the training set.

## Results

| Nominal bits | Q-memory probe accuracy | Reconstruction probe accuracy | Reconstruction empirical bits |
|---:|---:|---:|---:|
| 1 | 100.00% +/- 0.00% | 51.55% +/- 0.96% | 0.98 |
| 2 | 100.00% +/- 0.00% | 50.97% +/- 1.02% | 1.95 |
| 3 | 99.99% +/- 0.02% | 50.49% +/- 1.29% | 2.91 |

The reconstruction baseline improved token cross-entropy from 3.07 for the one-bit Q code to 2.58 at one bit, and to 2.45 at three bits. It therefore learned its assigned textual objective and spent essentially the full code budget, but did not preserve the decision boundary.

With one-bit Q labels randomly permuted, probe accuracy fell to 50.27% +/- 1.85%. The Q-memory advantage therefore depends on the correct decision labels rather than an evident test leak or architecture shortcut.

The exact pilot also passed: one-bit Q-memory achieved 100% action agreement and zero Q-distortion, while reconstruction and sliding-window baselines were near chance.

## Audit fixes

The final results exclude earlier runs affected by any of the following weaknesses:

1. Test-set statistics used to fit decoder buckets.
2. Quadratic reconstruction-centroid evaluation.
3. Stochastic Gumbel sampling during evaluation.
4. A bag-of-words baseline that could not represent ordering.
5. Reconstruction codebook collapse below the declared bit budget.

## Claim boundary and next gate

The pilot supports only this statement: under severe rate constraints in the controlled environment, training a recurrent discrete memory on exact action advantages preserves delayed decision information better than training the same encoder to reconstruct text.

The next experiment must replace exact labels with snapshot-replay estimates, report simultaneous uncertainty and branch cost, and compare against policy-logit, reward-predictive, and value-aware baselines. Failure at that stage can still invalidate the practical project even though the controlled mechanism passed.
