# Policy-Class Q-Sufficient Memory: CPU Pilot

This pilot implements Stage A from the proposal in a finite, deterministic text POMDP. It is an exact mechanism test, not a claim about language-agent benchmark performance.

Run: `python3 qmemory_pilot.py --examples 2000 --out pilot_results`

The runner uses only the Python standard library and writes `results.json` plus `rate_curve.csv`. It reports action agreement, one-step regret, Q-distortion, a bag-of-words reconstruction proxy, and paired-history diagnostics over 1--8 bit budgets.

`q_memory` is an idealized one-bit Q code, `reconstruction` is a fixed-rate lexical fingerprint, and `sliding_window` is a recency baseline. The pilot gate is that Q-memory is exact at one bit and has lower Q-distortion than reconstruction at that budget. This validates the diagnostic and data plumbing; it does not validate a learned neural encoder.

No GPU is required. It runs on a modern laptop CPU in seconds and uses well under 1 GB RAM. A later neural-memory stage should start with one NVIDIA GPU with 24--48 GB VRAM (or one 80 GB A100); request SSH access only after this CPU gate passes and the replay estimator is specified.

## Learned-memory pilot

`learned_memory_pilot.py` compares identical GRU encoders with discrete 1--3 bit bottlenecks. One model is trained to predict exact action advantages; the matched baseline is trained to reconstruct the full ordered token sequence. A code-usage penalty prevents the reconstruction baseline from silently collapsing below its declared budget. Deterministic argmax codes and a frozen majority probe measure how much decision information each learned code retains on held-out histories.

```bash
python learned_memory_pilot.py --seeds 5 --out learned_results
```

## External MiniGrid benchmark

`minigrid_qmemory_benchmark.py` evaluates the memory mechanism in the public MiniGrid Memory environment. Training uses expert demonstrations and explicit policy-equivalence labels; evaluation uses only standard partial observations and the fixed non-oracle action space.

```bash
python minigrid_qmemory_benchmark.py \
  --train-episodes 2000 --test-episodes 500 \
  --seeds 0 1 2 3 4 \
  --models q4 q1 continuous window retrieval reconstruction \
  --epochs 50 --out minigrid_qmemory_results
```

The confirmatory S13 result is 99.76% +/- 0.54% for four-code policy-class memory, versus 100.00% for continuous state and approximately 49--52% for one-code, recent-window, observation-reconstruction, and policy-only controls. See `MINIGRID_EXPERIMENT_CONCLUSION.md` and the machine-generated `minigrid_suite_summary/SUMMARY.md` for cross-size transfer, learned retrieval, capacity, data efficiency, statistical intervals, and negative controls.
