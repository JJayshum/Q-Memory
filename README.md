# Policy-Class Q-Sufficient Memory

This repository contains the mechanism experiments and ICASSP 2027 manuscript for policy-class Q-sufficient memory. The work studies whether a low-rate recurrent code preserves distinctions needed for decisions under a declared policy class. It intentionally does not claim universal state sufficiency or performance on open-ended agent benchmarks.

## ICASSP 2027 manuscript

- `POLICY_CLASS_Q_MEMORY_ICASSP2027.tex`: submission source.
- `shen.pdf`: upload-ready filename after the latest verified build.
- `ICASSP2027_ABSTRACT.txt`: web-form abstract.
- `ICASSP2027_SUBMISSION_CHECKLIST.md`: official format and portal checks.
- `AUTHOR_INPUT_REQUIRED.md`: author metadata, scientific decisions, and attestations still required before submission.

Compile with the official 2027 `spconf.sty` using a full TeX Live installation:

```bash
mkdir -p icassp2027_build
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=icassp2027_build POLICY_CLASS_Q_MEMORY_ICASSP2027.tex
```

The public repository contains scripts and reported result artifacts, but it does not yet contain every environment lockfile, split manifest, or model checkpoint needed for a complete archival reproduction.

## Stage A CPU pilot

The initial pilot implements Stage A in a finite, deterministic text POMDP. It is an exact mechanism test, not a claim about language-agent benchmark performance.

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
