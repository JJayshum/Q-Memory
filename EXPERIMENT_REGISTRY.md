# Experiment Registry

All entries use deterministic generation seeds and report seed variation separately from episode-level uncertainty.

| ID | Script | Model / environment | Main setting | Status |
|---|---|---|---|---|
| A1 | `qmemory_pilot.py` | finite deterministic POMDP | exact Q, 1--8 bits, held-out split | complete |
| A2 | `learned_memory_pilot.py` | GRU discrete memory | Q vs ordered reconstruction, five seeds | complete |
| B1 | `stage_b_replay.py` | TextWorld 1.7.0 | 24 games, 16/8 train/test, replay/copy audit | complete |
| B2 | `stage_b_ranking.py` | TextWorld labels | grouped action ranking, 1/2/4/8 codes | complete |
| C1 | `ambiguous_memory_pilot.py` | action-ambiguous language | 2,000 test examples, five seeds | complete |
| C2 | `attention_memory_pilot.py` | natural-language paraphrases | Q/reconstruction/reward objectives, five seeds | complete |
| C3 | `multistate_memory_benchmark.py` | four-state language | 8,000/3,000 train/test, direct 4-code ablation | complete |
| D1 | `qwen_agent_benchmark.py` | Qwen2.5-3B | 100 full/truncated/compressed episodes | complete |
| D2 | `qwen_learned_multi.py` | Qwen2.5-3B + learned memory | 150 episodes, three memory seeds | complete |
| D3 | `qwen_textworld_agent.py` | Qwen2.5-3B + TextWorld | 20 games, full/truncated | complete |
| D4 | `qwen_textworld_rank.py` | Qwen2.5-3B + TextWorld | candidate log-likelihood ranking | complete |
| D5 | `qwen_textworld_learned_hint.py` | Qwen2.5-3B + learned hint | 20 games, full/hint/hint-only | complete |
| D6 | `qwen7_textworld_agent.py` | Qwen2.5-7B + TextWorld | 20-game backbone scale control | complete |
| E1 | `minigrid_qmemory_benchmark.py` | MiniGrid Memory S13 Random | non-oracle control, six matched conditions, five seeds | complete |
| E2 | `minigrid_qmemory_benchmark.py` | MiniGrid Memory S11/S17 | zero-shot length transfer, 500 tasks per seed | complete |
| E3 | `minigrid_qmemory_benchmark.py` | MiniGrid Memory | 1/3/4/8-code capacity sweep | complete |
| E4 | `minigrid_qmemory_benchmark.py` | MiniGrid Memory | nested 250/500/1,000/2,000-trajectory sweep | complete |
| E5 | `minigrid_qmemory_benchmark.py` | MiniGrid Memory | learned retrieval, policy-only, sparse decision-target controls | complete |
| E6 | `minigrid_oracle_audit.py` | MiniGrid Memory S11/S13/S17 | expert solvability and cue-balance audit | complete |
| E7 | `minigrid_qmemory_benchmark.py`, `minigrid_replay_q_audit.py` | MiniGrid Memory S11/S13/S17 | counterfactual replay-Q labels, five seeds, paired trajectory audit | complete |
| F1 | `alfworld_lora_train.py`, `alfworld_llm_agent.py` | ALFWorld `valid_unseen`, Qwen2.5-7B + QLoRA | 180-game training; 30 paired full/window/original-memory episodes | complete |
| F2 | `alfworld_llm_agent.py`, `alfworld_final_analysis.py` | ALFWorld `valid_unseen`, same fixed games | adapter-disabled memory-summary correction; 30 paired corrected-memory episodes and McNemar/bootstrap audit | complete |

## Reproducibility notes

- Qwen models were downloaded through ModelScope because the server is in mainland China.
- The 3B and 7B models were loaded in FP16 on a single RTX 4090.
- TextWorld replay uses privileged admissible commands and is explicitly labeled as a replay-estimator gate.
- Real TextWorld agent results are low absolute success baselines; they are not converted into memory claims.
- The strongest external-control result is MiniGrid policy-class memory; the strongest language result remains the action-ambiguous learned-memory condition, not the open TextWorld baseline.
- MiniGrid evaluation uses only partial observations and a fixed three-action interface; no admissible-action oracle is used at test time.
- MiniGrid's positive four-code result is reproduced with anonymous classes derived from counterfactual replay rewards and terminal standard observations. Sparse downstream action-target-only training failed and is retained as a negative control; end-to-end discovery from one sparse reward is not claimed.
- MiniGrid raw episode logs, checkpoints, nested data sweeps, and machine-generated statistical summaries are stored under `minigrid_*` directories.
- ALFWorld raw control/corrected results, partial snapshots, training report, evaluation log, and paired analysis are stored as `results.json`, `alfworld_lora_formal_corrected_results.json`, `alfworld_lora_formal_training_report.json`, `alfworld_lora_formal_corrected_eval.log`, and `alfworld_final_analysis.json`.
- ALFWorld corrected memory is a diagnostic external result: it improves over the original implementation but remains below full history and the recent window; no broad agent-performance claim is made.
