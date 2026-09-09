"""Create a seed-aware statistical summary for the MiniGrid benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def environment_result(run, primary_env, environment):
    return run if environment == primary_env else run["zero_shot"][environment]


def hierarchical_paired_bootstrap(q_runs, baseline_runs, primary_env, environment, samples=20_000):
    rng = np.random.default_rng(20260827)
    paired = []
    for q_run, baseline_run in zip(q_runs, baseline_runs, strict=True):
        q_rows = environment_result(q_run, primary_env, environment)["rows"]
        baseline_rows = environment_result(baseline_run, primary_env, environment)["rows"]
        assert [row["seed"] for row in q_rows] == [row["seed"] for row in baseline_rows]
        paired.append([int(q["success"]) - int(b["success"]) for q, b in zip(q_rows, baseline_rows)])
    paired_array = np.asarray(paired, dtype=np.int8)
    seed_count, task_count = paired_array.shape
    estimates = np.empty(samples, dtype=np.float64)
    batch_size = 250
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        count = stop - start
        chosen_runs = rng.integers(0, seed_count, size=(count, seed_count))
        batch_estimates = np.zeros(count, dtype=np.float64)
        for slot in range(seed_count):
            selected_rows = paired_array[chosen_runs[:, slot]]
            chosen_tasks = rng.integers(0, task_count, size=(count, task_count))
            sampled = np.take_along_axis(selected_rows, chosen_tasks, axis=1)
            batch_estimates += sampled.mean(axis=1) / seed_count
        estimates[start:stop] = batch_estimates
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=Path("minigrid_qmemory_results/results.json"))
    parser.add_argument("--out", type=Path, default=Path("minigrid_qmemory_results/summary.json"))
    parser.add_argument("--markdown", type=Path, default=Path("minigrid_qmemory_results/SUMMARY.md"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    primary_env = data["protocol"]["env"]
    environments = [primary_env, *data["protocol"]["eval_envs"]]
    models = data["protocol"]["models"]
    by_model = {
        model: sorted(
            [run for run in data["runs"] if run["model"] == model],
            key=lambda run: run["training_seed"],
        )
        for model in models
    }
    summary = {
        "protocol": data["protocol"],
        "wall_seconds": data["seconds"],
        "environments": {},
        "paired_q4_effects": {},
    }
    for environment in environments:
        env_summary = {}
        for model in models:
            rates = [environment_result(run, primary_env, environment)["success_rate"] for run in by_model[model]]
            counts = [environment_result(run, primary_env, environment)["successes"] for run in by_model[model]]
            totals = [environment_result(run, primary_env, environment)["episodes"] for run in by_model[model]]
            env_summary[model] = {
                "rates": rates,
                "mean": statistics.mean(rates),
                "seed_sd": statistics.stdev(rates),
                "per_seed_wilson_95": [wilson(k, n) for k, n in zip(counts, totals)],
                "failures": [n - k for k, n in zip(counts, totals)],
            }
        summary["environments"][environment] = env_summary

        q_runs = by_model["q4"]
        effects = {}
        for baseline in [model for model in models if model != "q4"]:
            baseline_runs = by_model[baseline]
            differences = []
            for q_run, baseline_run in zip(q_runs, baseline_runs, strict=True):
                q_rate = environment_result(q_run, primary_env, environment)["success_rate"]
                baseline_rate = environment_result(baseline_run, primary_env, environment)["success_rate"]
                differences.append(q_rate - baseline_rate)
            positive = sum(value > 0 for value in differences)
            negative = sum(value < 0 for value in differences)
            non_ties = positive + negative
            one_sided_sign_p = sum(math.comb(non_ties, k) for k in range(positive, non_ties + 1)) / (2**non_ties) if non_ties else 1.0
            effects[baseline] = {
                "per_seed_differences": differences,
                "mean_difference": statistics.mean(differences),
                "hierarchical_bootstrap_95": hierarchical_paired_bootstrap(
                    q_runs, baseline_runs, primary_env, environment
                ),
                "positive_seed_replicates": positive,
                "non_tied_seed_replicates": non_ties,
                "one_sided_sign_p": one_sided_sign_p,
            }
        summary["paired_q4_effects"][environment] = effects

    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# MiniGrid Q-Memory Result Summary",
        "",
        f"- Training: {data['protocol']['train_episodes']} expert trajectories per seed, {len(data['protocol']['seeds'])} seeds.",
        f"- Evaluation: {data['protocol']['test_episodes']} held-out environment seeds per condition.",
        f"- Runtime: {data['seconds'] / 60:.1f} wall-clock minutes on {data['protocol']['device']}.",
        "- Intervals below compare Q4 and each baseline with a hierarchical paired bootstrap over training seeds and test tasks.",
        "",
    ]
    for environment in environments:
        lines.extend([f"## {environment}", "", "| Model | Seed rates | Mean +/- seed SD |", "|---|---:|---:|"])
        for model in models:
            entry = summary["environments"][environment][model]
            rates = ", ".join(f"{100 * value:.1f}" for value in entry["rates"])
            lines.append(f"| {model} | {rates} | {100 * entry['mean']:.2f} +/- {100 * entry['seed_sd']:.2f}% |")
        lines.extend(["", "| Q4 comparison | Mean difference | Hierarchical paired 95% CI | Positive seeds |", "|---|---:|---:|---:|"])
        for baseline, effect in summary["paired_q4_effects"][environment].items():
            low, high = effect["hierarchical_bootstrap_95"]
            lines.append(
                f"| vs {baseline} | {100 * effect['mean_difference']:.2f} pp | "
                f"[{100 * low:.2f}, {100 * high:.2f}] pp | "
                f"{effect['positive_seed_replicates']}/{effect['non_tied_seed_replicates']} |"
            )
        lines.append("")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
