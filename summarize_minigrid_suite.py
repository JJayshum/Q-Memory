"""Combine all confirmatory MiniGrid experiments into one auditable summary."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from analyze_minigrid_results import hierarchical_paired_bootstrap


ROOT = Path(__file__).parent


def load(path: str):
    return json.loads((ROOT / path).read_text())


def runs(data, model):
    return sorted(
        [run for run in data["runs"] if run["model"] == model],
        key=lambda run: run["training_seed"],
    )


def env_result(run, primary, environment):
    return run if environment == primary else run["zero_shot"][environment]


def rates(model_runs, primary, environment):
    return [env_result(run, primary, environment)["success_rate"] for run in model_runs]


def describe(values):
    return {
        "rates": values,
        "mean": statistics.mean(values),
        "seed_sd": statistics.stdev(values),
    }


def main():
    main_data = load("minigrid_qmemory_results/results.json")
    capacity = load("minigrid_capacity_results/results.json")
    retrieval = load("minigrid_strong_controls/results.json")
    policy = load("minigrid_policy_control_final/results.json")
    nested = {
        size: load(f"minigrid_nested{size}/results.json") for size in (250, 500, 1000, 2000)
    }
    decision_attempts = [
        load("minigrid_decision_smoke/results.json"),
        load("minigrid_decision_smoke2/results.json"),
    ]
    primary = main_data["protocol"]["env"]
    environments = [primary, *main_data["protocol"]["eval_envs"]]
    sources = {
        "q4": (main_data, "q4"),
        "continuous": (main_data, "continuous"),
        "learned_retrieval": (retrieval, "learned_retrieval"),
        "q1": (main_data, "q1"),
        "window": (main_data, "window"),
        "observation_reconstruction": (main_data, "reconstruction"),
        "discrete_policy": (policy, "discrete_policy"),
    }
    q_runs = runs(main_data, "q4")
    output = {
        "protocol": main_data["protocol"],
        "environment_results": {},
        "paired_q4_effects": {},
        "capacity": {},
        "nested_data_efficiency": {},
        "negative_decision_target_attempts": [],
        "q4_codes_used": {},
    }

    for environment in environments:
        output["environment_results"][environment] = {}
        output["paired_q4_effects"][environment] = {}
        for label, (data, model) in sources.items():
            model_runs = runs(data, model)
            output["environment_results"][environment][label] = describe(
                rates(model_runs, primary, environment)
            )
            if label == "q4":
                continue
            differences = [
                q_rate - baseline_rate
                for q_rate, baseline_rate in zip(
                    rates(q_runs, primary, environment),
                    rates(model_runs, primary, environment),
                    strict=True,
                )
            ]
            output["paired_q4_effects"][environment][label] = {
                "per_seed_differences": differences,
                "mean_difference": statistics.mean(differences),
                "hierarchical_bootstrap_95": hierarchical_paired_bootstrap(
                    q_runs, model_runs, primary, environment
                ),
            }

    capacity_sources = {
        "q1": (main_data, "q1"),
        "q3": (capacity, "q3"),
        "q4": (main_data, "q4"),
        "q8": (capacity, "q8"),
    }
    for label, (data, model) in capacity_sources.items():
        output["capacity"][label] = describe(rates(runs(data, model), primary, primary))

    for size, data in nested.items():
        output["nested_data_efficiency"][str(size)] = describe(
            rates(runs(data, "q4"), primary, primary)
        )

    for attempt in decision_attempts:
        entry = attempt["summary"]["decision_memory"]
        output["negative_decision_target_attempts"].append(entry)

    for run in q_runs:
        output["q4_codes_used"][str(run["training_seed"])] = sorted(
            {code for row in run["rows"] for code in row["codes"]}
        )

    formal_files = [main_data, capacity, retrieval, policy, *nested.values()]
    output["formal_wall_seconds"] = sum(data["seconds"] for data in formal_files)
    out_dir = ROOT / "minigrid_suite_summary"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# MiniGrid External Benchmark Suite",
        "",
        "All values are recomputed from the raw episode logs. Rates are percentages; uncertainty after +/- is the sample SD across five independently trained models.",
        "",
    ]
    for environment in environments:
        lines.extend([f"## {environment}", "", "| Model | Per-seed success | Mean +/- SD | Q4 minus model (paired 95% CI) |", "|---|---:|---:|---:|"])
        for label in sources:
            entry = output["environment_results"][environment][label]
            seed_rates = ", ".join(f"{100 * value:.1f}" for value in entry["rates"])
            effect = "reference"
            if label != "q4":
                comparison = output["paired_q4_effects"][environment][label]
                low, high = comparison["hierarchical_bootstrap_95"]
                effect = f"{100 * comparison['mean_difference']:.2f} pp [{100 * low:.2f}, {100 * high:.2f}]"
            lines.append(
                f"| {label} | {seed_rates} | {100 * entry['mean']:.2f} +/- {100 * entry['seed_sd']:.2f} | {effect} |"
            )
        lines.append("")

    lines.extend(["## Capacity", "", "| Codes | Per-seed success | Mean +/- SD |", "|---:|---:|---:|"])
    for label, entry in output["capacity"].items():
        seed_rates = ", ".join(f"{100 * value:.1f}" for value in entry["rates"])
        lines.append(f"| {label[1:]} | {seed_rates} | {100 * entry['mean']:.2f} +/- {100 * entry['seed_sd']:.2f} |")

    lines.extend(["", "## Nested Data Efficiency", "", "| Training trajectories | Per-seed success | Mean +/- SD |", "|---:|---:|---:|"])
    for size, entry in output["nested_data_efficiency"].items():
        seed_rates = ", ".join(f"{100 * value:.1f}" for value in entry["rates"])
        lines.append(f"| {size} | {seed_rates} | {100 * entry['mean']:.2f} +/- {100 * entry['seed_sd']:.2f} |")

    lines.extend(
        [
            "",
            "## Audit Notes",
            "",
            "- Q4 uses three of four available codes in every seed: pre-cue, key cue, and ball cue.",
            "- The learned retrieval baseline can solve the task but is unstable across seeds; this is not evidence that retrieval is intrinsically incapable.",
            "- Two action-target-only optimization attempts stayed at 47.0% and 47.2%; the positive Q4 result therefore requires explicit policy-class/Q-equivalence supervision.",
            "- Evaluation uses only standard partial observations and the fixed {left, right, forward} action space. Environment internals are used only to generate training demonstrations and sufficient-class labels.",
            f"- Recorded suite runtime, including one repeated policy-only control: {output['formal_wall_seconds'] / 60:.1f} wall-clock minutes on one RTX 4090.",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
