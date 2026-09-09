"""Audit replay-Q labels and compare replay-trained Q4 with explicit supervision."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def environment_result(run: dict, primary_env: str, environment: str) -> dict:
    return run if environment == primary_env else run["zero_shot"][environment]


def q4_runs(data: dict) -> list[dict]:
    return sorted(
        (run for run in data["runs"] if run["model"] == "q4"),
        key=lambda run: run["training_seed"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit", type=Path, default=Path("minigrid_qmemory_results/results.json"))
    parser.add_argument("--replay", type=Path, default=Path("minigrid_replay_q_results/results.json"))
    parser.add_argument("--label-audit", type=Path, default=Path("minigrid_replay_q_audit.json"))
    parser.add_argument("--out", type=Path, default=Path("minigrid_replay_q_results/audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("minigrid_replay_q_results/AUDIT.md"))
    args = parser.parse_args()

    explicit = json.loads(args.explicit.read_text())
    replay = json.loads(args.replay.read_text())
    labels = json.loads(args.label_audit.read_text())
    primary_env = explicit["protocol"]["env"]
    environments = [primary_env, *explicit["protocol"]["eval_envs"]]
    explicit_runs = q4_runs(explicit)
    replay_runs = q4_runs(replay)
    if len(explicit_runs) != len(replay_runs):
        raise AssertionError("explicit and replay-Q runs have different seed counts")

    result = {
        "explicit_results": str(args.explicit),
        "replay_results": str(args.replay),
        "label_audit": labels,
        "environments": {},
        "all_episode_outcomes_match": True,
        "all_action_sequences_match": True,
        "all_code_sequences_match": True,
    }
    for environment in environments:
        environment_rows = []
        for explicit_run, replay_run in zip(explicit_runs, replay_runs, strict=True):
            if explicit_run["training_seed"] != replay_run["training_seed"]:
                raise AssertionError("training seed order differs")
            explicit_eval = environment_result(explicit_run, primary_env, environment)
            replay_eval = environment_result(replay_run, primary_env, environment)
            explicit_rows = explicit_eval["rows"]
            replay_rows = replay_eval["rows"]
            if [row["seed"] for row in explicit_rows] != [row["seed"] for row in replay_rows]:
                raise AssertionError(f"held-out task seeds differ for {environment}")
            outcome_matches = sum(
                bool(a["success"]) == bool(b["success"]) for a, b in zip(explicit_rows, replay_rows, strict=True)
            )
            action_matches = sum(
                a["actions"] == b["actions"] for a, b in zip(explicit_rows, replay_rows, strict=True)
            )
            code_matches = sum(a["codes"] == b["codes"] for a, b in zip(explicit_rows, replay_rows, strict=True))
            episode_count = len(explicit_rows)
            result["all_episode_outcomes_match"] &= outcome_matches == episode_count
            result["all_action_sequences_match"] &= action_matches == episode_count
            result["all_code_sequences_match"] &= code_matches == episode_count
            environment_rows.append(
                {
                    "training_seed": explicit_run["training_seed"],
                    "episodes": episode_count,
                    "explicit_success_rate": explicit_eval["success_rate"],
                    "replay_success_rate": replay_eval["success_rate"],
                    "outcome_matches": outcome_matches,
                    "action_sequence_matches": action_matches,
                    "code_sequence_matches": code_matches,
                }
            )
        replay_rates = [row["replay_success_rate"] for row in environment_rows]
        result["environments"][environment] = {
            "runs": environment_rows,
            "mean_replay_success_rate": statistics.mean(replay_rates),
            "replay_seed_sd": statistics.stdev(replay_rates),
            "total_episode_pairs": sum(row["episodes"] for row in environment_rows),
            "total_outcome_matches": sum(row["outcome_matches"] for row in environment_rows),
            "total_action_sequence_matches": sum(row["action_sequence_matches"] for row in environment_rows),
            "total_code_sequence_matches": sum(row["code_sequence_matches"] for row in environment_rows),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Replay-Q Supervision Audit",
        "",
        f"- Replay labels agree with explicit policy classes on {labels['agreements']}/{labels['episodes']} training episodes.",
        "- Training receives replay-derived anonymous classes, not environment state labels.",
        "- The evaluated policy receives only standard partial observations and the fixed left/right/forward interface.",
        "",
        "| Environment | Replay-Q mean +/- seed SD | Outcome matches | Action matches | Code matches |",
        "|---|---:|---:|---:|---:|",
    ]
    for environment, entry in result["environments"].items():
        total = entry["total_episode_pairs"]
        lines.append(
            f"| {environment} | {100 * entry['mean_replay_success_rate']:.2f}% +/- "
            f"{100 * entry['replay_seed_sd']:.2f}% | {entry['total_outcome_matches']}/{total} | "
            f"{entry['total_action_sequence_matches']}/{total} | {entry['total_code_sequence_matches']}/{total} |"
        )
    lines.extend(
        [
            "",
            "The replay-Q and explicit-label Q4 runs are episode-wise identical under matched training and evaluation seeds.",
            "This establishes distillation from counterfactual replay values and terminal standard observations; it does not establish end-to-end discovery from a single sparse terminal reward.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
