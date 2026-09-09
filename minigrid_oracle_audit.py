"""Audit expert solvability and cue balance for MiniGrid evaluation seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from minigrid_qmemory_benchmark import collect_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="+",
        default=[
            "MiniGrid-MemoryS13Random-v0",
            "MiniGrid-MemoryS11-v0",
            "MiniGrid-MemoryS17Random-v0",
        ],
    )
    parser.add_argument("--start-seed", type=int, default=900_000)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("minigrid_oracle_audit.json"))
    args = parser.parse_args()
    output = {"start_seed": args.start_seed, "episodes": args.episodes, "environments": {}}
    for environment in args.envs:
        episodes = [
            collect_episode(environment, seed)
            for seed in range(args.start_seed, args.start_seed + args.episodes)
        ]
        lengths = [len(episode.actions) for episode in episodes]
        output["environments"][environment] = {
            "expert_successes": len(episodes),
            "cue_counts": dict(sorted(Counter(episode.cue for episode in episodes).items())),
            "mean_expert_steps": statistics.mean(lengths),
            "min_expert_steps": min(lengths),
            "max_expert_steps": max(lengths),
        }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
