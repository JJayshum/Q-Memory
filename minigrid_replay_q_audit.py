"""Compare reward/observation-derived replay classes with explicit state labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from minigrid_qmemory_benchmark import collect_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--start-seed", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--out", type=Path, default=Path("minigrid_replay_q_audit.json"))
    args = parser.parse_args()
    agreements = 0
    replay_counts = Counter()
    mismatches = []
    for seed in range(args.start_seed, args.start_seed + args.episodes):
        explicit = collect_episode(args.env, seed, "explicit")
        replay = collect_episode(args.env, seed, "replay_q")
        matches = explicit.cue == replay.cue and (explicit.memory_states == replay.memory_states).all()
        agreements += int(matches)
        replay_counts[replay.cue] += 1
        if not matches:
            mismatches.append(
                {"seed": seed, "explicit_class": explicit.cue, "replay_class": replay.cue}
            )
    output = {
        "environment": args.env,
        "start_seed": args.start_seed,
        "episodes": args.episodes,
        "agreements": agreements,
        "agreement_rate": agreements / args.episodes,
        "replay_class_counts": dict(sorted(replay_counts.items())),
        "mismatches": mismatches,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
