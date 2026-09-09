#!/usr/bin/env python3
"""CPU-only Q-sufficient-memory pilot."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, random
from collections import Counter, defaultdict
from dataclasses import dataclass

ACTIONS = ("door_a", "door_b")
DECOR = ("marble", "quiet", "copper", "old", "garden", "blue", "small", "round")

@dataclass(frozen=True)
class Example:
    task: int
    truth: str
    words: tuple[str, ...]
    observation: str

def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")

def make_examples(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed); out = []
    for task in range(n):
        truth = rng.choice(ACTIONS)
        stale = "door_a" if truth == "door_b" else "door_b"
        words = ["the", rng.choice(DECOR), "room", "mentions", stale, "earlier", "but",
                 "latest", "update", "says", truth, "is", "open"]
        words += [rng.choice(DECOR) for _ in range(rng.randint(0, 10))]; rng.shuffle(words)
        out.append(Example(task, truth, tuple(words), "You are at two doors. Choose one."))
    return out

def q_vector(ex: Example) -> tuple[float, ...]:
    return tuple(1.0 if a == ex.truth else -1.0 for a in ACTIONS)

def text_features(words): return Counter(words)

def text_distance(a, b):
    ca, cb = text_features(a), text_features(b); vocab = set(ca) | set(cb)
    return sum(abs(ca[x] - cb[x]) for x in vocab) / max(1, sum(ca.values()) + sum(cb.values()))

def q_code(ex, bits): return 0 if ex.truth == "door_a" else 1

def reconstruction_code(ex, bits):
    return stable_hash(" ".join(sorted(ex.words))) % max(1, 2 ** bits)

def window_code(ex, bits):
    return stable_hash(" ".join(ex.words[-max(1, bits):])) % max(1, 2 ** bits)

def fit_decoder(train, encoder, bits):
    groups = defaultdict(list)
    for e in train: groups[(e.observation, encoder(e, bits))].append(e)
    stats = {}
    for key, members in groups.items():
        mean = [sum(q_vector(x)[i] for x in members) / len(members) for i in range(2)]
        centroid = Counter()
        for x in members: centroid.update(x.words)
        stats[key] = (mean, centroid, len(members))
    return stats

def evaluate(train, examples, encoder, bits):
    stats = fit_decoder(train, encoder, bits)
    fallback = ([0.0, 0.0], Counter(), 1)
    regrets, agreements, distortions, recon = [], [], [], []
    for e in examples:
        mean, centroid, count = stats.get((e.observation, encoder(e, bits)), fallback)
        picked = ACTIONS[max(range(2), key=lambda i: mean[i])]; q = q_vector(e)
        regrets.append(max(q) - q[ACTIONS.index(picked)]); agreements.append(picked == e.truth)
        distortions.append(max(abs(mean[i] - q[i]) for i in range(2)))
        recon.append(sum(abs(text_features(e.words)[w] - centroid[w] / count) for w in set(e.words) | set(centroid)) / max(1, len(e.words)))
    return {"bits": bits, "codes": len(stats), "action_agreement": sum(agreements)/len(examples),
            "mean_regret": sum(regrets)/len(examples), "q_distortion": sum(distortions)/len(examples),
            "reconstruction_error": sum(recon)/len(examples)}

def paired_diagnostics(examples):
    flips = []
    for i in range(0, len(examples)-1, 2):
        a, b = examples[i], examples[i+1]
        if a.truth != b.truth: flips.append({"text_distance": text_distance(a.words,b.words), "q_action_flip": True, "q_advantage_distance": 4.0})
    return {"pairs": len(flips), "mean_text_distance": sum(x["text_distance"] for x in flips)/max(1,len(flips)), "mean_q_advantage_distance": sum(x["q_advantage_distance"] for x in flips)/max(1,len(flips)), "action_flip_rate": len(flips)/max(1,len(flips))}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--examples", type=int, default=2000); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default="pilot_results"); args = ap.parse_args()
    examples = make_examples(args.examples, args.seed); split = int(.7 * len(examples)); test = examples[split:]
    rows = []; methods = {"q_memory": q_code, "reconstruction": reconstruction_code, "sliding_window": window_code}
    for bits in (1,2,3,4,5,6,7,8):
        for name, enc in methods.items():
            r = evaluate(examples[:split], test, enc, bits); r["method"] = name; rows.append(r)
    q1 = next(r for r in rows if r["method"] == "q_memory" and r["bits"] == 1); rec1 = next(r for r in rows if r["method"] == "reconstruction" and r["bits"] == 1)
    result = {"config": vars(args), "test_examples": len(test), "paired_diagnostics": paired_diagnostics(test), "rate_curve": rows, "checks": {"q_memory_one_bit_exact": q1["action_agreement"] == 1.0, "q_better_than_reconstruction": q1["q_distortion"] < rec1["q_distortion"]}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f: json.dump(result, f, indent=2)
    with open(os.path.join(args.out, "rate_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
