"""External MiniGrid Memory benchmark for policy-class Q-sufficient memory.

Training uses demonstrations from the environment's shortest-path state oracle. At
evaluation time every policy receives only MiniGrid's standard partial observation
and selects from the fixed, non-oracle action space {left, right, forward}.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import minigrid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LEFT, RIGHT, FORWARD = 0, 1, 2
ACTION_NAMES = ("left", "right", "forward")
minigrid.register_minigrid_envs()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str):
    return gym.make(env_id)


def expert_action(env, phase: str) -> tuple[int, str, bool]:
    """Return an expert action, next phase, and whether this is the branch decision."""
    base = env.unwrapped
    x, y = map(int, base.agent_pos)
    direction = int(base.agent_dir)
    mid = base.height // 2

    if phase == "seek":
        if x > 1:
            if direction != 2:
                return RIGHT, phase, False
            return FORWARD, phase, False
        if direction != 3:
            return RIGHT, phase, False
        return RIGHT, "corridor", False

    if phase == "corridor":
        junction_x = int(base.success_pos[0])
        if x < junction_x:
            if direction != 0:
                return LEFT, phase, False
            return FORWARD, phase, False
        target_north = int(base.success_pos[1]) < mid
        desired = 3 if target_north else 1
        if direction != desired:
            return (LEFT if target_north else RIGHT), "finish", True
        return FORWARD, "finish", False

    return FORWARD, phase, False


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    decisions: np.ndarray
    memory_states: np.ndarray
    cue: int


def replay_q_class(env_id: str, seed: int, action_prefix: list[int]) -> tuple[int, list[float]]:
    """Derive the remembered object class from branch replay rewards and observations."""
    q_values = []
    terminal_objects = []
    for candidate in (LEFT, RIGHT):
        branch = make_env(env_id)
        obs, _ = branch.reset(seed=seed)
        for action in action_prefix:
            obs, _, terminated, truncated, _ = branch.step(action)
            if terminated or truncated:
                raise RuntimeError("replay prefix terminated before the branch snapshot")
        obs, _, terminated, truncated, _ = branch.step(candidate)
        if terminated or truncated:
            raise RuntimeError("branch turn terminated before forward evaluation")
        obs, reward, terminated, truncated, _ = branch.step(FORWARD)
        if not terminated or truncated:
            raise RuntimeError("branch replay did not terminate as expected")
        object_ids = obs["image"][..., 0]
        visible = object_ids[(object_ids == 5) | (object_ids == 6)]
        if visible.size != 1:
            raise RuntimeError(f"expected one terminal object, found {visible.tolist()}")
        q_values.append(float(reward))
        terminal_objects.append(int(visible[0]))
        branch.close()
    positive = [index for index, value in enumerate(q_values) if value > 0]
    if len(positive) != 1 or terminal_objects[0] == terminal_objects[1]:
        raise RuntimeError(f"invalid replay labels: q={q_values}, objects={terminal_objects}")
    successful_object = terminal_objects[positive[0]]
    return {5: 1, 6: 2}[successful_object], q_values


def collect_episode(env_id: str, seed: int, label_source: str = "explicit") -> Episode:
    env = make_env(env_id)
    obs, _ = env.reset(seed=seed)
    phase = "seek"
    observations, actions, decisions, memory_states = [], [], [], []
    if label_source not in {"explicit", "replay_q"}:
        raise ValueError(label_source)
    cue_state = None
    if label_source == "explicit":
        cue_object = env.unwrapped.grid.get(1, env.unwrapped.height // 2 - 1).type
        cue_state = {"key": 1, "ball": 2}[cue_object]
    memory_state = 0
    cue_index = None
    replay_class = None
    terminated = truncated = False
    final_reward = 0.0
    while not (terminated or truncated):
        base = env.unwrapped
        if phase == "seek" and int(base.agent_pos[0]) == 1 and int(base.agent_dir) == 3:
            cue_index = len(observations)
            if label_source == "explicit":
                memory_state = int(cue_state)
        action, phase, decision = expert_action(env, phase)
        if decision and label_source == "replay_q":
            replay_class, _ = replay_q_class(env_id, seed, actions)
        observations.append(obs["image"].copy())
        actions.append(action)
        decisions.append(decision)
        memory_states.append(memory_state)
        obs, final_reward, terminated, truncated, _ = env.step(action)
        if len(actions) > env.unwrapped.max_steps:
            raise RuntimeError("expert trajectory exceeded max_steps")
    if final_reward <= 0:
        raise RuntimeError(f"expert failed for environment seed {seed}")
    if label_source == "replay_q":
        if cue_index is None or replay_class is None:
            raise RuntimeError("replay-derived episode is missing cue or branch labels")
        memory_states[cue_index:] = [replay_class] * (len(memory_states) - cue_index)
        cue_state = replay_class
    env.close()
    return Episode(
        observations=np.asarray(observations, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        decisions=np.asarray(decisions, dtype=np.bool_),
        memory_states=np.asarray(memory_states, dtype=np.int64),
        cue=int(cue_state),
    )


def collect_dataset(env_id: str, seeds: list[int], label_source: str = "explicit") -> list[Episode]:
    return [collect_episode(env_id, seed, label_source) for seed in seeds]


def batch_episodes(episodes: list[Episode], device: torch.device):
    lengths = torch.tensor([len(ep.actions) for ep in episodes], device=device)
    max_len = int(lengths.max())
    shape = episodes[0].observations.shape[1:]
    obs = torch.zeros((len(episodes), max_len, *shape), dtype=torch.long, device=device)
    actions = torch.zeros((len(episodes), max_len), dtype=torch.long, device=device)
    decisions = torch.zeros((len(episodes), max_len), dtype=torch.bool, device=device)
    memory_states = torch.zeros((len(episodes), max_len), dtype=torch.long, device=device)
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    cues = torch.tensor([ep.cue for ep in episodes], dtype=torch.long, device=device)
    for i, ep in enumerate(episodes):
        n = len(ep.actions)
        obs[i, :n] = torch.from_numpy(ep.observations.astype(np.int64)).to(device)
        actions[i, :n] = torch.from_numpy(ep.actions).to(device)
        decisions[i, :n] = torch.from_numpy(ep.decisions).to(device)
        memory_states[i, :n] = torch.from_numpy(ep.memory_states).to(device)
    return obs, actions, mask, decisions, memory_states, cues


class ObservationEncoder(nn.Module):
    def __init__(self, output_dim: int = 96):
        super().__init__()
        self.object_embedding = nn.Embedding(16, 12)
        self.color_embedding = nn.Embedding(8, 6)
        self.state_embedding = nn.Embedding(4, 4)
        self.net = nn.Sequential(
            nn.Linear(7 * 7 * 22, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obj = self.object_embedding(obs[..., 0])
        color = self.color_embedding(obs[..., 1])
        state = self.state_embedding(obs[..., 2])
        cells = torch.cat((obj, color, state), dim=-1)
        return self.net(cells.flatten(start_dim=-3))


class MemoryPolicy(nn.Module):
    def __init__(self, kind: str, codes: int = 4, window: int = 4):
        super().__init__()
        self.kind = kind
        self.codes = codes
        self.window = window
        self.encoder = ObservationEncoder()
        self.gru = nn.GRU(96, 128, batch_first=True)
        self.code_head = nn.Linear(128, codes)
        self.state_head = nn.Linear(128, 3)
        self.code_embedding = nn.Parameter(torch.randn(codes, 32) * 0.05)
        if kind == "learned_retrieval":
            self.retrieve_query = nn.Linear(96, 64)
            self.retrieve_key = nn.Linear(96, 64)
            self.retrieve_value = nn.Linear(96, 96)
        else:
            self.retrieve_query = None
            self.retrieve_key = None
            self.retrieve_value = None
        context_dim = 128 if kind == "continuous" else 32 if kind in {"q", "decision", "policy", "reconstruction"} else 96
        if kind == "window":
            context_dim = 128
        if kind == "retrieval":
            context_dim = 96
        self.policy = nn.Sequential(nn.Linear(96 + context_dim, 128), nn.ReLU(), nn.Linear(128, 3))
        self.reconstruction = nn.Linear(32, 7 * 7 * 16)

    def sequence_features(self, obs: torch.Tensor, temperature: float = 1.0, hard: bool = False):
        features = self.encoder(obs)
        hidden, _ = self.gru(features)
        code_logits = self.code_head(hidden)
        probs = F.softmax(code_logits / temperature, dim=-1)
        if hard:
            index = probs.argmax(dim=-1)
            one_hot = F.one_hot(index, self.codes).float()
            probs = one_hot - probs.detach() + probs
        codes = probs @ self.code_embedding
        return features, hidden, code_logits, codes

    def forward(self, obs: torch.Tensor, temperature: float = 1.0, hard: bool = False):
        features, hidden, code_logits, codes = self.sequence_features(obs, temperature, hard)
        if self.kind in {"q", "decision", "policy", "reconstruction"}:
            context = codes
        elif self.kind == "continuous":
            context = hidden
        elif self.kind == "window":
            chunks = []
            for t in range(features.shape[1]):
                start = max(0, t + 1 - self.window)
                local, _ = self.gru(features[:, start : t + 1])
                chunks.append(local[:, -1])
            context = torch.stack(chunks, dim=1)
        elif self.kind == "retrieval":
            # Causal content retrieval outside the recent window. Object IDs 5/6 are
            # generic key/ball salience signals, not target or action information.
            salient = ((obs[..., 0] == 5) | (obs[..., 0] == 6)).sum(dim=(-1, -2))
            retrieved = []
            for t in range(features.shape[1]):
                end = max(1, t + 1 - self.window)
                index = salient[:, :end].argmax(dim=1)
                retrieved.append(features[torch.arange(features.shape[0], device=obs.device), index])
            context = torch.stack(retrieved, dim=1)
        elif self.kind == "learned_retrieval":
            assert self.retrieve_query is not None
            assert self.retrieve_key is not None
            assert self.retrieve_value is not None
            queries = self.retrieve_query(features)
            keys = self.retrieve_key(features)
            values = self.retrieve_value(features)
            scores = torch.einsum("btd,bsd->bts", queries, keys) / math.sqrt(keys.shape[-1])
            times = torch.arange(features.shape[1], device=obs.device)
            allowed = times[None, :] <= (times[:, None] - self.window)
            allowed[:, 0] = True
            weights = F.softmax(scores.masked_fill(~allowed[None], -1e9), dim=-1)
            context = torch.einsum("bts,bsd->btd", weights, values)
        elif self.kind == "none":
            context = features
        else:
            raise ValueError(self.kind)
        logits = self.policy(torch.cat((features, context), dim=-1))
        return logits, code_logits, codes


def masked_policy_loss(logits, actions, mask, decisions):
    loss = F.cross_entropy(logits.transpose(1, 2), actions, reduction="none")
    weights = torch.where(decisions, torch.full_like(loss, 8.0), torch.ones_like(loss))
    return (loss * weights * mask).sum() / (weights * mask).sum()


def code_regularizer(code_logits, mask):
    probs = F.softmax(code_logits, dim=-1)
    valid = probs[mask]
    mean = valid.mean(dim=0)
    balance = (mean * (mean.clamp_min(1e-8).log())).sum() + math.log(probs.shape[-1])
    confidence = -(valid * valid.clamp_min(1e-8).log()).sum(dim=-1).mean()
    return balance + 0.02 * confidence


def train_policy(model, train, device, epochs, batch_size, lr, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = random.Random(seed)
    model.train()
    history = []
    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            obs, actions, mask, decisions, memory_states, _ = batch_episodes(batch, device)
            temperature = max(0.35, 1.25 * (0.92**epoch))
            logits, code_logits, codes = model(obs, temperature, hard=epoch >= epochs // 2)
            loss = masked_policy_loss(logits, actions, mask, decisions)
            if model.kind == "q" and model.codes >= 3:
                state_loss = F.cross_entropy(code_logits[mask], memory_states[mask])
                loss = loss + 0.75 * state_loss + 0.01 * code_regularizer(code_logits, mask)
            elif model.kind == "decision":
                # Class 0 is reserved for the pre-decision phase; left/right expert
                # decisions map directly to policy classes 1/2 without state labels.
                decision_targets = 1 + actions[decisions]
                decision_loss = F.cross_entropy(code_logits[decisions], decision_targets)
                loss = loss + 2.0 * decision_loss + 0.01 * code_regularizer(code_logits, mask)
            elif model.kind == "continuous":
                _, hidden, _, _ = model.sequence_features(obs)
                state_loss = F.cross_entropy(model.state_head(hidden)[mask], memory_states[mask])
                loss = loss + 0.75 * state_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss) * len(batch)
        history.append(total / len(train))
    return history


def train_reconstruction(model, train, device, epochs, batch_size, lr, seed):
    pretrain_params = list(model.encoder.parameters()) + list(model.gru.parameters())
    pretrain_params += list(model.code_head.parameters()) + [model.code_embedding]
    pretrain_params += list(model.reconstruction.parameters())
    optimizer = torch.optim.AdamW(pretrain_params, lr=lr, weight_decay=1e-5)
    rng = random.Random(seed)
    model.train()
    pretrain_history = []
    for epoch in range(epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            obs, _, mask, _, _, _ = batch_episodes(batch, device)
            _, _, code_logits, codes = model.sequence_features(obs, max(0.35, 1.25 * 0.92**epoch), epoch >= epochs // 2)
            rec = model.reconstruction(codes).view(*codes.shape[:2], 7, 7, 16)
            target = obs[..., 0]
            loss_map = F.cross_entropy(rec.permute(0, 4, 1, 2, 3), target, reduction="none")
            loss = (loss_map * mask[:, :, None, None]).sum() / (mask.sum() * 49)
            loss = loss + 0.03 * code_regularizer(code_logits, mask)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_params, 2.0)
            optimizer.step()
            total += float(loss) * len(batch)
        pretrain_history.append(total / len(train))

    for module in (model.encoder, model.gru, model.code_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model.code_embedding.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=lr)
    probe_history = []
    for epoch in range(max(5, epochs // 2)):
        order = list(range(len(train)))
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            obs, actions, mask, decisions, _, _ = batch_episodes(batch, device)
            with torch.no_grad():
                features, _, _, codes = model.sequence_features(obs, hard=True)
            logits = model.policy(torch.cat((features, codes), dim=-1))
            loss = masked_policy_loss(logits, actions, mask, decisions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        probe_history.append(total / len(train))
    return {"pretrain": pretrain_history, "probe": probe_history}


@torch.no_grad()
def act(model, observations, device):
    obs = torch.from_numpy(np.asarray(observations, dtype=np.int64))[None].to(device)
    logits, code_logits, _ = model(obs, hard=True)
    return int(logits[0, -1].argmax()), int(code_logits[0, -1].argmax())


@torch.no_grad()
def evaluate(model, env_id, seeds, device):
    model.eval()
    rows = []
    for seed in seeds:
        env = make_env(env_id)
        obs, _ = env.reset(seed=seed)
        observations, actions, codes = [], [], []
        terminated = truncated = False
        reward = 0.0
        max_eval_steps = 4 * env.unwrapped.width
        while not (terminated or truncated) and len(actions) < max_eval_steps:
            observations.append(obs["image"].copy())
            if model.kind == "window":
                model_obs = observations[-model.window :]
            elif model.kind == "none":
                model_obs = observations[-1:]
            else:
                model_obs = observations
            action, code = act(model, model_obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            codes.append(code)
        rows.append(
            {
                "seed": seed,
                "success": bool(reward > 0),
                "reward": float(reward),
                "steps": len(actions),
                "actions": [ACTION_NAMES[a] for a in actions],
                "codes": codes,
            }
        )
        env.close()
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "rows": rows,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - radius, center + radius]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-MemoryS13Random-v0")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--label-source", choices=["explicit", "replay_q"], default="explicit")
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument(
        "--train-seed-stride",
        type=int,
        default=0,
        help="Nonzero stride makes per-seed datasets nested across training-set sizes.",
    )
    parser.add_argument(
        "--eval-envs",
        nargs="*",
        default=["MiniGrid-MemoryS11-v0", "MiniGrid-MemoryS17Random-v0"],
        help="Additional zero-shot environments evaluated after training on --env.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=["q4", "q1", "continuous", "window", "retrieval", "reconstruction"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("minigrid_qmemory_results"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs = []

    for training_seed in args.seeds:
        stride = args.train_seed_stride or args.train_episodes
        train_start = 100_000 + training_seed * stride
        train_seeds = list(range(train_start, train_start + args.train_episodes))
        test_seeds = list(range(900_000, 900_000 + args.test_episodes))
        train = collect_dataset(args.env, train_seeds, args.label_source)
        for model_name in args.models:
            seed_everything(training_seed)
            if model_name.startswith("q"):
                kind, codes = "q", int(model_name[1:])
            elif model_name == "discrete_policy":
                kind, codes = "policy", 4
            elif model_name == "decision_memory":
                kind, codes = "decision", 4
            elif model_name == "reconstruction":
                kind, codes = model_name, 4
            else:
                kind, codes = model_name, 1
            model = MemoryPolicy(kind, codes=codes).to(device)
            if kind == "reconstruction":
                losses = train_reconstruction(model, train, device, args.epochs, args.batch_size, args.lr, training_seed)
            else:
                losses = train_policy(model, train, device, args.epochs, args.batch_size, args.lr, training_seed)
            result = evaluate(model, args.env, test_seeds, device)
            result["environment"] = args.env
            result["zero_shot"] = {
                eval_env: evaluate(model, eval_env, test_seeds, device) for eval_env in args.eval_envs
            }
            result.update(
                {
                    "training_seed": training_seed,
                    "model": model_name,
                    "losses": losses,
                    "wilson_95": wilson(result["successes"], result["episodes"]),
                }
            )
            runs.append(result)
            print(training_seed, model_name, result["success_rate"], result["mean_reward"], flush=True)
            torch.save(model.state_dict(), args.out / f"{model_name}_seed{training_seed}.pt")
            with (args.out / "results.partial.json").open("w") as handle:
                json.dump({"runs": runs}, handle, indent=2)

    summary = {}
    for model_name in args.models:
        values = [run["success_rate"] for run in runs if run["model"] == model_name]
        summary[model_name] = {
            "mean_success_rate": float(np.mean(values)),
            "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "rates": values,
        }
    output = {
        "protocol": vars(args) | {"out": str(args.out), "device": str(device)},
        "summary": summary,
        "runs": runs,
        "seconds": time.time() - started,
    }
    with (args.out / "results.json").open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
