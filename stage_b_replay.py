#!/usr/bin/env python3
"""Stage B TextWorld replay audit and counterfactual branch estimator.

This stage deliberately uses TextWorld's admissible-action list as privileged
candidate coverage. It is a replay-estimator gate and is not the confirmatory
frozen-generator result promised by the proposal.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, statistics, time
from pathlib import Path

import textworld
from textworld import EnvInfos
from textworld.generator import GameOptions, compile_game, make_game

INFOS = EnvInfos(description=True, feedback=True, score=True, won=True, lost=True,
                 moves=True, last_action=True, last_command=True,
                 admissible_commands=True, possible_commands=True, game=True)

def sig(obs, reward=None, done=None):
    return {"feedback": obs.get("feedback", ""), "description": obs.get("description", ""),
            "score": obs.get("score", 0), "moves": obs.get("moves", 0),
            "won": obs.get("won", False), "lost": obs.get("lost", False),
            "reward": reward, "done": done}

def make_game_file(root: Path, seed: int, rooms: int, objects: int, quest_length: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"game_{seed}.z8"
    opt = GameOptions(); opt.nb_rooms = rooms; opt.nb_objects = objects
    opt.quest_length = quest_length; opt.seeds = seed; opt.path = str(path)
    compile_game(make_game(opt), opt)
    return path

def run(env, command):
    out, reward, done = env.step(command)
    return out, float(reward), bool(done)

def audit_game(path: Path, seed: int, max_steps: int = 12) -> dict:
    a = textworld.start(str(path), INFOS); b = textworld.start(str(path), INFOS)
    sa, sb = a.reset(), b.reset(); reset_match = sig(sa) == sig(sb)
    commands = []
    cur = sa
    for _ in range(max_steps):
        acts = sorted(cur.get("admissible_commands", []))
        if not acts or cur.get("won") or cur.get("lost"): break
        cmd = acts[(seed + len(commands)) % len(acts)]; commands.append(cmd)
        cur, _, done = run(a, cmd); run(b, cmd)
        if done: break
    # Replay the same sequence from fresh resets and compare every transition.
    c = textworld.start(str(path), INFOS); d = textworld.start(str(path), INFOS)
    c.reset(); d.reset(); replay_matches = []
    for cmd in commands:
        oc, rc, dc = run(c, cmd); od, rd, dd = run(d, cmd)
        replay_matches.append(sig(oc, rc, dc) == sig(od, rd, dd))
    # Snapshot copy audit: copy at the midpoint, then execute the same branch twice.
    e = textworld.start(str(path), INFOS); state = e.reset(); prefix=[]
    for _ in range(max(1, len(commands)//2)):
        acts=sorted(state.get("admissible_commands", []))
        if not acts or state.get("won") or state.get("lost"): break
        cmd=acts[0]; prefix.append(cmd); state,_,done=run(e,cmd)
        if done: break
    clone=e.copy(); branch=sorted(state.get("admissible_commands", []))[:3]
    copy_matches=[]
    for cmd in branch:
        e1=e.copy(); e2=clone.copy(); o1,r1,d1=run(e1,cmd); o2,r2,d2=run(e2,cmd)
        copy_matches.append(sig(o1,r1,d1)==sig(o2,r2,d2))
    return {"game": path.name, "seed": seed, "reset_match": reset_match,
            "replay_transition_match_rate": sum(replay_matches)/max(1,len(replay_matches)),
            "replay_transitions": len(replay_matches), "copy_branch_match_rate": sum(copy_matches)/max(1,len(copy_matches)),
            "copy_branches": len(copy_matches), "prefix_length": len(prefix)}

def _admissible(env):
    acts = sorted(getattr(env, "_last_state", {}).get("admissible_commands", []))
    if not acts:
        look, _, _ = run(env, "look")
        env._last_state = look
        acts = sorted(look.get("admissible_commands", []))
    return acts

def search_success(env, horizon: int, max_branch: int = 8) -> tuple[float, int, bool]:
    """Bounded DFS oracle used only to make reachable terminal labels."""
    acts = _admissible(env)[:max_branch]
    if not acts or horizon <= 0:
        return 0.0, 0, False
    best = (-float("inf"), 0, False)
    for cmd in acts:
        child = env.copy(); obs, reward, done = run(child, cmd)
        total, used, won = float(reward), 1, bool(obs.get("won", False))
        if not done:
            child._last_state = obs
            tail, tail_used, tail_won = search_success(child, horizon - 1, max_branch)
            total += tail; used += tail_used; won = tail_won
        if won:
            return total, used, True
        candidate = (total, used, won)
        if candidate[2] and not best[2] or (candidate[2] == best[2] and candidate[0] > best[0]):
            best = candidate
    return (0.0, 0, False) if best[0] == -float("inf") else best

def rollout_walkthrough(env, commands: list[str], horizon: int) -> tuple[float, int, bool]:
    total = 0.0; steps = 0
    for command in commands[:horizon]:
        obs, reward, done = run(env, command); total += reward; steps += 1
        env._last_state = obs
        if done:
            return total, steps, bool(obs.get("won", False))
    return total, steps, bool(getattr(env, "_last_state", {}).get("won", False))

def rollout_policy(env, policy: str, rng: random.Random, horizon: int) -> tuple[float, int, bool]:
    if policy == "oracle_success":
        return search_success(env, horizon)
    total = 0.0; steps = 0; done = False
    for _ in range(horizon):
        state = env.state if hasattr(env, "state") else None
        # TextWorld returns the current observation on each step; admissible
        # commands are exposed by the wrapper's last state.
        acts = sorted(getattr(env, "_last_state", {}).get("admissible_commands", []))
        if not acts:
            # Recover state information without changing the world.
            look, _, _ = run(env, "look"); acts = sorted(look.get("admissible_commands", []))
        if not acts: break
        cmd = acts[0] if policy == "lexicographic" else acts[rng.randrange(len(acts))]
        obs, reward, done = run(env, cmd); total += reward; steps += 1
        env._last_state = obs
        if done: break
    return total, steps, done

def branch_values(path: Path, snapshots: int, horizon: int, rollouts: int, seed: int, max_actions: int) -> tuple[list[dict], dict]:
    env = textworld.start(str(path), INFOS); state = env.reset(); env._last_state = state; env._prefix = []
    try:
        env._walkthrough = json.loads(path.with_suffix(".walkthrough").read_text())
    except FileNotFoundError:
        env._walkthrough = []
    rows=[]; total_steps=0; branches=0
    rng = random.Random(seed)
    for t in range(snapshots):
        actions = sorted(state.get("admissible_commands", []))[:max_actions]
        if len(actions) < 2 or state.get("won") or state.get("lost"): break
        # Each row is a snapshot plus paired continuation estimates for two
        # frozen reference policies. The environment snapshot is cloned before
        # every first action; future observations never enter the label input.
        q_by_policy={}
        walkthrough = getattr(env, "_walkthrough", [])
        prefix = getattr(env, "_prefix", [])
        for policy in ("oracle_success", "random_uniform"):
            vals={}
            for ai, action in enumerate(actions):
                samples=[]
                for j in range(rollouts):
                    clone=env.copy(); first, reward, done=run(clone, action); total=reward; used=1
                    if not done:
                        clone._last_state=first
                        if policy == "oracle_success":
                            if prefix + [action] == walkthrough[:len(prefix)+1]:
                                cont, cont_steps, _ = rollout_walkthrough(clone, walkthrough[len(prefix)+1:], horizon)
                            else:
                                cont, cont_steps = 0.0, 0
                        else:
                            cont, cont_steps, _=rollout_policy(clone, policy, random.Random(seed*100000+t*1000+j), horizon)
                        total += cont; used += cont_steps
                    samples.append(total); branches += 1; total_steps += used
                mean=statistics.mean(samples); sd=statistics.stdev(samples) if len(samples)>1 else 0.0
                vals[action]={"mean":mean,"std":sd,"ci95":1.96*sd/math.sqrt(max(1,len(samples))),"n":len(samples)}
            center=statistics.mean(v["mean"] for v in vals.values())
            q_by_policy[policy]={a:{**v,"advantage":v["mean"]-center} for a,v in vals.items()}
        rows.append({"game":path.name,"seed":seed,"t":t,"observation":state.get("description","")[:1000],
                     "actions":actions,"q":q_by_policy})
        # Advance the live environment with a fixed legal action to obtain the
        # next history; this path is used only to enumerate snapshots.
        # Enumerate snapshots along the known quest path when possible, so
        # later snapshots retain a reachable positive action instead of
        # becoming label-sparse after an arbitrary detour.
        if prefix and walkthrough and len(prefix) < len(walkthrough) and walkthrough[len(prefix)] in actions:
            advance = walkthrough[len(prefix)]
        elif not prefix and walkthrough and walkthrough[0] in actions:
            advance = walkthrough[0]
        else:
            advance=actions[(seed+t)%len(actions)]
        state,_,done=run(env,advance); env._last_state=state; prefix.append(advance); env._prefix=prefix
        if done: break
    return rows,{"branch_rollouts":branches,"branch_env_steps":total_steps}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--games",type=int,default=24); p.add_argument("--train-games",type=int,default=16)
    p.add_argument("--snapshots",type=int,default=5); p.add_argument("--horizon",type=int,default=6); p.add_argument("--rollouts",type=int,default=8)
    p.add_argument("--max-actions",type=int,default=6); p.add_argument("--seed",type=int,default=20260818); p.add_argument("--out",default="stage_b_results"); a=p.parse_args()
    started=time.time(); root=Path(a.out)/"games"; audits=[]; labels=[]; ledger={"branch_rollouts":0,"branch_env_steps":0}
    seeds=[a.seed+i for i in range(a.games)]
    for i,seed in enumerate(seeds):
        path=make_game_file(root,seed,rooms=3,objects=5,quest_length=3)
        env_meta = json.loads(path.with_suffix(".json").read_text())
        # The generator's metadata contains the exact deterministic quest plan.
        # It is used only for the privileged replay-label sensitivity gate.
        path.with_suffix(".walkthrough").write_text(json.dumps(env_meta.get("metadata", {}).get("walkthrough", [])))
        audits.append(audit_game(path,seed))
        rows,cost=branch_values(path,min(a.snapshots,5),a.horizon,a.rollouts,seed,a.max_actions)
        split = "train" if i < min(a.train_games, a.games) else "test"
        for row in rows: row["split"] = split
        labels.extend(rows)
        for k,v in cost.items(): ledger[k]=ledger.get(k,0)+v
    replay_rates=[x["replay_transition_match_rate"] for x in audits]; copy_rates=[x["copy_branch_match_rate"] for x in audits]
    q_rows=[]
    for row in labels:
        for pol, vals in row["q"].items():
            for action,v in vals.items(): q_rows.append({"game":row["game"],"seed":row["seed"],"t":row["t"],"split":row.get("split",""),"policy":pol,"action":action,**v})
    os.makedirs(a.out,exist_ok=True)
    with open(Path(a.out)/"audit.json","w") as f: json.dump(audits,f,indent=2)
    with open(Path(a.out)/"labels.json","w") as f: json.dump(labels,f,indent=2)
    with open(Path(a.out)/"q_labels.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(q_rows[0])); w.writeheader(); w.writerows(q_rows)
    result={"config":vars(a),"environment":"TextWorld 1.7.0","candidate_mode":"oracle_admissible_actions",
            "games":a.games,"snapshots":len(labels),"split":{"train_games":min(a.train_games,a.games),"test_games":max(0,a.games-a.train_games),"train_snapshots":sum(x.get("split")=="train" for x in labels),"test_snapshots":sum(x.get("split")=="test" for x in labels)},"audit":{"reset_match_rate":sum(x["reset_match"] for x in audits)/len(audits),"replay_transition_match_rate":statistics.mean(replay_rates),"copy_branch_match_rate":statistics.mean(copy_rates)},
            "coverage":{"mean_actions_per_snapshot":statistics.mean([len(x["actions"]) for x in labels]) if labels else 0,"min_actions":min([len(x["actions"]) for x in labels],default=0)},
            "cost":{"wall_seconds":time.time()-started,"cpu_hours":(time.process_time()/3600),"gpu_hours":0.0,**ledger},
            "gate":{"replay_pass":min(replay_rates,default=0)==1.0 and min(copy_rates,default=0)==1.0,"enough_labels":len(labels)>=a.games}}
    with open(Path(a.out)/"results.json","w") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
