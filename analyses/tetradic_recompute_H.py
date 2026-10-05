"""Recompute four-way H stats over all COMPLETE tetratic runs (no checkpoint file).
Persistent: re-run anytime. Prints per-model incoming H + per-seed H->gpt-oss + counts."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
SHORT = {"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss",
         "mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4"}
perH = defaultdict(list); byseed = defaultdict(list); runs = []
for db in sorted(HERE.glob("tetra_s*_r*.db")):
    base = db.stem
    if (HERE / f"{base}_checkpoint.json").exists():
        continue  # incomplete
    mp = str(db).replace(".db", ".metadata.json")
    if not Path(mp).exists():
        continue
    seed = base.split("_")[1]
    fam = {int(k): SHORT.get(v["model"]) for k, v in json.load(open(mp))["agent_model_map"].items()}
    try:
        c = sqlite3.connect(db)
        posts = c.execute("SELECT post_id,user_id FROM post").fetchall()
        coms = c.execute("SELECT post_id,user_id FROM comment").fetchall(); c.close()
    except Exception:
        continue
    owner = {p: u for p, u in posts}; F = sorted(set(fam.values()))
    nf = {f: sum(1 for v in fam.values() if v == f) for f in F}; ntot = sum(nf.values())
    edges = [(fam[cu], fam[owner[p]]) for p, cu in coms
             if owner.get(p) and owner[p] != cu and cu in fam and owner[p] in fam]
    tot = len(edges)
    if tot < 100:
        continue
    obs = defaultdict(int)
    for x, y in edges: obs[(x, y)] += 1
    runs.append(base)
    for Y in F:
        h = np.mean([obs.get((X, Y), 0)/tot/((nf[X]/ntot)*((nf[Y])/(ntot-1)))-1 for X in F if X != Y])
        perH[Y].append(h)
        if Y == "gpt-oss": byseed[seed].append(h)

print(f"=== tetratic H over {len(runs)} complete four-way runs ===")
print(f"runs: {sorted(runs)}")
print(f"{'model':8s} {'meanH':>7} {'sd':>6} {'min':>7} {'max':>7}  n")
for m in ["gpt-oss", "Qwen", "GLM-4", "Mag"]:
    v = np.array(perH[m]) if perH[m] else np.array([np.nan])
    print(f"{m:8s} {v.mean():+7.2f} {v.std():6.2f} {v.min():+7.2f} {v.max():+7.2f}  {len(perH[m])}")
print("H->gpt-oss by seed:")
for s in sorted(byseed):
    v = byseed[s]; print(f"  seed {s:5s}: {np.mean(v):+.2f}  (n={len(v)})")
