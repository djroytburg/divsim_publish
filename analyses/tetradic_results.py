"""Tetradic (four-way) results: pooled 4x4 H heatmap + descriptive stats + variance decomposition.
Over all COMPLETE tetradic runs (no checkpoint file). Persistent; re-run as more land.
Outputs:
  - figures/fig_h_tetradic.png  (heatmap, style matched to plot_matrices.py)
  - _paper/analyses/tetradic_results.md  (numbers to hand to DR)
"""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

HERE = Path(__file__).parent
FIGDIR = Path(".")
OUTMD = Path("out/tetradic_results.md")
SHORT = {"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss",
         "mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4"}
MODELS = ["GLM-4","Mag","Qwen","gpt-oss"]
try:
    for ttf in (Path.home()/"fonts").rglob("*.ttf"): fontManager.addfont(str(ttf))
    FAM = "Ubuntu Mono"
except Exception:
    FAM = "monospace"

perH = defaultdict(list)          # incoming H per target across runs
matruns = []                      # list of 4x4 H matrices
lead = last = 0; rho = []; nrun = 0
agents = []                       # for variance decomposition

for db in sorted(HERE.glob("tetra_s*_r*.db")):
    base = db.stem
    if (HERE/f"{base}_checkpoint.json").exists(): continue
    mp = str(db).replace(".db",".metadata.json")
    if not Path(mp).exists(): continue
    fam = {int(k)+1: SHORT.get(v["model"]) for k,v in json.load(open(mp))["agent_model_map"].items()}
    if any(v is None for v in fam.values()): continue
    try:
        c = sqlite3.connect(db)
        posts = c.execute("SELECT post_id,user_id FROM post").fetchall()
        coms  = c.execute("SELECT post_id,user_id FROM comment").fetchall()
        users = dict(c.execute("SELECT user_id,user_name FROM user").fetchall()); c.close()
    except Exception: continue
    owner = {p:u for p,u in posts}; F = sorted(set(fam.values()))
    nf = {f: sum(1 for v in fam.values() if v==f) for f in F}; ntot = sum(nf.values())
    edges = [(fam[cu], fam[owner[p]]) for p,cu in coms
             if owner.get(p) and owner[p]!=cu and cu in fam and owner[p] in fam]
    tot = len(edges)
    if tot < 100: continue
    nrun += 1
    obs = defaultdict(int)
    for x,y in edges: obs[(x,y)] += 1
    # 4x4 matrix (rows=source/commenter, cols=target/author)
    Mx = np.full((4,4), np.nan)
    for i,X in enumerate(MODELS):
        for j,Y in enumerate(MODELS):
            exp = (nf[X]/ntot)*((nf[Y]-(1 if X==Y else 0))/(ntot-1))
            if exp>0: Mx[i,j] = obs.get((X,Y),0)/tot/exp - 1
    matruns.append(Mx)
    # incoming H per target (mean over X!=Y)
    for Y in MODELS:
        perH[Y].append(np.nanmean([Mx[i,MODELS.index(Y)] for i,X in enumerate(MODELS) if X!=Y]))
    # in-degree per agent + lead/last + rho
    ind = defaultdict(int)
    for p,cu in coms:
        o = owner.get(p)
        if o in fam and o!=cu: ind[o]+=1
    fam_in = defaultdict(list)
    for u,f in fam.items(): fam_in[f].append(ind.get(u,0))
    mean_in = {f: np.mean(v) for f,v in fam_in.items()}
    if max(mean_in, key=mean_in.get)=="gpt-oss": lead+=1
    if min(mean_in, key=mean_in.get)=="Mag": last+=1
    tot_in = sum(sum(v) for v in fam_in.values())
    share = sum(fam_in["gpt-oss"])/tot_in if tot_in else 0
    rho.append(share/(nf["gpt-oss"]/ntot))
    for u,f in fam.items():
        agents.append({"run":base,"model":f,"persona":users.get(u,f"u{u}"),"indeg":ind.get(u,0)})

# ---- pooled matrix + heatmap ----
Mpool = np.nanmean(np.stack(matruns), axis=0)
fig, ax = plt.subplots(figsize=(4.2,3.6))
vmax = np.nanmax(np.abs(Mpool))
im = ax.imshow(Mpool, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(MODELS, fontproperties=FontProperties(family=FAM,size=9), rotation=30, ha="right")
ax.set_yticklabels(MODELS, fontproperties=FontProperties(family=FAM,size=9))
ax.set_xlabel("post author (target)", fontproperties=FontProperties(family=FAM,size=9))
ax.set_ylabel("commenter (source)", fontproperties=FontProperties(family=FAM,size=9))
for i in range(4):
    for j in range(4):
        v = Mpool[i,j]
        ax.text(j,i,f"{v:+.2f}",ha="center",va="center",
                fontproperties=FontProperties(family=FAM,size=8),
                color="white" if abs(v)>0.55*vmax else "black")
cb = fig.colorbar(im, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
ax.set_title(f"Four-way $H_{{X\\to Y}}$ (pooled, n={nrun} runs)",
             fontproperties=FontProperties(family=FAM,size=10))
fig.tight_layout(); fig.savefig(FIGDIR/"fig_h_tetradic.png", dpi=300, bbox_inches="tight")
print("wrote", FIGDIR/"fig_h_tetradic.png")

# ---- variance decomposition (in-sample R^2, matches tab:ols_decomp convention) ----
y = np.array([a["indeg"] for a in agents], float)
def dummies(keys):
    levels = sorted(set(keys)); idx = {l:i for i,l in enumerate(levels)}
    D = np.zeros((len(keys), len(levels)))
    for r,k in enumerate(keys): D[r, idx[k]] = 1
    return D[:,1:]  # drop one for intercept
def r2(*blocks):
    blocks = [b for b in blocks if b is not None and b.shape[1]>0]
    X = np.column_stack([np.ones(len(y))] + blocks) if blocks else np.ones((len(y),1))
    b,*_ = np.linalg.lstsq(X, y, rcond=None); pred = X@b
    return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
P = dummies([a["persona"] for a in agents])
Mod = dummies([a["model"] for a in agents])
R = dummies([a["run"] for a in agents])
ossind = np.array([[1.0 if a["model"]=="gpt-oss" else 0.0] for a in agents])
vd = {"persona_only":r2(P), "model_only":r2(Mod), "oss_only":r2(ossind),
      "run_only":r2(R), "persona+model":r2(P,Mod), "persona+run":r2(P,R), "full":r2(P,Mod,R)}
dmodel = vd["full"] - vd["persona+run"]

# ---- write md ----
o = [f"# Tetradic (four-way) results — {nrun} complete runs\n",
     "## Incoming H per model (mean over commenter families, across runs)\n",
     "| model | mean H | SD | min | max |", "|---|---:|---:|---:|---:|"]
for m in ["gpt-oss","Qwen","GLM-4","Mag"]:
    v = np.array(perH[m]); o.append(f"| {m} | {v.mean():+.2f} | {v.std():.2f} | {v.min():+.2f} | {v.max():+.2f} |")
o += [f"\n- gpt-oss is the **most-commented family in {lead}/{nrun} runs** ({100*lead/nrun:.0f}%).",
      f"- Magistral is the **least-commented in {last}/{nrun} runs** ({100*last/nrun:.0f}%).",
      f"- gpt-oss over-representation rho = comment-share / fair-share: **mean {np.mean(rho):.2f}, median {np.median(rho):.2f}**.",
      "\n## Variance decomposition (per-agent raw in-degree, in-sample R^2)\n",
      "| block | R^2 |", "|---|---:|"]
for k in ["persona_only","model_only","oss_only","run_only","persona+model","persona+run","full"]:
    o.append(f"| {k.replace('_',' ')} | {vd[k]:.3f} |")
o.append(f"| **Delta R^2 model** (full - persona+run) | **{dmodel:+.3f}** |")
o.append(f"\nn agents = {len(agents)}. Compare dyad Delta R^2_model +.044, triad +.071.")
OUTMD.write_text("\n".join(o)+"\n"); print("\n".join(o)); print("wrote", OUTMD)
