"""
Variance overview (moved deeper; WORKING mock, house style).
  (a) ABSOLUTE variance in agent engagement explained by persona vs model selection,
      by regime -- total grows and the model-selection component scales.
  (b) per-model attractiveness = mean incoming excess cross-model engagement H.
"""
import sys
from pathlib import Path
from collections import defaultdict
import json, sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from housestyle import serif, mono, MODEL_COLOR

RUNS = Path("runs")
OUT = Path(__file__).parent / "fig_variance_overview.png"
SHORT = {"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss",
         "mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4",
         "google/gemma-4-31B-it":"Gemma"}

# ---- Panel A: PROPORTIONAL variance of comments-per-post (CPP) explained by
#      own base model vs persona, by regime (in-sample R^2, %). Model overtakes persona.
absvar = {
    "Two-way":   {"persona": 8.5, "model": 9.9},
    "Three-way": {"persona": 7.3, "model": 13.3},
    "Four-way":  {"persona": 5.8, "model": 25.4},
}

# ---- Panel B: per-model attractiveness from run DBs ----
def attractiveness():
    inc = defaultdict(list)
    for db in sorted(RUNS.glob("sweep_*.db")):
        if any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]): continue
        mp = str(db).replace(".db",".metadata.json")
        if not Path(mp).exists(): continue
        try: amm = json.load(open(mp)).get("agent_model_map",{})
        except: continue
        if not amm: continue
        fam = {int(k): SHORT.get(v["model"]) for k,v in amm.items()}
        if any(v is None for v in fam.values()): continue
        try:
            c = sqlite3.connect(db)
            posts = c.execute("SELECT post_id,user_id FROM post").fetchall()
            coms = c.execute("SELECT post_id,user_id FROM comment").fetchall(); c.close()
        except: continue
        if len(coms) < 200: continue
        owner = {p:u for p,u in posts}; F = sorted(set(fam.values()))
        nf = {f: sum(1 for v in fam.values() if v==f) for f in F}; ntot = sum(nf.values())
        edges = [(fam[cu],fam[owner[p]]) for p,cu in coms if owner.get(p) and owner[p]!=cu and cu in fam and owner[p] in fam]
        tot = len(edges) or 1
        obs = defaultdict(int)
        for x,y in edges: obs[(x,y)] += 1
        for Y in F:
            hs=[obs.get((X,Y),0)/tot/((nf[X]/ntot)*((nf[Y])/(ntot-1)))-1 for X in F if X!=Y and nf[X]]
            if hs: inc[Y].append(np.mean(hs))
    return {m:(np.mean(v), np.std(v)/np.sqrt(len(v))) for m,v in inc.items()}

def temporal_incoming_H(n_bins=4):
    """Per-model incoming H over simulation quartiles (comment-order bins), pooled over runs."""
    acc = defaultdict(lambda: defaultdict(list))   # model -> bin -> [H per run]
    for db in sorted(RUNS.glob("sweep_*.db")):
        if any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]): continue
        mp = str(db).replace(".db",".metadata.json")
        if not Path(mp).exists(): continue
        try: amm = json.load(open(mp)).get("agent_model_map",{})
        except: continue
        if not amm: continue
        fam = {int(k): SHORT.get(v["model"]) for k,v in amm.items()}
        if any(v is None for v in fam.values()): continue
        try:
            c = sqlite3.connect(db)
            posts = c.execute("SELECT post_id,user_id FROM post").fetchall()
            coms = c.execute("SELECT post_id,user_id FROM comment ORDER BY comment_id").fetchall(); c.close()
        except: continue
        if len(coms) < 200: continue
        owner = {p:u for p,u in posts}; F = sorted(set(fam.values()))
        nf = {f: sum(1 for v in fam.values() if v==f) for f in F}; ntot = sum(nf.values())
        edges = [(fam[cu],fam[owner[p]]) for p,cu in coms
                 if owner.get(p) and owner[p]!=cu and cu in fam and owner[p] in fam]
        if len(edges) < n_bins*20: continue
        for b in range(n_bins):
            seg = edges[b*len(edges)//n_bins:(b+1)*len(edges)//n_bins]
            tot = len(seg) or 1
            obs = defaultdict(int)
            for xx,yy in seg: obs[(xx,yy)] += 1
            for Y in F:
                hs=[obs.get((X,Y),0)/tot/((nf[X]/ntot)*((nf[Y])/(ntot-1)))-1 for X in F if X!=Y and nf[X]]
                if hs: acc[Y][b].append(np.mean(hs))
    return acc

plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300})
fig,(axA,axB) = plt.subplots(1,2, figsize=(7.2,2.6), gridspec_kw={"width_ratios":[1,1.05]})

# Panel A: absolute grouped bars
regimes=["Two-way","Three-way","Four-way"]; x=np.arange(3); w=0.38
pc="#c0392b"; mc="#2c6fbb"
for i,key in enumerate(["persona","model"]):
    vals=[absvar[r][key] for r in regimes]; off=(-w/2 if i==0 else w/2)
    axA.bar(x+off, vals, w, color=(pc if i==0 else mc),
            label=("persona" if i==0 else "base model"), edgecolor="white", lw=0.5)
    for xi,vv in zip(x+off,vals):
        axA.text(xi, vv+0.4, f"{vv:.1f}", ha="center", va="bottom", fontproperties=mono(8))
axA.set_xticks(x); axA.set_xticklabels(regimes, fontproperties=mono(9))
axA.set_ylabel("CPP variance explained ($R^2$, \\%)", fontproperties=mono(8.5))
axA.set_ylim(0, 29)
axA.legend(prop=mono(8), frameon=False, loc="upper left")
axA.set_title("(a) Per-post engagement: model overtakes persona", fontproperties=serif(9.5, bold=True))
axA.tick_params(labelsize=8)

# Panel B: per-model incoming H across mixture sizes (one line per base model), centered at y=0
TETDIR = RUNS / "tetratic"
def regime_H():
    acc = defaultdict(lambda: defaultdict(list))   # model -> regime -> [H per run]
    def proc(globpat, tet=False):
        for db in sorted(globpat):
            if not tet and any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]): continue
            if tet and (TETDIR/f"{db.stem}_checkpoint.json").exists(): continue
            mp = str(db).replace(".db",".metadata.json")
            if not Path(mp).exists(): continue
            try: amm = json.load(open(mp)).get("agent_model_map",{})
            except: continue
            if not amm: continue
            fam = {int(k): SHORT.get(v["model"]) for k,v in amm.items()}
            if any(v is None for v in fam.values()): continue
            try:
                c = sqlite3.connect(db)
                posts = c.execute("SELECT post_id,user_id FROM post").fetchall()
                coms = c.execute("SELECT post_id,user_id FROM comment").fetchall(); c.close()
            except: continue
            if len(coms) < (100 if tet else 200): continue
            F = sorted(set(fam.values())); reg = {2:"dyad",3:"triad",4:"tetrad"}.get(len(F))
            if not reg: continue
            owner = {p:u for p,u in posts}; nf = {f: sum(1 for v in fam.values() if v==f) for f in F}; N = sum(nf.values())
            edges = [(fam[cu],fam[owner[p]]) for p,cu in coms if owner.get(p) and owner[p]!=cu and cu in fam and owner[p] in fam]
            tot = len(edges) or 1; obs = defaultdict(int)
            for a,b in edges: obs[(a,b)] += 1
            for Y in F:
                hs = [obs.get((X,Y),0)/tot/((nf[X]/N)*(nf[Y]/(N-1)))-1 for X in F if X!=Y]
                if hs: acc[Y][reg].append(np.mean(hs))
    proc(RUNS.glob("sweep_*.db")); proc(TETDIR.glob("tetra_s*_r*.db"), tet=True)
    return acc
rH = regime_H()
regs = ["dyad","triad","tetrad"]; xr = np.arange(3)
axB.axhline(0, color="#666", lw=0.9, ls=":")
for m in ["gpt-oss","Qwen","GLM-4","Gemma","Mag"]:
    if m not in rH: continue
    ys = [np.mean(rH[m][r]) if rH[m].get(r) else np.nan for r in regs]
    axB.plot(xr, ys, marker="o", ms=4.5, lw=1.8, color=MODEL_COLOR[m], label=m)
allv = [np.mean(rH[m][r]) for m in rH for r in regs if rH[m].get(r)]
M = max(abs(v) for v in allv) * 1.18
axB.set_ylim(-M, M)
axB.set_xticks(xr); axB.set_xticklabels(["dyad","triad","tetrad"], fontproperties=mono(9))
axB.set_xlim(-0.15, 2.15)
axB.set_ylabel("incoming $H$ (excess engagement)", fontproperties=mono(8.5))
axB.set_title("(b) Attraction sharpens with mixture size", fontproperties=serif(9.5, bold=True))
axB.legend(prop=mono(7), frameon=False, ncol=2, loc="lower left", handlelength=1.3)
axB.tick_params(labelsize=8)

plt.tight_layout(w_pad=1.5)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved {OUT.name}")
