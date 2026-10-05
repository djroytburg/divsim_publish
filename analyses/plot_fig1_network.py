"""
Figure 1 (WORKING): the interaction network of one triadic run with a strong
attractor (gpt-oss) and a strong repeller (Magistral). Nodes = agents colored by
base model, sized by in-degree; directed edges = comment flow. Side panel: per-model
stats. Inset: a chat-window excerpt of the most-commented post.
Saves SVG (for redesign) + PNG (for the paper).
"""
import sqlite3, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
sys.path.insert(0, str(Path(__file__).parent))
from housestyle import serif, mono, MODEL_COLOR

DB = "runs/sweep_tri_oss_mag_glm_s42.db"
OUT_SVG = Path(__file__).parent / "fig1_network.svg"
OUT_PNG = Path(__file__).parent / "fig1_network.png"
SHORT = {"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss",
         "mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4",
         "google/gemma-4-31B-it":"Gemma"}

meta = json.load(open(DB.replace(".db",".metadata.json")))
fam = {int(k): SHORT.get(v["model"]) for k,v in meta["agent_model_map"].items()}
c = sqlite3.connect(DB)
posts = c.execute("SELECT post_id,user_id,content FROM post").fetchall()
coms  = c.execute("SELECT post_id,user_id,content FROM comment").fetchall()
users = dict(c.execute("SELECT user_id,user_name FROM user").fetchall())
c.close()
owner = {p:u for p,u,_ in posts}

# edges (commenter -> author), weighted; in-degree
w = defaultdict(int); indeg = defaultdict(int); outdeg = defaultdict(int)
for pid,cu,_ in coms:
    au = owner.get(pid)
    if au and au != cu and cu in fam and au in fam:
        w[(cu,au)] += 1; indeg[au]+=1; outdeg[cu]+=1

# most-commented post (restricted to agent-authored posts with a known model)
pc = defaultdict(int)
for pid,_,_ in coms: pc[pid]+=1
agent_posts = [p for p in pc if owner.get(p) in fam]
top_pid = max(agent_posts, key=pc.get)
top_author = owner.get(top_pid)
top_text = next((t for p,u,t in posts if p==top_pid), "")
top_n = pc[top_pid]

# per-model stats
fams = ["gpt-oss","Mag","GLM-4","Qwen","Gemma"]
fams = [f for f in fams if f in fam.values()]
nf = {f:sum(1 for v in fam.values() if v==f) for f in fams}
ntot = sum(nf.values()); tot = sum(w.values())
obs = defaultdict(int)
for (cu,au),ww in w.items(): obs[(fam[cu],fam[au])]+=ww
def Hin(Y):
    hs=[]
    for X in fams:
        if X==Y: continue
        o=obs.get((X,Y),0)/tot; e=(nf[X]/ntot)*((nf[Y])/(ntot-1))
        if e>0: hs.append(o/e-1)
    return np.mean(hs) if hs else float("nan")
stats = {f: (np.mean([indeg[u] for u in fam if fam[u]==f]), Hin(f)) for f in fams}

# ---- build graph ----
import math
# model-grouped layout: each model's agents cluster at a polygon vertex, so
# cross-model comment flow (and convergence on the attractor) is legible.
order_for_anchor = [f for f in ["gpt-oss","Mag","GLM-4","Qwen","Gemma"] if f in set(fam.values())]
anchors = {}
for i,f in enumerate(order_for_anchor):
    ang = 2*math.pi*i/len(order_for_anchor) + math.pi/2
    anchors[f] = (1.25*math.cos(ang), 1.25*math.sin(ang))
rng = np.random.default_rng(3)
pos = {u: (anchors[fam[u]][0] + rng.normal(0,0.30),
           anchors[fam[u]][1] + rng.normal(0,0.30)) for u in fam}

fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.3, 1.0],
                      wspace=0.06, hspace=0.30)
axN = fig.add_subplot(gs[:, 0]); axN.axis("off"); axN.set_aspect("equal")
axS = fig.add_subplot(gs[0, 1]); axS.axis("off")
axC = fig.add_subplot(gs[1, 1]); axC.axis("off")

# edges: declutter by drawing only weight>=2; width/alpha by weight
wmax = max(w.values())
for (cu,au),ww in w.items():
    if ww < 2: continue
    x1,y1 = pos[cu]; x2,y2 = pos[au]
    axN.annotate("", xy=(x2,y2), xytext=(x1,y1),
                 arrowprops=dict(arrowstyle="-|>", color="#9aa0a6",
                                 alpha=0.18+0.55*(ww/wmax), lw=0.4+1.8*(ww/wmax),
                                 shrinkA=5, shrinkB=6, connectionstyle="arc3,rad=0.08"))
# nodes (sqrt scaling so the attractor doesn't dominate the canvas)
for u in fam:
    x,y = pos[u]
    s = 35 + 45*np.sqrt(indeg[u])
    axN.scatter([x],[y], s=s, c=MODEL_COLOR[fam[u]], edgecolors="white", linewidths=0.8, zorder=3)
# faint cluster labels at anchors
for f,(ax_,ay_) in anchors.items():
    axN.text(ax_*1.32, ay_*1.32, f, fontproperties=mono(8.5, bold=True),
             color=MODEL_COLOR[f], ha="center", va="center", zorder=4)
xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
m=0.45
axN.set_xlim(min(xs)-m, max(xs)+m); axN.set_ylim(min(ys)-m, max(ys)+m)
axN.set_title("Interaction network: one three-way run",
              fontproperties=serif(12, bold=True), pad=10)
# legend
import matplotlib.lines as mlines
handles=[mlines.Line2D([],[],marker="o",ls="",mfc=MODEL_COLOR[f],mec="white",ms=9,label=f) for f in fams]
axN.legend(handles=handles, loc="lower left", prop=mono(8), frameon=False, handletextpad=0.2, borderpad=0.2)
axN.text(0.5,-0.03,"node size $\\propto$ in-degree · edge = comment flow",
         transform=axN.transAxes, ha="center", fontproperties=mono(7.5), color="#555")

# ---- side stats ----
axS.set_title("per-model", fontproperties=serif(10, bold=True), loc="left", pad=2)
axS.text(0, 0.86, f"{'model':<8}{'in-deg':>7}{'H':>7}", fontproperties=mono(8.5, bold=True), transform=axS.transAxes)
for i,f in enumerate(fams):
    md, h = stats[f]
    axS.text(0, 0.86-0.13*(i+1),
             f"{f:<8}{md:>7.1f}{h:>+7.2f}", fontproperties=mono(8.5),
             color=MODEL_COLOR[f], transform=axS.transAxes)
axS.text(0, 0.86-0.13*(len(fams)+1.4),
         f"{tot} cross-comments\n{len(posts)} posts, n={ntot} agents",
         fontproperties=mono(7.5), color="#444", transform=axS.transAxes, va="top")

# ---- chat window ----
box = FancyBboxPatch((0.02,0.04),0.96,0.92, boxstyle="round,pad=0.02,rounding_size=0.04",
                     fc="#f3f4f6", ec="#cfd3da", lw=1.0, transform=axC.transAxes)
axC.add_patch(box)
def ascii_only(s): return s.encode("ascii","ignore").decode().strip()
auth_name = ascii_only(users.get(top_author, f"u{top_author}")) or f"u{top_author}"
auth_model = fam.get(top_author,"?")
header = f"{auth_name[:18]}  ({auth_model})"
axC.text(0.07,0.85, header, fontproperties=mono(8.5,bold=True),
         color=MODEL_COLOR.get(auth_model,"#333"), transform=axC.transAxes)
import textwrap
body = ascii_only(top_text.replace("\n"," "))
wrapped = "\n".join(textwrap.fill(body, width=32).split("\n")[:4])
if len(body) > len(wrapped): wrapped += " ..."
axC.text(0.07,0.72, wrapped, fontproperties=mono(8), color="#222",
         transform=axC.transAxes, va="top")
axC.text(0.07,0.10, f">> {top_n} comments", fontproperties=mono(9,bold=True),
         color="#1f77b4", transform=axC.transAxes)
axC.set_title("most-engaged post", fontproperties=serif(10, bold=True), loc="left", pad=2)

plt.savefig(OUT_SVG, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"saved {OUT_SVG.name} + {OUT_PNG.name}")
print(f"run={Path(DB).stem}  top_post nc={top_n} author={auth_name}({auth_model})")
for f in fams: print(f"  {f}: in-deg={stats[f][0]:.1f} H={stats[f][1]:+.2f}")
