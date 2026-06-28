"""Model style directions (centroids), their pairwise comparison + stability,
and their cosine distance to an 'engagingness' direction (linear engagement probe).
Style space = agent-level TF-IDF -> SVD (K=30), axis 1 (language) dropped."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from numpy.linalg import lstsq, norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
RUNS=Path("runs")
OUT=Path("out/style_geometry.md")
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
K=30; MODELS=["gpt-oss","Qwen","GLM-4","Mag","Gemma"]
ag=[]
for db in sorted(RUNS.glob("sweep_*.db")):
    if any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]):continue
    mp=str(db).replace(".db",".metadata.json")
    if not Path(mp).exists():continue
    try:amm=json.load(open(mp)).get("agent_model_map",{})
    except:continue
    if not amm:continue
    fam={int(k)+1:SHORT.get(v["model"]) for k,v in amm.items()}
    if any(v is None for v in fam.values()):continue
    try:
        c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();c.close()
    except:continue
    if len(coms)<200:continue
    owner={p:u for p,u,_ in posts};ind=defaultdict(int)
    for p,cu in coms:
        o=owner.get(p)
        if o in fam and o!=cu:ind[o]+=1
    txt=defaultdict(list)
    for p,u,cont in posts:
        if u in fam and cont:txt[u].append(cont)
    for u,tl in txt.items():
        if tl:ag.append({"run":db.stem,"model":fam[u],"indeg":ind.get(u,0),"text":" ".join(tl)})
n=len(ag);print(n,"agents")
vc=TfidfVectorizer(ngram_range=(1,2),analyzer="word",min_df=10,max_features=20000,sublinear_tf=True,norm="l2",stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
X=vc.fit_transform([a["text"]for a in ag])
S=TruncatedSVD(K,random_state=0).fit_transform(X)
Z=((S-S.mean(0))/(S.std(0)+1e-9))[:,1:]   # drop axis 1 (language)
model=np.array([a["model"]for a in ag]);runs=np.array([a["run"]for a in ag])
y=np.zeros(n);bd=defaultdict(list)
for i,a in enumerate(ag):bd[a["run"]].append(i)
for r,idx in bd.items():
    v=np.array([ag[i]["indeg"]for i in idx],float);z=(v-v.mean())/(v.std()+1e-9)
    for k,i in enumerate(idx):y[i]=z[k]
def u(x):return x/(norm(x)+1e-12)
cos=lambda a,b:float(u(a)@u(b))
cent={m:Z[model==m].mean(0) for m in MODELS}
# engagingness direction = OLS engagement ~ style (linear probe)
A=np.column_stack([np.ones(n),Z]);beta=lstsq(A,y,rcond=None)[0][1:];e=u(beta)
o=["# Style geometry: model directions, stability, and the engagingness vector\n",
   "Style space = agent TF-IDF -> SVD(30), axis 1 (language) dropped (29 dims).\n",
   "## 1. Model directions (centroids) -- pairwise cosine",
   "| | "+" | ".join(MODELS)+" |","|"+"---|"*(len(MODELS)+1)]
for m in MODELS:o.append(f"| {m} | "+" | ".join(f"{cos(cent[m],cent[m2]):+.2f}" for m2 in MODELS)+" |")
# ---- 2. stability via disjoint run-splits ----
rng=np.random.default_rng(0);uruns=np.array(sorted(set(runs)));B=60
selfc={m:[] for m in MODELS};e_self=[]
for _ in range(B):
    rng.shuffle(uruns);Aset=set(uruns[:len(uruns)//2]);am=np.array([r in Aset for r in runs])
    for m in MODELS:
        a1=(model==m)&am;b1=(model==m)&(~am)
        if a1.sum()<10 or b1.sum()<10:continue
        selfc[m].append(cos(Z[a1].mean(0),Z[b1].mean(0)))
    # engagingness vector stability
    bA=lstsq(np.column_stack([np.ones(am.sum()),Z[am]]),y[am],rcond=None)[0][1:]
    bB=lstsq(np.column_stack([np.ones((~am).sum()),Z[~am]]),y[~am],rcond=None)[0][1:]
    e_self.append(cos(bA,bB))
o.append("\n## 2. Direction stability (cosine between centroids from disjoint run-halves, mean over 60 splits)")
o.append("| model | self-cosine (1=perfectly stable) |");o.append("|---|---|")
for m in MODELS:o.append(f"| {m} | {np.mean(selfc[m]):.2f} |")
o.append(f"\nEngagingness vector self-cosine across run-halves: **{np.mean(e_self):.2f}** "
         f"(stability of the engagement direction itself).")
# ---- 3. distance from engagingness vector ----
o.append("\n## 3. Cosine of each model direction to the engagingness vector (higher = style closer to 'what engages')")
o.append("| model | cos(centroid, engagingness) |");o.append("|---|---|")
ranked=sorted(MODELS,key=lambda m:cos(cent[m],e),reverse=True)
for m in ranked:o.append(f"| {m} | {cos(cent[m],e):+.2f} |")
o.append(f"\nClosest to the engagingness direction: **{ranked[0]}**. "
         "If that is gpt-oss, the engaging style direction coincides with the attractor's style.")
OUT.write_text("\n".join(o)+"\n");print("\n".join(o));print("wrote",OUT)
