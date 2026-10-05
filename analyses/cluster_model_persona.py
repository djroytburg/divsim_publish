"""Does agent style cluster more by model or by persona? Fair test via permutation-
adjusted variance-explained (controls for differing group counts: 5 models vs 43 personas).
Style space = SVD(30), axis 1 (language) dropped."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
RUNS=Path("runs")
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
ag=[]
for db in sorted(RUNS.glob("sweep_*.db")):
    if any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]):continue
    mp=str(db).replace(".db",".metadata.json")
    if not Path(mp).exists():continue
    try:amm=json.load(open(mp)).get("agent_model_map",{})
    except:continue
    if not amm:continue
    fam={int(k):SHORT.get(v["model"]) for k,v in amm.items()}
    if any(v is None for v in fam.values()):continue
    try:
        c=sqlite3.connect(db);posts=c.execute("SELECT user_id,content FROM post").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());coms=c.execute("SELECT 1 FROM comment").fetchall();c.close()
    except:continue
    if len(coms)<200:continue
    txt=defaultdict(list)
    for u,cont in posts:
        if u in fam and cont:txt[u].append(cont)
    for u,tl in txt.items():
        if tl:ag.append({"model":fam[u],"persona":users.get(u,f"u{u}"),"text":" ".join(tl)})
n=len(ag);print(n,"agents")
vc=TfidfVectorizer(ngram_range=(1,2),analyzer="word",min_df=10,max_features=20000,sublinear_tf=True,norm="l2",stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
X=vc.fit_transform([a["text"]for a in ag])
S=TruncatedSVD(30,random_state=0).fit_transform(X)
Z=((S-S.mean(0))/(S.std(0)+1e-9))[:,1:]
model=np.array([a["model"]for a in ag]);persona=np.array([a["persona"]for a in ag])
tot=((Z-Z.mean(0))**2).sum()
def eta2(lab):
    w=0.0
    for g in set(lab):
        Zi=Z[lab==g];w+=((Zi-Zi.mean(0))**2).sum()
    return 1-w/tot
rng=np.random.default_rng(0)
def null(lab,B=200):
    lab=lab.copy();vals=[]
    for _ in range(B):
        rng.shuffle(lab);vals.append(eta2(lab))
    return np.mean(vals)
em,ep=eta2(model),eta2(persona)
nm,np_=null(model),null(persona)
print(f"{'group':8s} {'#groups':>7} {'eta2':>7} {'null':>7} {'excess':>7}")
print(f"{'model':8s} {len(set(model)):7d} {em:7.3f} {nm:7.3f} {em-nm:7.3f}")
print(f"{'persona':8s} {len(set(persona)):7d} {ep:7.3f} {np_:7.3f} {ep-np_:7.3f}")
print(f"\nratio of excess (model/persona): {(em-nm)/(ep-np_):.2f}")
# also: within-group mean pairwise cosine (tightness), cheap proxy, fair-ish
def tight(lab):
    cs=[]
    for g in set(lab):
        Zi=Z[lab==g]
        if len(Zi)<3:continue
        c=Zi.mean(0);cs.append(np.mean([float(zi@c/((np.linalg.norm(zi)*np.linalg.norm(c))+1e-9)) for zi in Zi]))
    return np.mean(cs)
print(f"\nmean within-group cosine-to-centroid: model={tight(model):.3f}  persona={tight(persona):.3f}  (higher=tighter)")
