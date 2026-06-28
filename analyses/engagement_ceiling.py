"""Is the low text->engagement R^2 a ceiling or an artifact of SVD/linearization?
Predict per-agent within-run-z CPP from: full TF-IDF (Ridge), SVD-30/100/500 (Ridge),
and a nonlinear model (HistGBT on SVD-100). 5-model pool, posts-only agent text."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
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
    fam={int(k)+1:SHORT.get(v["model"]) for k,v in amm.items()}
    if any(v is None for v in fam.values()):continue
    try:
        c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();c.close()
    except:continue
    if len(coms)<200:continue
    owner={p:u for p,u,_ in posts};ind=defaultdict(int);npost=defaultdict(int);txt=defaultdict(list)
    for p,cu in coms:
        o=owner.get(p)
        if o in fam and o!=cu:ind[o]+=1
    for p,u,cont in posts:
        if u in fam and cont:npost[u]+=1;txt[u].append(cont)
    for u in fam:
        if npost[u]>=2:ag.append({"run":db.stem,"cpp":ind[u]/npost[u],"text":" ".join(txt[u])})
n=len(ag);print(n,"agents")
runs=np.array([a["run"]for a in ag])
y=np.zeros(n);bd=defaultdict(list)
for i,a in enumerate(ag):bd[a["run"]].append(i)
for r,idx in bd.items():
    v=np.array([ag[i]["cpp"]for i in idx],float);m=v.mean();sd=v.std() or 1
    for j,i in enumerate(idx):y[i]=(v[j]-m)/sd
vc=TfidfVectorizer(ngram_range=(1,2),analyzer="word",min_df=10,max_features=40000,sublinear_tf=True,norm="l2",stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
X=vc.fit_transform([a["text"]for a in ag]);print("TF-IDF dims:",X.shape[1])
def cvr2(Xf,est_fn):
    kf=KFold(5,shuffle=True,random_state=0);sse=sst=0
    for tr,te in kf.split(Xf if not hasattr(Xf,'tocsr') else np.arange(n)):
        e=est_fn();e.fit(Xf[tr],y[tr]);pp=e.predict(Xf[te])
        sse+=((y[te]-pp)**2).sum();sst+=((y[te]-y[tr].mean())**2).sum()
    return 1-sse/sst
# full TF-IDF Ridge (a few alphas)
print(f"full TF-IDF Ridge(alpha=5):  R2={cvr2(X,lambda:Ridge(alpha=5.0)):.3f}",flush=True)
for K in [30,100]:
    S=TruncatedSVD(K,random_state=0).fit_transform(X)
    print(f"SVD-{K} Ridge:  R2={cvr2(S,lambda:Ridge(alpha=1.0)):.3f}",flush=True)
# nonlinear on SVD-100
S100=TruncatedSVD(100,random_state=0).fit_transform(X)
print(f"SVD-100 HistGBT (nonlinear):  R2={cvr2(S100,lambda:HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05)):.3f}")
