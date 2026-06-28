"""Content->engagement, FINAL: blocks = length, style (SVD axes 2-30, axis1=language dropped),
model, persona. No format block. Within-run z-scored in-degree; CV R^2 + leave-one-block-out unique."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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
        c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());c.close()
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
        if tl:ag.append({"run":db.stem,"model":fam[u],"persona":users.get(u,f"u{u}"),"indeg":ind.get(u,0),"nw":sum(len(t.split())for t in tl),"text":" ".join(tl)})
n=len(ag)
vc=TfidfVectorizer(ngram_range=(1,2),analyzer="word",min_df=10,max_features=20000,sublinear_tf=True,norm="l2",stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
X=vc.fit_transform([a["text"]for a in ag])
S=TruncatedSVD(30,random_state=0).fit_transform(X)
Sz=((S-S.mean(0))/(S.std(0)+1e-9))[:,1:]   # axes 2-30
model=np.array([a["model"]for a in ag]);persona=np.array([a["persona"]for a in ag])
logw=np.array([np.log1p(a["nw"])for a in ag]);L=((logw-logw.mean())/logw.std()).reshape(-1,1)
M=np.array([[1.0 if a["model"]==m else 0 for m in sorted(set(model))[1:]]for a in ag])
P=np.array([[1.0 if a["persona"]==p else 0 for p in sorted(set(persona))[1:]]for a in ag])
y=np.zeros(n);bd=defaultdict(list)
for i,a in enumerate(ag):bd[a["run"]].append(i)
for r,idx in bd.items():
    v=np.array([ag[i]["indeg"]for i in idx],float);z=(v-v.mean())/(v.std()+1e-9)
    for k,i in enumerate(idx):y[i]=z[k]
def cvr2(*B):
    B=[b for b in B if b is not None and b.shape[1]>0]
    if not B:return 0.0
    Xx=np.hstack(B);kf=KFold(5,shuffle=True,random_state=0);sse=sst=0
    for tr,te in kf.split(Xx):
        A=np.column_stack([np.ones(len(tr)),Xx[tr]]);Bt=np.column_stack([np.ones(len(te)),Xx[te]])
        b,*_=np.linalg.lstsq(A,y[tr],rcond=None);pp=Bt@b;sse+=((y[te]-pp)**2).sum();sst+=((y[te]-y[tr].mean())**2).sum()
    return 1-sse/sst
blk={"length":L,"style":Sz,"model":M,"persona":P}
single={k:cvr2(v)for k,v in blk.items()};allr=cvr2(*blk.values())
uniq={k:allr-cvr2(*[v for kk,v in blk.items() if kk!=k])for k in blk}
print(f"{n} agents, style = SVD axes 2-30")
print(f"{'block':8s} {'single':>7} {'unique':>7}")
for k in ["length","style","model","persona"]:print(f"{k:8s} {single[k]:7.3f} {uniq[k]:7.3f}")
print(f"{'all4':8s} {'':>7} {allr:7.3f}")
