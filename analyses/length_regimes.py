"""(A) disentangle 'length': total-words (volume) vs mean post length, as engagement predictors.
(B) per-regime (dyad/triad/tetrad) raw vs length-adjusted per-model attractiveness (within-run z in-degree)."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
RUNS=Path("runs");TET=RUNS/"tetratic"
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
def load(globpat,tet=False):
    out=[]
    for db in sorted(globpat):
        if not tet and any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]):continue
        if tet and (TET/f"{db.stem}_checkpoint.json").exists():continue
        mp=str(db).replace(".db",".metadata.json")
        if not Path(mp).exists():continue
        try:amm=json.load(open(mp)).get("agent_model_map",{})
        except:continue
        if not amm:continue
        fam={int(k):SHORT.get(v["model"]) for k,v in amm.items()}
        if any(v is None for v in fam.values()):continue
        try:
            c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();c.close()
        except:continue
        if len(coms)<(100 if tet else 200):continue
        owner={p:u for p,u,_ in posts};ind=defaultdict(int)
        for p,cu in coms:
            o=owner.get(p)
            if o in fam and o!=cu:ind[o]+=1
        nw=defaultdict(int);npost=defaultdict(int)
        for p,u,cont in posts:
            if u in fam and cont:nw[u]+=len(cont.split());npost[u]+=1
        F=sorted(set(fam.values()));reg={2:"dyad",3:"triad",4:"tetrad"}.get(len(F))
        for u in fam:
            if npost[u]>0:out.append({"run":db.stem,"model":fam[u],"indeg":ind.get(u,0),"meanlen":nw[u]/npost[u],"totw":nw[u],"reg":reg})
    return out
ag=load(RUNS.glob("sweep_*.db"))+load(TET.glob("tetra_s*_r*.db"),tet=True)
ag=[a for a in ag if a["reg"]]
def wz(x,run):
    o=np.zeros(len(x));bd=defaultdict(list)
    for i,r in enumerate(run):bd[r].append(i)
    for r,idx in bd.items():
        v=x[np.array(idx)];m=v.mean();sd=v.std() or 1
        for j,i in enumerate(idx):o[i]=(v[j]-m)/sd
    return o
run=np.array([a["run"]for a in ag]);model=np.array([a["model"]for a in ag]);reg=np.array([a["reg"]for a in ag])
y=wz(np.array([a["indeg"]for a in ag],float),run)
lmean=wz(np.log1p(np.array([a["meanlen"]for a in ag])),run)
ltot=wz(np.log1p(np.array([a["totw"]for a in ag])),run)
# (A) which 'length' predicts engagement
print("(A) within-run corr with engagement:  mean-post-length r=%.3f   total-words(volume) r=%.3f"%(np.corrcoef(lmean,y)[0,1],np.corrcoef(ltot,y)[0,1]))
# (B) per-regime raw vs length(mean)-adjusted attractiveness
print("\n(B) per-model attractiveness (within-run z in-degree): raw vs length-adjusted, by regime")
print(f"{'model':8s} "+" ".join(f"{r}:raw  {r}:adj " for r in ["dyad","triad","tetrad"]))
rows={}
for r in ["dyad","triad","tetrad"]:
    m_=reg==r;b=np.polyfit(lmean[m_],y[m_],1)[0];resid=y.copy();resid[m_]=y[m_]-b*lmean[m_]
    rows[r]=(b,resid)
for mod in ["gpt-oss","Qwen","GLM-4","Mag","Gemma"]:
    cells=[]
    for r in ["dyad","triad","tetrad"]:
        b,resid=rows[r];m_=(reg==r)&(model==mod)
        if m_.sum()==0:cells.append("  --     --  ")
        else:cells.append(f"{y[m_].mean():+5.2f}  {resid[m_].mean():+5.2f}")
    print(f"{mod:8s} "+"  ".join(cells))
print("\nslopes (length->engagement) per regime:",{r:round(rows[r][0],2) for r in rows})
