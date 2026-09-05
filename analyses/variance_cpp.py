"""Finding 2 redone on comments-per-post (CPP = in-degree/n_posts).
Decomposition (dy/tri/tet), LOMO (dy/tri), totals + absolute components,
and length-residualized per-model attractiveness for ALL models."""
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
        # OFF-BY-ONE FIX: DBs are 0-indexed (user_id and agent_model_map keys both start at MIN);
        # original "+1" dropped agent 0 and mislabeled the first agent of each model-block.
        # fam={int(k)+1:SHORT.get(v["model"]) for k,v in amm.items()}
        umin=min(int(k) for k in amm)
        fam={int(k)+umin:SHORT.get(v["model"]) for k,v in amm.items()}
        if any(v is None for v in fam.values()):continue
        try:
            c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());c.close()
        except:continue
        if len(coms)<(100 if tet else 200):continue
        owner={p:u for p,u,_ in posts};ind=defaultdict(int);npost=defaultdict(int);nw=defaultdict(int)
        for p,cu in coms:
            o=owner.get(p)
            if o in fam and o!=cu:ind[o]+=1
        for p,u,cont in posts:
            if u in fam and cont:npost[u]+=1;nw[u]+=len(cont.split())
        F=sorted(set(fam.values()));reg={2:"dyad",3:"tri",4:"tet"}.get(len(F))
        for u in fam:
            if npost[u]>=2:
                out.append({"run":db.stem,"model":fam[u],"persona":users.get(u,f"u{u}"),"cpp":ind[u]/npost[u],
                            "meanlen":nw[u]/npost[u],"net":F})
    return out
rows=load(RUNS.glob("sweep_*.db"))+load(TET.glob("tetra_s*_r*.db"),tet=True)
def dum(keys):
    lv=sorted(set(map(str,keys)));idx={l:i for i,l in enumerate(lv)};D=np.zeros((len(keys),len(lv)))
    for r,k in enumerate(keys):D[r,idx[str(k)]]=1
    return D[:,1:]
def r2(y,*B):
    B=[b for b in B if b is not None and b.shape[1]>0];X=np.column_stack([np.ones(len(y))]+B) if B else np.ones((len(y),1))
    b,*_=np.linalg.lstsq(X,y,rcond=None);p=X@b;return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
def partner(r):return frozenset(m for m in r["net"] if m!=r["model"])
print("=== Finding 2 on CPP: decomposition ===")
print(f"{'block':14s} {'dy':>7} {'tri':>7} {'tet':>7}")
def col(fn):
    return [fn([r for r in rows if {'dyad':'dyad','tri':'tri','tet':'tet'}[r_reg:=({2:'dyad',3:'tri',4:'tet'}[len(r['net'])])]==m]) for m in ['dyad','tri','tet']]
def sub(mode):return [r for r in rows if {2:'dyad',3:'tri',4:'tet'}[len(r['net'])]==mode]
def line(lab,fn):
    print(f"{lab:14s} "+" ".join(f"{fn(sub(m)):7.3f}" for m in ['dyad','tri','tet']))
P=lambda s:dum([r['persona']for r in s]);M=lambda s:dum([r['model']for r in s])
R=lambda s:dum([r['run']for r in s]);O=lambda s:np.array([[1.0 if r['model']=='gpt-oss' else 0]for r in s])
SEL=lambda s:np.column_stack([M(s),dum([partner(r)for r in s])])
Y=lambda s:np.array([r['cpp']for r in s])
line("persona",lambda s:r2(Y(s),P(s)))
line("model(own)",lambda s:r2(Y(s),M(s)))
line("model-selection",lambda s:r2(Y(s),SEL(s)))
line("gpt-oss ind",lambda s:r2(Y(s),O(s)))
line("run",lambda s:r2(Y(s),R(s)))
line("persona+model",lambda s:r2(Y(s),P(s),M(s)))
line("persona+run",lambda s:r2(Y(s),P(s),R(s)))
line("full",lambda s:r2(Y(s),P(s),M(s),R(s)))
line("dR2_model",lambda s:r2(Y(s),P(s),M(s),R(s))-r2(Y(s),P(s),R(s)))
print("\ntotal CPP variance + absolute components (R^2 x total):")
for m in ['dyad','tri','tet']:
    s=sub(m);y=Y(s);tot=y.var()
    print(f"  {m}: n={len(s)} totalvar={tot:.3f}  persona={r2(y,P(s))*tot:.3f}  model-sel={r2(y,SEL(s))*tot:.3f}  oss-ind={r2(y,O(s))*tot:.3f}")
print("\n=== LOMO on CPP (dy/tri): persona / own / selection ===")
for excl in ["baseline","Qwen","gpt-oss","Mag","GLM-4","Gemma"]:
    for m in ['dyad','tri']:
        s=[r for r in sub(m) if excl=="baseline" or excl not in r['net']]
        if len(s)<50:continue
        y=Y(s);print(f"  {excl:8s} {m:5s} n={len(s):4d} persona={r2(y,P(s))*100:5.1f}% own={r2(y,M(s))*100:5.1f}% sel={r2(y,SEL(s))*100:5.1f}%")
print("\n=== length-residualized CPP attractiveness, ALL models, by regime ===")
def wz(s,key):
    o={};bd=defaultdict(list)
    for r in s:bd[r['run']].append(r)
    arr={}
    vals=np.array([r[key]for r in s]);run=[r['run']for r in s]
    z=np.zeros(len(s))
    for rr in set(run):
        idx=[i for i,x in enumerate(run) if x==rr];v=vals[idx];m=v.mean();sd=v.std() or 1
        for i in idx:z[i]=(vals[i]-m)/sd
    return z
for m in ['dyad','tri','tet']:
    s=sub(m);cz=wz(s,'cpp');lz=wz(s,'meanlen');b=np.polyfit(lz,cz,1)[0];res=cz-b*lz
    md=np.array([r['model']for r in s])
    print(f"  {m} (length slope {b:+.2f}):")
    for mod in ["gpt-oss","Qwen","GLM-4","Mag","Gemma"]:
        mm=md==mod
        if mm.sum():print(f"     {mod:8s} raw {cz[mm].mean():+.2f}  len-adj {res[mm].mean():+.2f}")
