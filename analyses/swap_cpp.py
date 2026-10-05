"""Controlled gpt-oss -> Magistral swap: matched run-pairs in the same seed whose
persona->model map is identical EXCEPT gpt-oss and Magistral are swapped. For the
personas that flip, compare comments-per-post under gpt-oss vs under Magistral."""
import sqlite3, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
RUNS=Path("runs");TET=RUNS/"tetratic"
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
def seed_of(name):
    m=re.search(r"_s(\d+)",name) or re.search(r"_s(\d+)_r",name)
    m=re.search(r"s(\d+)",name); return m.group(1) if m else None
def load(db,tet=False):
    base=db.stem
    if tet and (TET/f"{base}_checkpoint.json").exists():return None
    mp=str(db).replace(".db",".metadata.json")
    if not Path(mp).exists():return None
    try:amm=json.load(open(mp)).get("agent_model_map",{})
    except:return None
    if not amm:return None
    fam={int(k):SHORT.get(v["model"]) for k,v in amm.items()}
    if any(v is None for v in fam.values()):return None
    try:
        c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());c.close()
    except:return None
    if len(coms)<100:return None
    owner={p:u for p,u in posts};ind=defaultdict(int);npost=defaultdict(int)
    for p,cu in coms:
        o=owner.get(p)
        if o in fam and o!=cu:ind[o]+=1
    for p,u in posts:
        if u in fam:npost[u]+=1
    # per user_id: model, persona, cpp
    rec={}
    for u in fam:
        if npost[u]>=2:rec[u]={"model":fam[u],"persona":users.get(u,f"u{u}"),"cpp":ind[u]/npost[u]}
    return {"seed":re.search(r"s(\d+)",base).group(1),"name":base,"rec":rec}

runs=[]
for db in sorted(RUNS.glob("sweep_*.db")):
    if any(x in db.name for x in ["archive","olm","hermes","glm47","pilot_","snapshot","qwq"]):continue
    r=load(db);
    if r:runs.append(r)
for db in sorted(TET.glob("tetra_s*_r*.db")):
    r=load(db,tet=True)
    if r:runs.append(r)
print(len(runs),"runs loaded")
byseed=defaultdict(list)
for r in runs:byseed[r["seed"]].append(r)

oss_vals=[];mag_vals=[];npairs=0;nflips=0
for seed,rs in byseed.items():
    for i in range(len(rs)):
        for j in range(i+1,len(rs)):
            A,B=rs[i]["rec"],rs[j]["rec"]
            shared=set(A)&set(B)
            if len(shared)<10:continue
            # require same persona per shared slot, and model identical except oss<->mag
            ok=True;flips=[]
            for u in shared:
                if A[u]["persona"]!=B[u]["persona"]:ok=False;break
                ma,mb=A[u]["model"],B[u]["model"]
                if ma==mb:continue
                if {ma,mb}=={"gpt-oss","Mag"}:flips.append(u)
                else:ok=False;break
            if not ok or not flips:continue
            npairs+=1
            for u in flips:
                oss=A[u] if A[u]["model"]=="gpt-oss" else B[u]
                mag=A[u] if A[u]["model"]=="Mag" else B[u]
                oss_vals.append(oss["cpp"]);mag_vals.append(mag["cpp"]);nflips+=1
o=np.array(oss_vals);m=np.array(mag_vals)
print(f"matched swap pairs: {npairs}; flipped persona-observations: {nflips}")
print(f"mean CPP powered by gpt-oss: {o.mean():.2f}")
print(f"mean CPP powered by Magistral: {m.mean():.2f}")
print(f"drop: {100*(o.mean()-m.mean())/o.mean():.0f}% fewer CPP under Magistral")
# paired (same persona,pair) drop
d=(o-m)
print(f"paired mean drop: {d.mean():.2f} CPP ({100*d.mean()/o.mean():.0f}%); median paired drop {np.median(d):.2f}")
