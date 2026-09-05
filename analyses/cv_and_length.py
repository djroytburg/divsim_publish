"""(A) Held-out 5-fold CV R^2 for persona-only / model-only / full (addresses the
43-vs-5 parameter-count confound in in-sample R^2).  (B) Per-model engagement effect
(within-run-z CPP) raw vs. after controlling for post length -> attractor/repeller survive."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
RUNS=Path("runs");TET=RUNS/"tetratic"
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
def load(globpat,tet=False):
    rows=[]
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
            c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id,content FROM post").fetchall()
            coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());c.close()
        except:continue
        if len(coms)<(100 if tet else 200):continue
        owner={p:u for p,u,_ in posts};ind=defaultdict(int);npost=defaultdict(int);loglen=defaultdict(list)
        for p,cu in coms:
            o=owner.get(p)
            if o in fam and o!=cu:ind[o]+=1
        for p,u,cont in posts:
            if u in fam:npost[u]+=1;loglen[u].append(np.log(max(1,len(cont or ""))))
        reg={2:"dyad",3:"triad",4:"tetrad"}[len(set(fam.values()))]
        for u in fam:
            if npost[u]>=2:
                rows.append({"run":db.stem,"reg":reg,"model":fam[u],"persona":users.get(u,f"u{u}"),
                             "cpp":ind[u]/npost[u],"loglen":np.mean(loglen[u])})
    return rows
rows=load(RUNS.glob("sweep_*.db"))+load(TET.glob("tetra_s*_r*.db"),tet=True)
df=pd.DataFrame(rows)
# within-run z CPP and z loglen
df["z"]=df.groupby("run")["cpp"].transform(lambda v:(v-v.mean())/(v.std() or 1))
df["zlen"]=df.groupby("run")["loglen"].transform(lambda v:(v-v.mean())/(v.std() or 1))
def cvr2(sub,cols):
    y=sub["z"].values;X=pd.get_dummies(sub[cols],columns=cols,drop_first=False).astype(float).values
    kf=KFold(5,shuffle=True,random_state=0);sc=[]
    for tr,te in kf.split(X):
        m=LinearRegression().fit(X[tr],y[tr]);sc.append(r2_score(y[te],m.predict(X[te])))
    return np.mean(sc)
print("=== (A) HELD-OUT 5-fold CV R^2 (within-run z CPP) ===")
print(f"{'regime':10s}{'persona':>10}{'model':>9}{'full(p+m)':>11}  (#personas / #models)")
for reg in ["dyad","triad","tetrad"]:
    s=df[df.reg==reg]
    pe=cvr2(s,["persona"]);mo=cvr2(s,["model"]);fu=cvr2(s,["persona","model"])
    print(f"{reg:10s}{pe:>10.3f}{mo:>9.3f}{fu:>11.3f}  ({s.persona.nunique()} / {s.model.nunique()}), n={len(s)}")
print("\n=== (B) per-model engagement (z CPP) RAW vs LENGTH-CONTROLLED ===")
for reg in ["dyad","triad","tetrad"]:
    s=df[df.reg==reg].copy()
    # length-adjusted z: residual of z on zlen
    b=np.polyfit(s["zlen"],s["z"],1);s["zadj"]=s["z"]-(b[0]*s["zlen"]+b[1])
    print(f"-- {reg} (length slope b={b[0]:+.2f}) --")
    for m in ["gpt-oss","Mag","GLM-4","Qwen","Gemma"]:
        sm=s[s.model==m]
        if len(sm)<5:continue
        print(f"   {m:8s} raw z={sm['z'].mean():+.2f}   length-adj z={sm['zadj'].mean():+.2f}   (n={len(sm)}, mean loglen z={sm['zlen'].mean():+.2f})")
