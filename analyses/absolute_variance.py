"""Exact absolute-variance decomposition of per-agent in-degree, by regime
(two-way / three-way from the 5-model pool; four-way from the tetradic pool).
Absolute component = single-block in-sample R^2 x total in-degree variance.
Keeps everything consistent with tab:ols_decomp and Fig variance_overview(a).
"""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
RUNS=Path("runs")
TET=RUNS/"tetratic"
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}

def rows_from(dbiter, tet=False):
    rows=[]
    for db in dbiter:
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
            c=sqlite3.connect(db);posts=c.execute("SELECT post_id,user_id FROM post").fetchall();coms=c.execute("SELECT post_id,user_id FROM comment").fetchall();users=dict(c.execute("SELECT user_id,user_name FROM user").fetchall());c.close()
        except:continue
        if len(coms)<(100 if tet else 200):continue
        owner={p:u for p,u in posts};ind=defaultdict(int)
        for p,cu in coms:
            o=owner.get(p)
            if o in fam and o!=cu:ind[o]+=1
        net=tuple(sorted(set(fam.values())))
        mode="tet" if tet else ("tri" if len(net)==3 else "dyad" if len(net)==2 else "other")
        for u,f in fam.items():
            rows.append({"run":db.stem,"model":f,"persona":users.get(u,f"u{u}"),"indeg":ind.get(u,0),"mode":mode,"net":net})
    return rows

def dummies(keys):
    lv=sorted(set(keys),key=lambda x:str(x));idx={l:i for i,l in enumerate(lv)}
    D=np.zeros((len(keys),len(lv)))
    for r,k in enumerate(keys):D[r,idx[k]]=1
    return D[:,1:]
def r2(y,*blocks):
    blocks=[b for b in blocks if b is not None and b.shape[1]>0]
    X=np.column_stack([np.ones(len(y))]+blocks) if blocks else np.ones((len(y),1))
    b,*_=np.linalg.lstsq(X,y,rcond=None);pred=X@b
    return 1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
def partner(r): return frozenset(m for m in r["net"] if m!=r["model"])

allrows=rows_from(sorted(RUNS.glob("sweep_*.db")))+rows_from(sorted(TET.glob("tetra_s*_r*.db")),tet=True)
out=["# Absolute variance decomposition (per-agent in-degree)\n",
     "| component | Two-way | Three-way | Four-way |","|---|---:|---:|---:|"]
comp={}
for mode in ["dyad","tri","tet"]:
    rs=[r for r in allrows if r["mode"]==mode]
    y=np.array([r["indeg"] for r in rs],float);tot=y.var()
    P=dummies([r["persona"] for r in rs]);Mo=dummies([r["model"] for r in rs])
    Sel=np.column_stack([Mo,dummies([partner(r) for r in rs])]) if dummies([partner(r) for r in rs]).shape[1]>0 else Mo
    oss=np.array([[1.0 if r["model"]=="gpt-oss" else 0.0] for r in rs]);Run=dummies([r["run"] for r in rs])
    comp[mode]={"n":len(rs),"total":tot,"persona":r2(y,P)*tot,"own":r2(y,Mo)*tot,
                "selection":r2(y,Sel)*tot,"oss":r2(y,oss)*tot,"run":r2(y,Run)*tot,
                "persona_r2":r2(y,P),"sel_r2":r2(y,Sel),"own_r2":r2(y,Mo),"oss_r2":r2(y,oss),"run_r2":r2(y,Run)}
def row(label,key):
    return f"| {label} | "+" | ".join(f"{comp[m][key]:.0f}" for m in ["dyad","tri","tet"])+" |"
out+=[row("Total variance","total"),"|---|---|---|---|",
      row("Persona","persona"),row("Model selection (own+partner)","selection"),
      row("\\quad own model","own"),row("\\quad gpt-oss indicator","oss"),row("Run (baseline)","run")]
out+=["\n## underlying R^2 (for tab:ols_decomp consistency)\n",
      "| block | Two-way | Three-way | Four-way |","|---|---:|---:|---:|"]
for lab,key in [("persona","persona_r2"),("model selection","sel_r2"),("own model","own_r2"),("gpt-oss ind","oss_r2"),("run","run_r2")]:
    out.append(f"| {lab} | "+" | ".join(f"{comp[m][key]:.3f}" for m in ['dyad','tri','tet'])+" |")
out.append(f"\nn agents: dyad {comp['dyad']['n']}, tri {comp['tri']['n']}, tet {comp['tet']['n']}")
print("\n".join(out))
Path("out/absolute_variance.md").write_text("\n".join(out)+"\n")
