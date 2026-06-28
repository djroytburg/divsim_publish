"""Surface the agent-level TF-IDF -> SVD style axes: per-axis top +/- terms (from V^T)
and explained-variance share. Matches the finding3 representation."""
import sqlite3, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
RUNS=Path("runs")
OUT=Path("out/svd_axes.md")
SHORT={"Qwen/Qwen3-32B":"Qwen","openai/gpt-oss-20b":"gpt-oss","mistralai/Magistral-Small-2509":"Mag","zai-org/GLM-4-32B-0414":"GLM-4","google/gemma-4-31B-it":"Gemma"}
K=30
docs=[];models=[]
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
        c=sqlite3.connect(db);posts=c.execute("SELECT user_id,content FROM post").fetchall();coms=c.execute("SELECT 1 FROM comment").fetchall();c.close()
    except:continue
    if len(coms)<200:continue
    txt=defaultdict(list)
    for u,cont in posts:
        if u in fam and cont:txt[u].append(cont)
    for u,tl in txt.items():
        if tl:docs.append(" ".join(tl));models.append(fam[u])
print(len(docs),"agent docs")
vc=TfidfVectorizer(ngram_range=(1,2),analyzer="word",min_df=10,max_features=20000,sublinear_tf=True,norm="l2",stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
X=vc.fit_transform(docs);vocab=np.array(vc.get_feature_names_out())
svd=TruncatedSVD(K,random_state=0);S=svd.fit_transform(X)
V=svd.components_   # K x n_terms  (= V^T rows are axes)
evr=svd.explained_variance_ratio_
models=np.array(models)
o=[f"# SVD style axes ({len(docs)} agent docs, {X.shape[1]} terms, K={K})\n",
   f"Cumulative explained variance (top {K}): {evr.sum()*100:.1f}%\n",
   "| axis | EV% | top +terms | top -terms | model loading (mean axis score, high->low) |","|---|---|---|---|---|"]
for k in range(12):
    w=V[k];pos=vocab[np.argsort(w)[::-1][:6]];neg=vocab[np.argsort(w)[:6]]
    ml={m:S[models==m,k].mean() for m in ["gpt-oss","Qwen","GLM-4","Mag","Gemma"]}
    order=sorted(ml,key=ml.get,reverse=True)
    mls=", ".join(f"{m}{ml[m]:+.2f}" for m in order)
    o.append(f"| {k+1} | {evr[k]*100:.1f} | {', '.join(pos)} | {', '.join(neg)} | {mls} |")
OUT.write_text("\n".join(o)+"\n");print("\n".join(o));print("wrote",OUT)
