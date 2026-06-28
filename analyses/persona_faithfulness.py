import json,math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
rows=[json.loads(l) for l in open("data/texts.jsonl")]
rows=[r for r in rows if len(r.get("text","") or "")>=1]
print(len(rows),"texts total")
pc=Counter(r["persona"] for r in rows);keep={p for p,c in pc.items() if c>=25}
sub=[r for r in rows if r["persona"] in keep]
print(len(sub),"texts;",len(keep),"personas (>=25)")
vec=TfidfVectorizer(ngram_range=(1,2),analyzer="word",max_features=60000,sublinear_tf=True,token_pattern=r"[A-Za-z'][A-Za-z']+")
vc=TfidfVectorizer(ngram_range=(3,5),analyzer="char_wb",max_features=60000,sublinear_tf=True)
import scipy.sparse as sp
Xw=vec.fit_transform([r["text"]for r in sub]);Xc=vc.fit_transform([r["text"]for r in sub]);X=sp.hstack([Xw,Xc]).tocsr()
y=np.array([r["persona"]for r in sub]);chance=1/len(keep)
pred=np.empty(len(y),dtype=object)
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    c=LogisticRegression(max_iter=300,C=2.0);c.fit(X[tr],y[tr]);pred[te]=c.predict(X[te])
acc=accuracy_score(y,pred)
print(f"PERSONA recovery: {acc*100:.1f}%  ({acc/chance:.0f}x chance {chance*100:.1f}%)")
SHORT={"gpt-oss":"gpt-oss","Qwen":"Qwen","GLM-4":"GLM-4","Mag":"Magistral","Gemma":"Gemma"}
for m in ["Gemma","Mag","gpt-oss","GLM-4","Qwen"]:
    s=[r for r in sub if r["model"]==m];yy=np.array([r["persona"]for r in s])
    pcm=Counter(yy);kk={p for p,c in pcm.items() if c>=15}
    s=[r for r in s if r["persona"] in kk];yy=np.array([r["persona"]for r in s])
    if len(set(yy))<3:continue
    Xs=sp.hstack([vec.transform([r["text"]for r in s]),vc.transform([r["text"]for r in s])]).tocsr()
    pr=np.empty(len(yy),dtype=object)
    for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xs,yy):
        c=LogisticRegression(max_iter=300,C=2.0);c.fit(Xs[tr],yy[tr]);pr[te]=c.predict(Xs[te])
    a=accuracy_score(yy,pr);ch=1/len(set(yy));print(f"  {m}: {a*100:.1f}%  {a/ch:.0f}x  ({len(set(yy))} personas)")
