"""Distinctive per-model vocabulary via several methods, so we can pick the best surface.
(1) log-odds-ratio w/ informative Dirichlet prior (Monroe 2008), word 1-2grams, stopwords removed
(2) same, bigrams only (phrases)
(3) classifier coefficient top terms"""
import json
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
rows=[json.loads(l) for l in open("data/texts.jsonl")]
rows=[r for r in rows if len(r.get("text","") or "")>=1 and r.get("model") in {"gpt-oss","Qwen","GLM-4","Mag","Gemma"}]
MODELS=["gpt-oss","Qwen","GLM-4","Mag","Gemma"]
NAME={"gpt-oss":"gpt-oss","Qwen":"Qwen","GLM-4":"GLM-4","Mag":"Magistral","Gemma":"Gemma"}
texts=[r["text"] for r in rows];model=np.array([r["model"] for r in rows])
def logodds(ngram):
    cv=CountVectorizer(ngram_range=ngram,min_df=30,max_features=50000,stop_words="english",token_pattern=r"[A-Za-z'][A-Za-z']+")
    X=cv.fit_transform(texts);vocab=np.array(cv.get_feature_names_out())
    tot=np.asarray(X.sum(0)).ravel();a0=tot.sum();alpha=tot.astype(float);N=tot.sum()
    out={}
    for m in MODELS:
        yi=np.asarray(X[model==m].sum(0)).ravel().astype(float)
        yr=tot-yi
        ni=yi.sum();nr=yr.sum()
        d=np.log((yi+alpha)/(ni+a0-yi-alpha))-np.log((yr+alpha)/(nr+a0-yr-alpha))
        var=1.0/(yi+alpha)+1.0/(yr+alpha)
        z=d/np.sqrt(var)
        out[m]=vocab[np.argsort(z)[::-1][:12]]
    return out
lo1=logodds((1,2));lo2=logodds((2,2))
print("=== (1) log-odds (word 1-2grams) ===")
for m in MODELS:print(f"  {NAME[m]:10s}: {', '.join(lo1[m])}")
print("\n=== (2) log-odds (bigrams / phrases) ===")
for m in MODELS:print(f"  {NAME[m]:10s}: {', '.join(lo2[m])}")
