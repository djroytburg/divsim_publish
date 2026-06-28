"""
Lexical-features table for Finding 3 (part 1), persistent.
Rows = models. Columns:
  - CV accuracy (per-model recall) with Wilson 95% CI
  - leave-personas-out accuracy (per-model recall) with Wilson 95% CI
  - top distinguishing lexical features by FIRING (weight x mean in-class TF-IDF activation)
Plus 1-2 qualitative example snippets per model.
Outputs markdown + a LaTeX table fragment.
"""
import json, math, re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from pathlib import Path

DATA = Path("data/texts.jsonl")
OUTMD = Path("out/lexical_table.md")
OUTTEX = Path("./_paper/latex/tables/lexical_table.tex")
OUTTEX.parent.mkdir(parents=True, exist_ok=True)
MODELS = ["gpt-oss","Qwen","GLM-4","Mag","Gemma"]
NAME = {"gpt-oss":"gpt-oss-20b","Qwen":"Qwen3-32B","GLM-4":"GLM-4-32B","Mag":"Magistral-Small","Gemma":"Gemma-4-31B"}

rows = [json.loads(l) for l in open(DATA)]
rows = [r for r in rows if r["model"] in MODELS and len(r.get("text","") or "")>=10]
texts = [r["text"] for r in rows]
y = np.array([r["model"] for r in rows])
persona = np.array([r["persona"] for r in rows])
print(f"{len(rows)} texts")

# two vectorizers: combined (word+char) for the classifier; word-only for readable feature display
vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=10, max_features=40000,
                      sublinear_tf=True, stop_words="english", token_pattern=r"[A-Za-z'][A-Za-z']+")
vec_char = TfidfVectorizer(ngram_range=(3,5), analyzer="char_wb", min_df=20, max_features=40000, sublinear_tf=True)
Xw = vec.fit_transform(texts); Xc = vec_char.fit_transform(texts)
import scipy.sparse as sp
X = sp.hstack([Xw, Xc]).tocsr()
vocab_w = np.array(vec.get_feature_names_out())
nw = Xw.shape[1]

def wilson(k, n, z=1.96):
    if n==0: return (float("nan"),float("nan"))
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h, c+h)

def per_model_recall(preds, ytrue):
    out={}
    for m in MODELS:
        idx = ytrue==m; n=int(idx.sum()); k=int((preds[idx]==m).sum())
        lo,hi=wilson(k,n); out[m]=(k/n, lo, hi, n)
    return out

# ---- CV (random 5-fold) ----
preds_cv = np.empty(len(y), dtype=object)
for tr,te in StratifiedKFold(5, shuffle=True, random_state=42).split(X,y):
    clf=LogisticRegression(max_iter=400, C=2.0, n_jobs=-1)
    clf.fit(X[tr],y[tr]); preds_cv[te]=clf.predict(X[te])
cv = per_model_recall(preds_cv, y)
print("CV done")

# ---- leave-personas-out (GroupKFold by persona) ----
preds_lpo = np.empty(len(y), dtype=object)
for tr,te in GroupKFold(5).split(X,y,groups=persona):
    clf=LogisticRegression(max_iter=400, C=2.0, n_jobs=-1)
    clf.fit(X[tr],y[tr]); preds_lpo[te]=clf.predict(X[te])
lpo = per_model_recall(preds_lpo, y)
print("LPO done")

# ---- firing features: fit on all, contribution = weight * mean in-class activation (word feats only) ----
clf_full = LogisticRegression(max_iter=600, C=2.0, n_jobs=-1).fit(X,y)
classes = list(clf_full.classes_)
top_feat = {}
for m in MODELS:
    ci = classes.index(m)
    w = clf_full.coef_[ci][:nw]                      # weights on word features
    mask = (y==m)
    mean_act = np.asarray(Xw[mask].mean(axis=0)).ravel()   # mean TF-IDF activation in-class
    contrib = w * mean_act                            # how hard each feature fires for this class
    order = np.argsort(contrib)[::-1]
    feats=[vocab_w[j] for j in order[:8]]
    top_feat[m]=feats

# ---- qualitative example snippets: posts strongly containing the model's top features ----
examples={}
top_set={m:set(top_feat[m]) for m in MODELS}
for m in MODELS:
    best=("",-1)
    for r in rows:
        if r["model"]!=m: continue
        t=r["text"]; tl=" "+re.sub(r"[^a-z' ]"," ",t.lower())+" "
        hits=sum(1 for f in top_set[m] if " "+f+" " in tl)
        if 60<=len(t)<=240 and hits>best[1]: best=(t,hits)
    examples[m]=best[0].replace("\n"," ").strip()

# ---- write markdown ----
md=["# Lexical features by model (model classifier)\n",
    f"{len(rows):,} texts. Classifier: TF-IDF (word 1-2gram + char 3-5gram) -> multinomial logistic regression. "
    "Per-model accuracy = recall (fraction of that model's texts correctly identified). "
    "Wilson 95% CIs. Leave-personas-out = train/test on disjoint persona sets. "
    "Top features ranked by FIRING (weight x mean in-class TF-IDF activation), word features only.\n",
    "| model | CV acc [95% CI] | leave-personas-out acc [95% CI] | top firing features |",
    "|---|---|---|---|"]
for m in MODELS:
    a,lo,hi,n=cv[m]; a2,lo2,hi2,n2=lpo[m]
    md.append(f"| {NAME[m]} | {a:.2f} [{lo:.2f}, {hi:.2f}] | {a2:.2f} [{lo2:.2f}, {hi2:.2f}] | {', '.join(top_feat[m][:6])} |")
md.append("\n## Example snippets (containing each model's top firing features)\n")
for m in MODELS:
    md.append(f"- **{NAME[m]}**: \"{examples[m][:200]}\"")
OUTMD.write_text("\n".join(md)+"\n")

# ---- write LaTeX fragment ----
tex=["% auto-generated by lexical_table.py",
     "\\begin{table}[t]\\centering\\small",
     "\\begin{tabular}{lccl}",
     "\\toprule",
     "model & CV acc.\\ [95\\% CI] & leave-personas-out [95\\% CI] & top firing features \\\\",
     "\\midrule"]
for m in MODELS:
    a,lo,hi,n=cv[m]; a2,lo2,hi2,n2=lpo[m]
    feats=", ".join(f"\\texttt{{{f}}}" for f in top_feat[m][:4])
    tex.append(f"{NAME[m]} & {a:.2f} [{lo:.2f}, {hi:.2f}] & {a2:.2f} [{lo2:.2f}, {hi2:.2f}] & {feats} \\\\")
tex+=["\\bottomrule","\\end{tabular}",
      "\\caption{Each model's text is identifiable from lexical features alone, and the signal is persona-invariant (leave-personas-out). Per-model accuracy is recall; top features are those that fire hardest for the model (weight $\\times$ mean in-class TF-IDF activation).}",
      "\\label{tab:lexical}","\\end{table}"]
OUTTEX.write_text("\n".join(tex)+"\n")

print("\n".join(md[:10]))
for m in MODELS: print(m, "->", top_feat[m][:6])
print(f"\nwrote {OUTMD} and {OUTTEX}")
