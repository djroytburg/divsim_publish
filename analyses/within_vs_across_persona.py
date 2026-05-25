"""
Within-persona vs across-persona variance decomposition for per-agent engagement.

Computes, separately for two-way and three-way mixtures and pooled:
  - mean and total standard deviation of in-degree
  - standard deviation of per-persona means (cross-persona spread)
  - pooled standard deviation around persona means (within-persona spread)
  - ratio of within / across
  - block R^2 on total variance for persona identity, own model identity,
    partner-pair indicators, model selection (own + partner pair), and a
    random-run baseline.

Numbers match the variance-decomposition table and Figure 1 in the paper.

Usage:
  python within_vs_across_persona.py [--runs-dir PATH]
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona_variance as pv

MODELS = ["Qwen", "gpt-oss", "Mag", "GLM-4", "Gemma"]


def annotate_network(rows):
    by_run = defaultdict(set)
    for r in rows:
        by_run[r["run"]].add(r["model"])
    for r in rows:
        r["network"] = tuple(sorted(by_run[r["run"]]))


def partner_pair(r):
    return frozenset(m for m in r["network"] if m != r["model"])


def r2(rows, cols):
    y = np.array([float(r["in_deg"]) for r in rows])
    X = np.column_stack([np.ones(len(rows))] + list(cols))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ beta
    if y.var() == 0:
        return 0.0
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def persona_cols(rows):
    ps = sorted(set(r["persona"] for r in rows))[:-1]
    return [[1.0 if r["persona"] == p else 0.0 for r in rows] for p in ps]


def own_model_cols(rows):
    return [[1.0 if r["model"] == m else 0.0 for r in rows] for m in MODELS[:-1]]


def partner_pair_cols(rows):
    pairs = sorted(set(partner_pair(r) for r in rows), key=lambda s: sorted(s))[:-1]
    return [[1.0 if partner_pair(r) == p else 0.0 for r in rows] for p in pairs]


def run_cols(rows):
    rs = sorted(set(r["run"] for r in rows))[:-1]
    return [[1.0 if r["run"] == rn else 0.0 for r in rows] for rn in rs]


def report(rows, label):
    y = np.array([float(r["in_deg"]) for r in rows])
    persona = np.array([r["persona"] for r in rows])

    persona_mean = {p: y[persona == p].mean() for p in set(persona)}
    y_p = np.array([persona_mean[p] for p in persona])

    per_persona_means = np.array([persona_mean[p] for p in sorted(set(persona))])
    std_across = per_persona_means.std(ddof=1)
    std_within_pooled = (y - y_p).std(ddof=1)

    persona_r2 = r2(rows, persona_cols(rows))
    own_r2     = r2(rows, own_model_cols(rows))
    partner_r2 = r2(rows, partner_pair_cols(rows))
    selection_r2 = r2(rows, own_model_cols(rows) + partner_pair_cols(rows))
    run_r2     = r2(rows, run_cols(rows))

    print(f"\n=== {label} (n={len(rows)}, {len(set(persona))} personas) ===")
    print(f"  Engagement: mean in_deg = {y.mean():.2f}, total std = {y.std(ddof=1):.2f}")
    print(f"  std ACROSS personas (cross-persona means):       {std_across:.2f}")
    print(f"  std WITHIN persona (pooled around persona mean): {std_within_pooled:.2f}")
    print(f"  ratio within / across:                           {std_within_pooled / std_across:.2f}x")
    print()
    print(f"  Single-block R^2 (denominator = total variance):")
    print(f"    persona identity:                  {persona_r2*100:5.2f}%")
    print(f"    own model identity:                {own_r2*100:5.2f}%")
    print(f"    partner-pair indicators:           {partner_r2*100:5.2f}%")
    print(f"    model selection (own + partners):  {selection_r2*100:5.2f}%")
    print(f"    run identity (random baseline):    {run_r2*100:5.2f}%")
    print(f"  model selection / persona:           {selection_r2/persona_r2:.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "./runs"))
    args = ap.parse_args()

    rows = pv.load_agent_rows(Path(args.runs_dir))
    if not rows:
        print("No valid runs found.")
        return
    annotate_network(rows)

    report([r for r in rows if r["mode"] == "dyad"], "TWO-WAY")
    report([r for r in rows if r["mode"] == "tri"],  "THREE-WAY")
    report(rows, "COMBINED")


if __name__ == "__main__":
    main()
