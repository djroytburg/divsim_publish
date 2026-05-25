"""
Leave-one-model-out robustness of the variance decomposition.

For each base model in the five-model pool, we drop every run containing
that model and re-estimate the variance decomposition on the remaining
runs. The procedure isolates which models the headline parity finding
(model selection vs. persona identity in three-way mixtures) actually
depends on.

Reported per regime (two-way, three-way):
  - persona-only R^2 on raw in-degree (single-block OLS)
  - own-model-only R^2 (five model dummies, or four when one is removed)
  - model-selection R^2 (own model + unordered partner-pair indicators)
  - run-identity R^2 (random baseline)
  - within-persona permutation ratio (Observed / Null mean)

Usage:
  python leave_one_model_out.py [--runs-dir PATH]
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
    if not cols or not rows:
        return float("nan")
    y = np.array([float(r["in_deg"]) for r in rows])
    X = np.column_stack([np.ones(len(rows))] + list(cols))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    if y.var() == 0:
        return 0.0
    return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def persona_cols(rows):
    ps = sorted(set(r["persona"] for r in rows))[:-1]
    return [[1.0 if r["persona"] == p else 0.0 for r in rows] for p in ps]


def own_model_cols(rows):
    ms = sorted(set(r["model"] for r in rows))[:-1]
    return [[1.0 if r["model"] == m else 0.0 for r in rows] for m in ms]


def partner_pair_cols(rows):
    pairs = sorted(set(partner_pair(r) for r in rows), key=lambda s: sorted(s))[:-1]
    return [[1.0 if partner_pair(r) == p else 0.0 for r in rows] for p in pairs]


def run_cols(rows):
    rs = sorted(set(r["run"] for r in rows))[:-1]
    return [[1.0 if r["run"] == rn else 0.0 for r in rows] for rn in rs]


def perm_ratio(rows, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    by_persona = defaultdict(list)
    for r in rows:
        by_persona[r["persona"]].append((r["model"], float(r["in_deg"])))
    multi = {p: v for p, v in by_persona.items() if len({m for m, _ in v}) >= 2}
    if not multi:
        return float("nan")

    def spread(d):
        s = []
        for _, obs in d.items():
            bm = defaultdict(list)
            for m, val in obs:
                bm[m].append(val)
            means = [np.mean(v) for v in bm.values()]
            if len(means) >= 2:
                s.append(np.std(means, ddof=1))
        return np.mean(s) if s else 0.0

    observed = spread(multi)
    nulls = []
    for _ in range(n_perm):
        shuffled = {}
        for p, vs in multi.items():
            ms = [m for m, _ in vs]
            ds = [d for _, d in vs]
            rng.shuffle(ms)
            shuffled[p] = list(zip(ms, ds))
        nulls.append(spread(shuffled))
    null_mean = np.mean(nulls)
    return observed / null_mean if null_mean > 0 else float("nan")


def block(rows, label):
    if not rows:
        print(f"  {label:12s}  (no runs)")
        return
    pr = r2(rows, persona_cols(rows))
    own = r2(rows, own_model_cols(rows))
    sel = r2(rows, own_model_cols(rows) + partner_pair_cols(rows))
    run_ = r2(rows, run_cols(rows))
    ratio = perm_ratio(rows)
    print(f"  {label:12s} n={len(rows):4d}  persona={pr*100:5.2f}%  "
          f"own={own*100:5.2f}%  selection={sel*100:5.2f}%  "
          f"run={run_*100:5.2f}%  perm={ratio:.3f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "./runs"))
    args = ap.parse_args()

    rows = pv.load_agent_rows(Path(args.runs_dir))
    if not rows:
        print("No valid runs found.")
        return
    annotate_network(rows)

    print("LEAVE-ONE-MODEL-OUT VARIANCE DECOMPOSITION")
    print("=" * 78)
    print("\nBaseline (all five models):")
    block([r for r in rows if r["mode"] == "dyad"], "Two-way")
    block([r for r in rows if r["mode"] == "tri"],  "Three-way")

    for drop in MODELS:
        kept = [r for r in rows if drop not in r["network"]]
        print(f"\nDropping all runs containing {drop} (n={len(kept)} agent-runs remaining):")
        block([r for r in kept if r["mode"] == "dyad"], "Two-way")
        block([r for r in kept if r["mode"] == "tri"],  "Three-way")


if __name__ == "__main__":
    main()
