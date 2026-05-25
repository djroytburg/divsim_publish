"""
H temporal dynamics analysis: decay, rotation stability, seed stability.

Splits each simulation's comment history into quartiles by comment_id and
computes directed-pair homophily H_{src->dst} per quartile. Aggregates across
runs to characterize each model's temporal signature (stable attractor, transient
attractor, delayed attractor, repeller), and measures rotation/seed stability
of the temporal trajectories.

Configuration:
    RUNS_DIR — directory containing sweep_*.db files (env var or --runs-dir)

Usage:
    python temporal_dynamics.py                    # dyadic decay (all targets)
    python temporal_dynamics.py --triadic          # triadic decay
    python temporal_dynamics.py --target gpt-oss   # specific target only
    python temporal_dynamics.py --runs-dir /path/to/runs
"""

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


SHORT = {
    'Qwen/Qwen3-32B':                 'Qwen',
    'openai/gpt-oss-20b':             'gpt-oss',
    'mistralai/Magistral-Small-2509': 'Mag',
    'zai-org/GLM-4-32B-0414':         'GLM-4',
    'google/gemma-4-31B-it':          'Gemma',
}

ALL_TARGETS = ['gpt-oss', 'Qwen', 'Mag', 'GLM-4', 'Gemma']


def H_in_window(db_path: Path, cmin: int, cmax: int) -> dict | None:
    """Compute H for comments with comment_id in [cmin, cmax)."""
    meta = str(db_path).replace('.db', '.metadata.json')
    if not Path(meta).exists():
        return None
    try:
        amm = json.load(open(meta)).get('agent_model_map', {})
        uf  = {int(k)+1: SHORT.get(v['model'], '?') for k, v in amm.items()}
    except Exception:
        return None
    fc = defaultdict(int)
    for fam in uf.values():
        fc[fam] += 1
    n = sum(fc.values())
    conn = sqlite3.connect(db_path)
    comments = conn.execute(
        f'SELECT c.user_id, p.user_id FROM comment c JOIN post p ON c.post_id=p.post_id '
        f'WHERE c.user_id!=p.user_id AND c.comment_id >= {cmin} AND c.comment_id < {cmax}'
    ).fetchall()
    conn.close()
    ec    = defaultdict(int)
    for cu, pu in comments:
        sf, df = uf.get(cu), uf.get(pu)
        if sf and df:
            ec[(sf, df)] += 1
    total = sum(ec.values())
    if total < 5:
        return None
    H = {}
    for s in fc:
        for d in fc:
            exp = (fc[s]/n) * ((fc[d] - (1 if s == d else 0)) / (n-1))
            if exp > 0:
                H[(s, d)] = ec.get((s, d), 0) / total / exp - 1
    return H


def trajectory(db_path: Path, target: str, n_bins: int = 4) -> list | None:
    """Return list of incoming-to-target mean H per quartile."""
    conn   = sqlite3.connect(db_path)
    max_id = conn.execute('SELECT MAX(comment_id) FROM comment').fetchone()[0]
    conn.close()
    if max_id is None or max_id < 40:
        return None
    bs   = max_id // n_bins
    traj = []
    for i in range(n_bins):
        lo = i*bs + 1
        hi = (i+1)*bs + 1 if i < n_bins-1 else max_id + 1
        H  = H_in_window(db_path, lo, hi)
        if H is None:
            traj.append(None)
            continue
        inc = [v for (s, d), v in H.items() if d == target and s != target]
        traj.append(np.mean(inc) if inc else None)
    return traj


def discover_dyadic_runs(runs_dir: Path, target: str | None = None) -> list:
    """Return list of (db_path, group_A_model, partner, seed) for dyadic runs."""
    out = []
    for db in runs_dir.glob('sweep_*.db'):
        if 'tri_' in db.name or 'pilot_' in db.name:
            continue
        meta = str(db).replace('.db', '.metadata.json')
        if not Path(meta).exists():
            continue
        try:
            amm      = json.load(open(meta)).get('agent_model_map', {})
            uid_model = {int(k): SHORT.get(v['model']) for k, v in amm.items()}
        except Exception:
            continue
        if any(v is None for v in uid_model.values()):
            continue
        fams = set(uid_model.values())
        if len(fams) != 2:
            continue
        if target and target not in fams:
            continue
        group_A = uid_model.get(0)
        seed    = next((s for s in [42, 137, 271, 314, 999] if f'_s{s}.' in db.name), None)
        if seed is None:
            continue
        partner = (fams - {group_A}).pop()
        out.append((str(db), group_A, partner, seed))
    return out


def discover_triadic_runs(runs_dir: Path, target: str | None = None) -> list:
    """Return list of (db_path, group_A_model, triplet_tuple, seed) for triadic runs."""
    out = []
    for db in runs_dir.glob('sweep_tri_*.db'):
        meta = str(db).replace('.db', '.metadata.json')
        if not Path(meta).exists():
            continue
        try:
            amm       = json.load(open(meta)).get('agent_model_map', {})
            uid_model = {int(k): SHORT.get(v['model']) for k, v in amm.items()}
        except Exception:
            continue
        if any(v is None for v in uid_model.values()):
            continue
        fams = sorted(set(uid_model.values()))
        if len(fams) != 3:
            continue
        if target and target not in fams:
            continue
        group_A = uid_model.get(0)
        seed    = next((s for s in [42, 271, 999] if f'_s{s}.' in db.name), None)
        if seed is None:
            continue
        out.append((str(db), group_A, tuple(fams), seed))
    return out


def dyadic_stability_report(runs_dir: Path, target: str):
    """Per-pair Q0..Q3 trajectory + rotation + seed stability for a target family."""
    runs = discover_dyadic_runs(runs_dir, target=target)
    by_pair_ord = defaultdict(lambda: defaultdict(list))
    for db, group_A, partner, seed in runs:
        traj = trajectory(Path(db), target)
        if traj is None or any(v is None for v in traj):
            continue
        ordering = 'target_A' if group_A == target else 'partner_A'
        by_pair_ord[partner][ordering].append((seed, traj))

    print(f"\n{'='*78}")
    print(f"DYADIC stability for incoming-to-{target}")
    print(f"{'='*78}")
    for partner in sorted(by_pair_ord):
        print(f"\n  {target} x {partner}")
        order_means = {}
        for ord_name, lst in by_pair_ord[partner].items():
            if not lst:
                continue
            trajs = np.array([t for _, t in lst])
            order_means[ord_name] = trajs.mean(axis=0)
            print(f"    [{ord_name}] n={len(trajs)} seeds | "
                  f"Q0..Q3: {' '.join(f'{v:+.2f}' for v in trajs.mean(axis=0))} | "
                  f"seed-SD: {' '.join(f'{s:.2f}' for s in trajs.std(axis=0))}")
        if len(order_means) == 2:
            keys    = list(order_means.keys())
            spreads = [abs(order_means[keys[0]][q] - order_means[keys[1]][q]) for q in range(4)]
            print(f"    rotation spread per Q: {' '.join(f'{s:.2f}' for s in spreads)}  "
                  f"max={max(spreads):.2f}")


def triadic_stability_report(runs_dir: Path, target: str):
    """Per-triplet Q0..Q3 trajectory + rotation + seed stability for a target family."""
    runs = discover_triadic_runs(runs_dir, target=target)
    by_trip_ord = defaultdict(lambda: defaultdict(list))
    for db, group_A, trip, seed in runs:
        if target not in trip:
            continue
        traj = trajectory(Path(db), target)
        if traj is None or any(v is None for v in traj):
            continue
        by_trip_ord[trip][group_A].append((seed, traj))

    print(f"\n{'='*78}")
    print(f"TRIADIC stability for incoming-to-{target}")
    print(f"{'='*78}")
    for trip in sorted(by_trip_ord):
        if sum(len(v) for v in by_trip_ord[trip].values()) < 3:
            continue
        print(f"\n  Triplet: {' + '.join(trip)}")
        per_ord_mean = {}
        for grpA, lst in by_trip_ord[trip].items():
            if not lst:
                continue
            trajs = np.array([t for _, t in lst])
            per_ord_mean[grpA] = trajs.mean(axis=0)
            print(f"    [group-A={grpA}] n={len(trajs)} seeds | "
                  f"Q0..Q3: {' '.join(f'{v:+.2f}' for v in trajs.mean(axis=0))} | "
                  f"seed-SD: {' '.join(f'{s:.2f}' for s in trajs.std(axis=0))}")
        if len(per_ord_mean) >= 2:
            spreads = [max(t[q] for t in per_ord_mean.values()) -
                       min(t[q] for t in per_ord_mean.values())
                       for q in range(4)]
            print(f"    rotation spread per Q: {' '.join(f'{s:.2f}' for s in spreads)}  "
                  f"max={max(spreads):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', default=os.environ.get("RUNS_DIR", "./runs"),
                        help="Directory containing sweep_*.db files.")
    parser.add_argument('--triadic',  action='store_true',
                        help="Run triadic stability report instead of dyadic.")
    parser.add_argument('--target',   default=None,
                        help="Restrict to one model family (e.g. gpt-oss). Default: all.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    targets  = [args.target] if args.target else ALL_TARGETS

    for t in targets:
        if args.triadic:
            triadic_stability_report(runs_dir, t)
        else:
            dyadic_stability_report(runs_dir, t)


if __name__ == '__main__':
    main()
