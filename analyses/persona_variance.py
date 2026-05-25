"""
Variance decomposition: how much does base model identity explain per-agent
engagement, above and beyond persona identity and run-level effects?

Two complementary strategies:

  Strategy 1 — Hierarchical OLS
    Incrementally adds persona, run, and model dummy matrices to an OLS
    regression on in_deg (raw cross-comment count received). Reports R²
    per block and the incremental ΔR² from adding model after persona + run.

  Strategy 2 — Within-persona permutation test
    Computes the observed within-persona spread (std of per-model mean in_deg
    per persona). Compares to a null distribution from 10,000 permutations that
    shuffle model labels within each persona while preserving marginal model
    frequencies. p-value = fraction of null spreads ≥ observed.

Data source:
    All sweep_*.db files in RUNS_DIR that have a valid metadata.json sidecar.
    Excludes pilot runs (olmo, hermes, glm47).
    Validity: nc ≥ 200 and nu ≥ 18 (dyadic) or nu ≥ 26 (triadic).

Configuration:
    RUNS_DIR — directory containing sweep_*.db files (env var or --runs-dir)

Usage:
    python persona_variance.py
    python persona_variance.py --runs-dir /path/to/runs
    python persona_variance.py --dyadic-only
    python persona_variance.py --triadic-only
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


def load_agent_rows(runs_dir: Path, mode_filter: str | None = None) -> list[dict]:
    """Load one row per (agent, run) from all valid sweep DBs.

    Each row contains: persona (agent name), model, run_id, mode, in_deg,
    out_deg, n_posts.
    """
    rows = []
    for db in sorted(runs_dir.glob('sweep_*.db')):
        name = db.name
        if any(x in name for x in ['archive', 'olm', 'hermes', 'glm47', 'pilot_']):
            continue
        meta_path = str(db).replace('.db', '.metadata.json')
        if not Path(meta_path).exists():
            continue
        try:
            meta = json.load(open(meta_path))
        except Exception:
            continue
        amm = meta.get('agent_model_map', {})
        if not amm:
            continue
        mode = 'tri' if 'tri_' in name else 'dyad'
        if mode_filter and mode != mode_filter:
            continue
        uid_to_model = {int(k)+1: SHORT.get(v['model']) for k, v in amm.items()}
        if any(v is None for v in uid_to_model.values()):
            continue

        try:
            conn   = sqlite3.connect(db)
            nu     = conn.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
            nc     = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
            threshold = 26 if mode == 'tri' else 18
            if nu < threshold or nc < 200:
                conn.close()
                continue

            posts    = conn.execute("SELECT post_id, user_id FROM post").fetchall()
            comments = conn.execute("SELECT post_id, user_id FROM comment").fetchall()
            users    = conn.execute("SELECT user_id, user_name FROM user").fetchall()
            conn.close()
        except Exception:
            continue

        post_owner = {pid: uid for pid, uid in posts}
        in_deg  = defaultdict(int)
        out_deg = defaultdict(int)
        for pid, c in comments:
            owner = post_owner.get(pid)
            if owner and owner != c:
                in_deg[owner] += 1
                out_deg[c]    += 1
        n_posts_by = defaultdict(int)
        for _, uid in posts:
            n_posts_by[uid] += 1

        uid_to_name = {uid: uname for uid, uname in users}
        run_id = db.stem

        for uid, model_name in uid_to_model.items():
            rows.append({
                'persona':  uid_to_name.get(uid, f'u{uid}'),
                'model':    model_name,
                'run':      run_id,
                'mode':     mode,
                'in_deg':   in_deg.get(uid, 0),
                'out_deg':  out_deg.get(uid, 0),
                'n_posts':  n_posts_by.get(uid, 0),
            })
    return rows


def build_dummy_matrix(labels: list) -> np.ndarray:
    """One-hot encode a list of categorical labels."""
    unique = sorted(set(labels))
    idx    = {v: i for i, v in enumerate(unique)}
    X      = np.zeros((len(labels), len(unique)))
    for row_i, label in enumerate(labels):
        X[row_i, idx[label]] = 1.0
    return X


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS R² (in-sample)."""
    if X.shape[1] == 0:
        return 0.0
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        return float('nan')


def strategy1_ols(rows: list[dict], split_label: str):
    """Hierarchical OLS variance decomposition."""
    print(f"\n{'='*70}")
    print(f"Strategy 1 — Hierarchical OLS  [{split_label}]  N={len(rows)}")
    print(f"{'='*70}")

    y = np.array([float(r['in_deg']) for r in rows])

    personas = [r['persona'] for r in rows]
    models   = [r['model']   for r in rows]
    runs     = [r['run']     for r in rows]

    X_p = build_dummy_matrix(personas)
    X_m = build_dummy_matrix(models)
    X_r = build_dummy_matrix(runs)

    intercept = np.ones((len(y), 1))

    configs = {
        'persona only':              np.hstack([intercept, X_p]),
        'model only':                np.hstack([intercept, X_m]),
        'run only':                  np.hstack([intercept, X_r]),
        'persona + model':           np.hstack([intercept, X_p, X_m]),
        'persona + run':             np.hstack([intercept, X_p, X_r]),
        'persona + run + model (full)': np.hstack([intercept, X_p, X_r, X_m]),
    }

    results = {}
    for name, X in configs.items():
        r2 = ols_r2(X, y)
        results[name] = r2
        print(f"  {name:<35s}  R² = {r2:.4f}")

    delta_model = results.get('persona + run + model (full)', float('nan')) - \
                  results.get('persona + run', float('nan'))
    print(f"\n  Δmodel | persona, run = {delta_model:+.4f}")
    return results


def strategy2_permutation(rows: list[dict], split_label: str,
                           n_permutations: int = 10000, seed: int = 42):
    """Within-persona permutation test for model effect."""
    print(f"\n{'='*70}")
    print(f"Strategy 2 — Within-persona permutation test  [{split_label}]  "
          f"n_perm={n_permutations}")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)

    by_persona = defaultdict(list)
    for r in rows:
        by_persona[r['persona']].append((r['model'], r['in_deg']))

    personas_with_multi = {p: v for p, v in by_persona.items()
                           if len({m for m, _ in v}) >= 2}
    if not personas_with_multi:
        print("  No personas observed under ≥2 models — skipping.")
        return

    def within_persona_spread(persona_data: dict) -> float:
        """Mean across personas of std of per-model mean in_deg."""
        spreads = []
        for p, obs in persona_data.items():
            by_model = defaultdict(list)
            for m, deg in obs:
                by_model[m].append(deg)
            means = [np.mean(v) for v in by_model.values()]
            if len(means) >= 2:
                spreads.append(np.std(means, ddof=1))
        return np.mean(spreads) if spreads else 0.0

    observed = within_persona_spread(personas_with_multi)

    null_spreads = []
    for _ in range(n_permutations):
        shuffled = {}
        for p, obs in personas_with_multi.items():
            models = [m for m, _ in obs]
            degs   = [d for _, d in obs]
            rng.shuffle(models)
            shuffled[p] = list(zip(models, degs))
        null_spreads.append(within_persona_spread(shuffled))

    null_arr = np.array(null_spreads)
    p_val    = (null_arr >= observed).mean()
    ratio    = observed / null_arr.mean() if null_arr.mean() > 0 else float('nan')

    print(f"  Observed within-persona spread: {observed:.4f}")
    print(f"  Null spread: mean={null_arr.mean():.4f}  SD={null_arr.std():.4f}")
    print(f"  p-value: {p_val:.4f}  |  Observed/Null ratio: {ratio:.3f}×")
    return {'observed': observed, 'null_mean': null_arr.mean(),
            'null_sd': null_arr.std(), 'p_value': p_val, 'ratio': ratio}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',     default=os.environ.get("RUNS_DIR", "./runs"))
    parser.add_argument('--dyadic-only',  action='store_true')
    parser.add_argument('--triadic-only', action='store_true')
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    mode_filter = None
    if args.dyadic_only:
        mode_filter = 'dyad'
    elif args.triadic_only:
        mode_filter = 'tri'

    print(f"Loading agent rows from {runs_dir} ...")
    all_rows = load_agent_rows(runs_dir, mode_filter=None)
    if not all_rows:
        print("No valid runs found. Check RUNS_DIR and that metadata.json files exist.")
        return

    dyad_rows = [r for r in all_rows if r['mode'] == 'dyad']
    tri_rows  = [r for r in all_rows if r['mode'] == 'tri']
    comb_rows = all_rows

    print(f"Loaded: {len(dyad_rows)} dyadic rows, {len(tri_rows)} triadic rows, "
          f"{len(comb_rows)} combined.")

    splits = []
    if not args.triadic_only:
        splits.append(('DYADS', dyad_rows))
    if not args.dyadic_only:
        splits.append(('TRIADS', tri_rows))
    if not args.dyadic_only and not args.triadic_only:
        splits.append(('COMBINED', comb_rows))

    for split_label, rows in splits:
        if not rows:
            print(f"\nNo rows for split {split_label} — skipping.")
            continue
        strategy1_ols(rows, split_label)
        strategy2_permutation(rows, split_label)


if __name__ == '__main__':
    main()
