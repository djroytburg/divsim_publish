"""
Smoke-test assertions for a completed divsim run.

Checks:
  1. The DB file exists and is non-empty.
  2. metadata.json sidecar exists and has an agent_model_map.
  3. Both model families have ≥1 comment.
  4. Total cross-family comments > 5.
  5. H matrix is computable, is a dict, and has finite values for all
     (src, dst) pairs covering both families.

Usage:
    python assert_h_matrix.py --db /path/to/sweep_*.db
    python assert_h_matrix.py --db /path/to/sweep_*.db --min-cross 5
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


SHORT = {
    'Qwen/Qwen3-32B':                 'Qwen',
    'openai/gpt-oss-20b':             'gpt-oss',
    'mistralai/Magistral-Small-2509': 'Mag',
    'zai-org/GLM-4-32B-0414':         'GLM-4',
    'google/gemma-4-31B-it':          'Gemma',
}


def compute_h(db_path: str) -> dict:
    """Compute directed pairwise H matrix from a run DB."""
    meta_path = db_path.replace('.db', '.metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    amm = meta.get('agent_model_map', {})
    if not amm:
        raise ValueError("agent_model_map is empty in metadata.json")

    uf = {int(k): SHORT.get(v['model'], v['model'][:8]) for k, v in amm.items()}
    fc = defaultdict(int)
    for fam in uf.values():
        fc[fam] += 1
    n = sum(fc.values())

    conn     = sqlite3.connect(db_path)
    comments = conn.execute(
        'SELECT c.user_id, p.user_id FROM comment c '
        'JOIN post p ON c.post_id=p.post_id WHERE c.user_id!=p.user_id'
    ).fetchall()
    conn.close()

    ec    = defaultdict(int)
    for cu, pu in comments:
        sf, df = uf.get(cu), uf.get(pu)
        if sf and df:
            ec[(sf, df)] += 1

    total = sum(ec.values())
    if total == 0:
        raise ValueError("No cross-family comment edges found.")

    H = {}
    for s in fc:
        for d in fc:
            obs = ec.get((s, d), 0) / total
            exp = (fc[s]/n) * ((fc[d] - (1 if s == d else 0)) / (n-1))
            if exp > 0:
                H[(s, d)] = obs/exp - 1
    return H, ec, fc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',        required=True, help='Path to sweep_*.db file.')
    parser.add_argument('--min-cross', type=int, default=5,
                        help='Minimum total cross-family comments required (default: 5).')
    args = parser.parse_args()

    db_path = args.db
    failures = []

    # 1. DB exists and is non-empty
    if not os.path.exists(db_path):
        print(f"FAIL: DB file does not exist: {db_path}")
        sys.exit(1)
    if os.path.getsize(db_path) < 1000:
        failures.append(f"DB file is too small ({os.path.getsize(db_path)} bytes)")

    # 2. Metadata exists
    meta_path = db_path.replace('.db', '.metadata.json')
    if not os.path.exists(meta_path):
        failures.append(f"metadata.json not found: {meta_path}")
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        if not meta.get('agent_model_map'):
            failures.append("agent_model_map is missing or empty in metadata.json")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)

    try:
        H, ec, fc = compute_h(db_path)
    except Exception as e:
        print(f"FAIL: H matrix computation failed: {e}")
        sys.exit(1)

    families = list(fc.keys())
    print(f"  Families: {families}")
    print(f"  Family sizes: {dict(fc)}")

    # 3. Both families have ≥1 comment
    comments_per_family = defaultdict(int)
    conn = sqlite3.connect(db_path)
    comments_all = conn.execute('SELECT user_id FROM comment').fetchall()
    conn.close()
    amm = meta.get('agent_model_map', {})
    uf  = {int(k): SHORT.get(v['model'], v['model'][:8]) for k, v in amm.items()}
    for (uid,) in comments_all:
        fam = uf.get(uid)
        if fam:
            comments_per_family[fam] += 1

    for fam in families:
        cnt = comments_per_family.get(fam, 0)
        if cnt < 1:
            failures.append(f"Family '{fam}' has 0 comments")
        else:
            print(f"  Family '{fam}': {cnt} comments emitted — OK")

    # 4. Cross-family comments > min_cross
    cross_total = sum(v for (s, d), v in ec.items() if s != d)
    print(f"  Cross-family comments: {cross_total} (min required: {args.min_cross})")
    if cross_total <= args.min_cross:
        failures.append(f"Cross-family comments {cross_total} ≤ {args.min_cross}")

    # 5. H matrix has finite values for all pairs
    print(f"  H matrix entries: {len(H)}")
    for pair, h in H.items():
        if not math.isfinite(h):
            failures.append(f"H[{pair}] is not finite: {h}")
        else:
            print(f"    H[{pair[0]}->{pair[1]}] = {h:+.4f}")

    if len(H) < len(families) * (len(families) - 1):
        failures.append(f"H matrix has fewer entries than expected "
                        f"({len(H)} < {len(families)*(len(families)-1)})")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll assertions passed.")
        sys.exit(0)


if __name__ == '__main__':
    main()
