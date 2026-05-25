"""
Dyadic (2-model) combinatorial sweep for the divsim paper.

Enumerates all ordered pairs from the 5 paper models, crossed with 5 persona seeds,
for a total of C(5,2)×2×5 = 100 configurations. Each configuration launches one
run of the OASIS simulator in hetero mode.

Configuration is read from environment variables:
    RUNS_DIR    — output directory for .db files (required)
    VENV_PATH   — path to the Python venv with oasis installed
    OASIS_RUNNER — path to run_v9.py (defaults to <this_dir>/../oasis/run_v9.py)
    LOGS_DIR    — directory for per-run log files

Usage:
    python sweep_dyadic.py --dry-run              # print all configs, no launch
    python sweep_dyadic.py --mode 2model          # all 2-model runs (default)
    python sweep_dyadic.py --max-parallel 2       # limit concurrent runs
    python sweep_dyadic.py --skip-existing        # skip runs whose DB already exists
"""

import argparse
import itertools
import os
import subprocess
import time
from pathlib import Path


RUNS_DIR = Path(os.environ.get("RUNS_DIR", "./runs"))
VENV     = os.environ.get("VENV_PATH", str(Path(__file__).parent.parent / ".venv"))
RUNNER   = Path(os.environ.get("OASIS_RUNNER",
                               str(Path(__file__).parent.parent / "oasis" / "run_v9.py")))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", "./logs/sweep"))

MODELS = {
    "qwen":   ("Qwen/Qwen3-32B",                   "http://localhost:8002/v1"),
    "oss":    ("openai/gpt-oss-20b",               "http://localhost:8003/v1"),
    "mag":    ("mistralai/Magistral-Small-2509",    "http://localhost:8004/v1"),
    "glm":    ("zai-org/GLM-4-32B-0414",           "http://localhost:8007/v1"),
    "gem":    ("google/gemma-4-31B-it",             "http://localhost:8009/v1"),
}

MODEL_KEYS = list(MODELS.keys())
SEEDS  = [42, 137, 271, 314, 999]
STEPS  = 100
N_AGENTS = 20  # 10 per group


def run_id(a: str, b: str, seed: int) -> str:
    return f"sweep_{a[:3]}_{b[:3]}_s{seed}"


def db_is_valid(db: Path) -> bool:
    """Return True if the DB has enough data to be considered a complete run."""
    if not db.exists():
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(db)
        nu = conn.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
        nc = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
        conn.close()
        return nu >= 18 and nc >= 200
    except Exception:
        return False


def launch(a: str, b: str, seed: int, dry_run: bool) -> dict:
    rid = run_id(a, b, seed)
    db  = RUNS_DIR / f"{rid}.db"
    log = LOGS_DIR / f"{rid}.log"
    ma, mb = MODELS[a], MODELS[b]

    env = {
        **os.environ,
        "VLLM_URL_A":   ma[1], "MODEL_NAME_A": ma[0],
        "VLLM_URL_B":   mb[1], "MODEL_NAME_B": mb[0],
        "OASIS_DB_PATH": str(db),
    }
    cmd = [
        f"{VENV}/bin/python3", str(RUNNER),
        "--mode",         "hetero",
        "--steps",        str(STEPS),
        "--agents",       str(N_AGENTS),
        "--seed-posts",   str(N_AGENTS),
        "--activate-prob", "0.3",
        "--rng-seed",     str(seed),
        "--db",           str(db),
    ]

    if dry_run:
        print(f"  [DRY] {rid}  A={a}({ma[0]})  B={b}({mb[0]})  seed={seed}")
        return {"rid": rid, "proc": None}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
    print(f"  [{rid}] PID={proc.pid}  A={a}  B={b}  seed={seed}")
    return {"rid": rid, "proc": proc}


def all_configs():
    configs = []
    for pair in itertools.combinations(MODEL_KEYS, 2):
        for perm in itertools.permutations(pair):
            for seed in SEEDS:
                configs.append((perm[0], perm[1], seed))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print configs without launching.")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Maximum concurrent runs.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip runs whose DB already exists and is valid.")
    args = parser.parse_args()

    configs = all_configs()
    print(f"Dyadic sweep: {len(configs)} configs "
          f"(C(5,2)×2 ordered pairs × {len(SEEDS)} seeds)")
    print(f"Seeds: {SEEDS}  max_parallel: {args.max_parallel}  "
          f"dry_run: {args.dry_run}")

    active = []
    for a, b, seed in configs:
        active = [r for r in active if r["proc"] and r["proc"].poll() is None]
        while len(active) >= args.max_parallel:
            time.sleep(10)
            active = [r for r in active if r["proc"] and r["proc"].poll() is None]

        if args.skip_existing:
            db = RUNS_DIR / f"{run_id(a, b, seed)}.db"
            if db_is_valid(db):
                print(f"  [skip] {run_id(a, b, seed)} (valid DB exists)")
                continue

        r = launch(a, b, seed, args.dry_run)
        if r["proc"]:
            active.append(r)
        time.sleep(2)

    for r in active:
        if r["proc"]:
            r["proc"].wait()
    print("All dyadic configs done.")


if __name__ == "__main__":
    main()
