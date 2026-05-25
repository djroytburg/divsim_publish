"""
Triadic (3-model) sweep for the divsim paper, with preemption support.

Enumerates all C(5,3)=10 model triplets, samples 3 of the 6 possible orderings
per triplet (the 3 with maximum positional diversity), crossed with 3 seeds.
Total: 10 triplets × 3 orderings × 3 seeds = 90 configurations.

Preemption support: run_v9.py catches SIGTERM and exits cleanly after the
current step commit. This script checks for partial DBs and resumes them via
--resume rather than restarting from scratch.

Configuration is read from environment variables:
    RUNS_DIR    — output directory for .db files (required)
    VENV_PATH   — path to the Python venv with oasis installed
    OASIS_RUNNER — path to run_v9.py (defaults to <this_dir>/../oasis/run_v9.py)
    LOGS_DIR    — directory for per-run log files

Usage:
    python sweep_triadic.py --dry-run              # print all configs, no launch
    python sweep_triadic.py --max-parallel 2       # limit concurrent runs
    python sweep_triadic.py --triplets qwen,oss,mag  # run a specific triplet only
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

SEEDS   = [42, 271, 999]
STEPS   = 100
N_AGENTS = 30  # 10 per group


def sampled_orderings(triplet: tuple) -> list:
    """Return 3 of the 6 possible permutations, evenly spaced for positional diversity."""
    perms = list(itertools.permutations(triplet))
    return [perms[0], perms[2], perms[4]]


def run_id(perm: tuple, seed: int) -> str:
    return f"sweep_tri_{'_'.join(m[:3] for m in perm)}_s{seed}"


def db_is_valid(db: Path) -> bool:
    """Return True if the DB has enough active agents to be a complete triadic run."""
    if not db.exists():
        return False
    try:
        import sqlite3
        conn = sqlite3.connect(db)
        nu = conn.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
        conn.close()
        return nu >= 26
    except Exception:
        return False


def launch(perm: tuple, seed: int, dry_run: bool) -> dict | None:
    rid = run_id(perm, seed)
    db  = RUNS_DIR / f"{rid}.db"
    log = LOGS_DIR / f"{rid}.log"

    if db_is_valid(db):
        print(f"  [skip] {rid} (valid DB exists)")
        return None

    resume = db.exists() and db.stat().st_size > 50_000

    ma, mb, mc = MODELS[perm[0]], MODELS[perm[1]], MODELS[perm[2]]
    env = {
        **os.environ,
        "VLLM_URL_A":   ma[1], "MODEL_NAME_A": ma[0],
        "VLLM_URL_B":   mb[1], "MODEL_NAME_B": mb[0],
        "VLLM_URL_C":   mc[1], "MODEL_NAME_C": mc[0],
        "OASIS_DB_PATH": str(db),
    }
    cmd = [
        f"{VENV}/bin/python3", str(RUNNER),
        "--mode",         "hetero3",
        "--steps",        str(STEPS),
        "--agents",       str(N_AGENTS),
        "--seed-posts",   str(N_AGENTS),
        "--activate-prob", "0.3",
        "--rng-seed",     str(seed),
        "--db",           str(db),
    ]
    if resume:
        cmd.append("--resume")

    status = "[resume]" if resume else "[run]   "
    print(f"  {status} {rid}")

    if dry_run:
        print(f"           A={perm[0]}  B={perm[1]}  C={perm[2]}  seed={seed}")
        return None

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
    return {"rid": rid, "proc": proc}


def all_triplets():
    return list(itertools.combinations(MODELS.keys(), 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print configs without launching.")
    parser.add_argument("--max-parallel", type=int, default=2,
                        help="Maximum concurrent runs.")
    parser.add_argument("--triplets",     nargs="*",
                        help="Restrict to specific triplets, e.g. qwen,oss,mag")
    args = parser.parse_args()

    triplets = all_triplets()
    if args.triplets:
        triplets = [tuple(t.split(",")) for t in args.triplets]

    configs = [
        (perm, seed)
        for triplet in triplets
        for perm in sampled_orderings(triplet)
        for seed in SEEDS
    ]

    print(f"Triadic sweep: {len(triplets)} triplets × 3 orderings × "
          f"{len(SEEDS)} seeds = {len(configs)} configs")
    print(f"Triplets: {['+'.join(t) for t in triplets]}")
    print()

    active = []
    for perm, seed in configs:
        active = [r for r in active if r["proc"].poll() is None]
        while len(active) >= args.max_parallel:
            time.sleep(15)
            active = [r for r in active if r["proc"].poll() is None]

        r = launch(perm, seed, args.dry_run)
        if r:
            active.append(r)
        if not args.dry_run:
            time.sleep(2)

    for r in active:
        r["proc"].wait()
    print("All triadic configs done.")


if __name__ == "__main__":
    main()
