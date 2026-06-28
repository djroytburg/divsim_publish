"""
Extract all post + comment text from every valid sweep DB and tag each row
with (model, persona, run, mode, kind). Writes a single JSONL.

Validity criterion matches persona_variance: nc>=200 & nu>=26 (for tri) or
nu>=18 (dyad), and metadata + good model map.
"""
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path("runs")
OUT = Path("data/texts.jsonl")

SHORT = {
    "Qwen/Qwen3-32B": "Qwen",
    "openai/gpt-oss-20b": "gpt-oss",
    "mistralai/Magistral-Small-2509": "Mag",
    "zai-org/GLM-4-32B-0414": "GLM-4",
    "google/gemma-4-31B-it": "Gemma",
}


def load_valid_runs():
    rows = []
    n_db = n_valid = 0
    for db in sorted(RUNS.glob("sweep_*.db")):
        name = db.name
        if any(x in name for x in ["archive", "snapshot", "olm", "hermes", "qwq", "glm47", "pilot_"]):
            continue
        meta_p = str(db).replace(".db", ".metadata.json")
        if not Path(meta_p).exists():
            continue
        n_db += 1
        try:
            meta = json.load(open(meta_p))
        except Exception:
            continue
        amm = meta.get("agent_model_map", {})
        if not amm:
            continue
        uid_to_model = {int(k) + 1: SHORT.get(v["model"]) for k, v in amm.items()}
        if any(v is None for v in uid_to_model.values()):
            continue
        mode = "tri" if "tri_" in name else "dyad"
        threshold = 26 if mode == "tri" else 18
        try:
            c = sqlite3.connect(db)
            nu = c.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
            nc = c.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
            if nu < threshold or nc < 200:
                c.close()
                continue
            users = {uid: uname for uid, uname in c.execute("SELECT user_id, user_name FROM user").fetchall()}
            posts = c.execute("SELECT post_id, user_id, content FROM post").fetchall()
            comments = c.execute("SELECT comment_id, user_id, content FROM comment").fetchall()
            c.close()
        except Exception as e:
            print(f"err {name}: {e}", file=sys.stderr)
            continue
        n_valid += 1
        run_id = db.stem
        for pid, uid, content in posts:
            if not content:
                continue
            rows.append({
                "run": run_id, "mode": mode, "kind": "post",
                "model": uid_to_model.get(uid), "persona": users.get(uid, f"u{uid}"),
                "text": content,
            })
        for cid, uid, content in comments:
            if not content:
                continue
            rows.append({
                "run": run_id, "mode": mode, "kind": "comment",
                "model": uid_to_model.get(uid), "persona": users.get(uid, f"u{uid}"),
                "text": content,
            })
    return rows, n_db, n_valid


def main():
    rows, n_db, n_valid = load_valid_runs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_model = defaultdict(lambda: [0, 0])
    by_kind = defaultdict(int)
    for r in rows:
        by_model[r["model"]][0 if r["kind"] == "post" else 1] += 1
        by_kind[r["kind"]] += 1
    print(f"DBs scanned: {n_db}  valid: {n_valid}  rows: {len(rows):,}")
    print(f"by kind: {dict(by_kind)}")
    print("by model (posts, comments):")
    for m, (p, cm) in sorted(by_model.items()):
        print(f"  {m:10s} posts={p:6d}  comments={cm:6d}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
