"""
Heatmap figures for the divsim paper.

Produces H-matrix, swap-stability, in-group self-preference, and Elo-rating
figures in three variants each: dyadic-only, triadic-only, combined.

Figures are written to OUTDIR (default: same directory as this script).

Configuration:
    RUNS_DIR   — directory containing sweep_*.db files (env var or --runs-dir)
    OUTDIR     — output directory for .png files (env var or --outdir)

Usage:
    python plot_matrices.py
    python plot_matrices.py --runs-dir /path/to/runs --outdir /path/to/figures
"""

import os
import glob
import json
import math
import sqlite3
import itertools
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, fontManager


# ── Font loading ──────────────────────────────────────────────────────────────
# Looks for Volkhov and Ubuntu Mono in ~/fonts; falls back gracefully.
FONT_DIR = Path.home() / "fonts"
if FONT_DIR.is_dir():
    for ttf in FONT_DIR.rglob("*.ttf"):
        fontManager.addfont(str(ttf))


def fp(family, size, bold=False):
    return FontProperties(family=family, size=size,
                          weight="bold" if bold else "normal")


FT = lambda s, bold=False: fp("Volkhov",     s, bold)
FB = lambda s, bold=False: fp("Ubuntu Mono", s, bold)

plt.rcParams.update({
    "font.family":       "Ubuntu Mono",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
})

SHORT = {
    "Qwen/Qwen3-32B":                 "Qwen",
    "openai/gpt-oss-20b":             "gpt-oss",
    "mistralai/Magistral-Small-2509": "Mag",
    "zai-org/GLM-4-32B-0414":         "GLM-4",
    "google/gemma-4-31B-it":          "Gemma",
}
MODELS    = ["GLM-4", "Mag", "Qwen", "gpt-oss", "Gemma"]
MODEL_SET = set(MODELS)


# ── H computation ──────────────────────────────────────────────────────────────

def load_h(db_path):
    """Compute directed pairwise H matrix for one run DB."""
    mp = str(db_path).replace(".db", ".metadata.json")
    if not os.path.exists(mp):
        return {}
    with open(mp) as f:
        meta = json.load(f)
    amm = meta.get("agent_model_map", {})
    if not amm:
        return {}
    uf = {int(k): SHORT.get(v["model"], v["model"][:6]) for k, v in amm.items()}
    fc = defaultdict(int)
    for fam in uf.values():
        fc[fam] += 1
    ec = defaultdict(int)
    conn = sqlite3.connect(db_path)
    for cu, pu in conn.execute(
            "SELECT c.user_id, p.user_id FROM comment c "
            "JOIN post p ON c.post_id=p.post_id WHERE c.user_id!=p.user_id"):
        sf, df = uf.get(cu), uf.get(pu)
        if sf and df:
            ec[(sf, df)] += 1
    conn.close()
    total = sum(ec.values())
    if not total:
        return {}
    n = sum(fc.values())
    H = {}
    for s in fc:
        for d in fc:
            obs = ec.get((s, d), 0) / total
            exp = (fc[s]/n) * ((fc[d] - (1 if s == d else 0)) / (n-1))
            if exp > 0:
                H[(s, d)] = obs/exp - 1
    return H


# ── Load all valid dyadic runs ────────────────────────────────────────────────

def load_dyadic(runs_dir: Path):
    all_H     = defaultdict(list)
    swap_vals = defaultdict(lambda: defaultdict(list))
    for db in sorted(runs_dir.glob("sweep_*.db")):
        name = str(db)
        if any(x in name for x in ["archive", "snapshot", "sweep_tri_",
                                    "olm", "hermes", "qwq", "glm47"]):
            continue
        try:
            mp = name.replace(".db", ".metadata.json")
            if not os.path.exists(mp):
                continue
            with open(mp) as f:
                meta = json.load(f)
            amm  = meta.get("agent_model_map", {})
            fams = set(SHORT.get(v["model"], v["model"][:6]) for v in amm.values())
            if len(fams) != 2 or not fams.issubset(MODEL_SET):
                continue
            conn = sqlite3.connect(db)
            nu = conn.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
            nc = conn.execute("SELECT COUNT(*) FROM comment").fetchone()[0]
            conn.close()
            if nu < 18 or nc < 200:
                continue
            H = load_h(db)
            seed     = str(db).split("_s")[-1].replace(".db", "")
            dyad_key = frozenset(fams)
            for pair, h in H.items():
                all_H[pair].append(h)
                swap_vals[pair][(dyad_key, seed)].append(h)
        except Exception:
            pass
    return all_H, swap_vals


def load_triadic(runs_dir: Path):
    all_H           = defaultdict(list)
    by_triplet_seed = defaultdict(list)
    for db in sorted(runs_dir.glob("sweep_tri_*.db")):
        if "archive" in str(db):
            continue
        try:
            conn = sqlite3.connect(db)
            nu = conn.execute("SELECT COUNT(DISTINCT user_id) FROM comment").fetchone()[0]
            conn.close()
            if nu < 26:
                continue
            mp = str(db).replace(".db", ".metadata.json")
            if not os.path.exists(mp):
                continue
            H = load_h(db)
            if not H:
                continue
            with open(mp) as f:
                meta = json.load(f)
            amm = meta.get("agent_model_map", {})
            run_models = frozenset(
                SHORT.get(v["model"], v["model"][:6]) for v in amm.values()
            )
            seed = str(db).split("_s")[-1].replace(".db", "")
            by_triplet_seed[(run_models, seed)].append(H)
            for pair, h in H.items():
                all_H[pair].append(h)
        except Exception:
            pass
    return all_H, by_triplet_seed


# ── Swap-stability matrices ───────────────────────────────────────────────────

def dyadic_swap_matrix(swap_vals, keys):
    """Mean |ΔH| across seeds for dyadic runs (exactly 2 orientations per seed)."""
    n    = len(keys)
    mean = np.full((n, n), np.nan)
    std  = np.full((n, n), np.nan)
    for i, s in enumerate(keys):
        for j, d in enumerate(keys):
            by_seed = swap_vals.get((s, d), {})
            diffs   = [abs(vs[0]-vs[1]) for vs in by_seed.values() if len(vs) == 2]
            if diffs:
                mean[i, j] = np.mean(diffs)
                std[i, j]  = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.
    return mean, std


def triadic_swap_matrix(tri_by_ts, keys):
    """Mean max-|ΔH| per (triplet, seed) group for triadic runs (3 orderings)."""
    n             = len(keys)
    pair_maxdiffs = defaultdict(list)
    for (triplet, seed), h_list in tri_by_ts.items():
        if len(h_list) < 2:
            continue
        all_pairs = set()
        for H in h_list:
            all_pairs.update(H.keys())
        for pair in all_pairs:
            vals = [H.get(pair, np.nan) for H in h_list]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) < 2:
                continue
            max_diff = max(abs(a-b) for a, b in itertools.combinations(vals, 2))
            pair_maxdiffs[pair].append(max_diff)
    mean = np.full((n, n), np.nan)
    std  = np.full((n, n), np.nan)
    for i, s in enumerate(keys):
        for j, d in enumerate(keys):
            diffs = pair_maxdiffs.get((s, d), [])
            if diffs:
                mean[i, j] = np.mean(diffs)
                std[i, j]  = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.
    return mean, std


def combined_swap_matrix(swap_dy_mean, swap_dy_std, swap_tri_mean, swap_tri_std, keys):
    """Unweighted average of dyadic and triadic swap means where both exist."""
    n    = len(keys)
    mean = np.full((n, n), np.nan)
    std  = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            vals = []
            if not np.isnan(swap_dy_mean[i, j]):
                vals.append(swap_dy_mean[i, j])
            if not np.isnan(swap_tri_mean[i, j]):
                vals.append(swap_tri_mean[i, j])
            if vals:
                mean[i, j] = np.mean(vals)
                std[i, j]  = np.std(vals, ddof=0) if len(vals) > 1 else 0.
    return mean, std


# ── Per-matrix statistics ─────────────────────────────────────────────────────

def mat_stats(H_dict, keys):
    n    = len(keys)
    mean = np.full((n, n), np.nan)
    std  = np.full((n, n), np.nan)
    for i, s in enumerate(keys):
        for j, d in enumerate(keys):
            vals = H_dict.get((s, d), [])
            if vals:
                mean[i, j] = np.mean(vals)
                std[i, j]  = np.std(vals, ddof=1) if len(vals) > 1 else 0.
    return mean, std


# ── Cell annotation helpers ───────────────────────────────────────────────────

def label_cells(ax, mean, std, swap_mean=None, swap_std=None,
                skip_diag=False,
                fs_main=32, fs_std=22, fs_swap=20, thresh=0.28):
    n = mean.shape[0]
    for i in range(n):
        for j in range(n):
            if skip_diag and i == j:
                continue
            if np.isnan(mean[i, j]):
                continue
            v       = mean[i, j]
            txt_col = "white" if abs(v) > thresh else "#1a1a1a"
            ax.text(j, i - 0.28, f"{v:+.3f}",
                    ha="center", va="center",
                    fontproperties=FB(fs_main, bold=True), color=txt_col)
            ax.text(j, i + 0.02, f"±{std[i, j]:.2f}",
                    ha="center", va="center",
                    fontproperties=FB(fs_std), color=txt_col, alpha=0.90)
            if swap_mean is not None and not np.isnan(swap_mean[i, j]):
                s_v  = swap_mean[i, j]
                s_sd = swap_std[i, j] if swap_std is not None and not np.isnan(swap_std[i, j]) else 0.
                ax.text(j, i + 0.32,
                        f"|ΔH|={s_v:.2f}±{s_sd:.2f}",
                        ha="center", va="center",
                        fontproperties=FB(fs_swap, bold=True),
                        color=txt_col, alpha=0.85)


def style_axes(ax, labels, title, cbar_im, vmax, cbar_label="H"):
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontproperties=FB(22, bold=True), rotation=0)
    ax.set_yticklabels(labels, fontproperties=FB(22, bold=True))
    ax.set_xlabel("target family  (post author)", fontproperties=FB(20), labelpad=12)
    ax.set_ylabel("source family  (commenter)",   fontproperties=FB(20), labelpad=12)
    ax.set_title(title, fontproperties=FT(26, bold=True), pad=18)
    cb = plt.colorbar(cbar_im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontproperties=FB(20))
    for tick in cb.ax.get_yticklabels():
        tick.set_fontproperties(FB(16))


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_h_matrix(H_dict, keys, title, outpath, cross_only=False,
                  swap_mean=None, swap_std=None):
    mean, std = mat_stats(H_dict, keys)
    if cross_only:
        np.fill_diagonal(mean, np.nan)
        np.fill_diagonal(std,  np.nan)
    vmax = np.nanmax(np.abs(mean)) * 1.08
    fig, ax = plt.subplots(figsize=(15.0, 12.5))
    im = ax.imshow(mean, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    label_cells(ax, mean, std,
                swap_mean=swap_mean, swap_std=swap_std,
                skip_diag=cross_only)
    style_axes(ax, keys, title, im, vmax)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {Path(outpath).name}")


def plot_swap(swap_mean, swap_std, keys, title, outpath):
    vmax = np.nanmax(swap_mean[~np.isnan(swap_mean)]) * 1.05
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(swap_mean, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="equal")
    n = len(keys)
    for i in range(n):
        for j in range(n):
            if np.isnan(swap_mean[i, j]):
                continue
            v       = swap_mean[i, j]
            txt_col = "white" if v > vmax * 0.65 else "#1a1a1a"
            ax.text(j, i - 0.14, f"{v:.3f}",
                    ha="center", va="center",
                    fontproperties=FB(18, bold=True), color=txt_col)
            ax.text(j, i + 0.26, f"±{swap_std[i, j]:.2f}",
                    ha="center", va="center",
                    fontproperties=FB(12), color=txt_col, alpha=0.80)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(keys, fontproperties=FB(13, bold=True), rotation=0)
    ax.set_yticklabels(keys, fontproperties=FB(13, bold=True))
    ax.set_xlabel("target family", fontproperties=FB(11), labelpad=7)
    ax.set_ylabel("source family", fontproperties=FB(11), labelpad=7)
    ax.set_title(title, fontproperties=FT(16, bold=True), pad=12)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$|\Delta H|$", fontproperties=FB(11))
    for tick in cb.ax.get_yticklabels():
        tick.set_fontproperties(FB(10))
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {Path(outpath).name}")


def model_color(m):
    import colorsys
    full = {v: k for k, v in SHORT.items()}.get(m, m)
    h = 0
    for c in full:
        h = (h*31 + ord(c)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    hue = abs(h) % 360
    r, g, b = colorsys.hls_to_rgb(hue/360, 0.45, 0.55)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def plot_self_pref(H_data, tag, label, outdir):
    self_vals = [H_data.get((m, m), []) for m in MODELS]
    self_m    = [np.mean(v) if v else np.nan for v in self_vals]
    self_s    = [np.std(v, ddof=1) if len(v) > 1 else 0. for v in self_vals]
    self_n    = [len(v) for v in self_vals]

    colors = []
    for v in self_m:
        if np.isnan(v):   colors.append("#aaa")
        elif v > 0:       colors.append("#c0392b")
        else:             colors.append("#2980b9")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = range(len(MODELS))
    ax.bar(xs, self_m, color=colors, alpha=0.85, width=0.6,
           edgecolor="white", linewidth=0.6)
    ax.errorbar(xs, self_m, yerr=self_s, fmt="none", color="#333",
                capsize=5, lw=1.5, capthick=1.5)
    ax.axhline(0, color="#555", lw=0.8)
    for x, m, s, n in zip(xs, self_m, self_s, self_n):
        if np.isnan(m):
            continue
        yoff = m + s + 0.02 if m >= 0 else m - s - 0.07
        va   = "bottom" if m >= 0 else "top"
        ax.text(x, yoff, f"n={n}", ha="center", va=va,
                fontproperties=FB(9), color="#555")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(MODELS, fontproperties=FB(11, bold=True))
    ax.set_ylabel(r"$H_{X \to X}$  (in-group self-preference)", fontproperties=FB(10))
    ax.set_title(f"{label} in-group self-preference", fontproperties=FT(14, bold=True), pad=10)
    plt.tight_layout()
    outpath = outdir / f"fig_self_pref_{tag}.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {outpath.name}")


def plot_elo(all_H_dy, outdir):
    """Elo rating from dyadic H-matrix net-attractor advantage."""
    matches = []
    for pair, vals in all_H_dy.items():
        if pair[0] == pair[1]:
            continue
        x, y = pair[0], pair[1]
        h_xy = np.mean(vals) if vals else 0
        rev_pair = (y, x)
        h_yx = np.mean(all_H_dy.get(rev_pair, [0]))
        net_x = h_yx - h_xy
        matches.append((x, y, net_x))

    elo = defaultdict(lambda: 1200.0)
    K   = 32
    for x, y, net in matches:
        exp_x = 1 / (1 + 10**((elo[y]-elo[x])/400))
        sx    = 1 / (1 + math.exp(-net/0.5))
        elo[x] += K * (sx - exp_x)
        elo[y] += K * ((1-sx) - (1-exp_x))

    elo_items  = sorted(elo.items(), key=lambda kv: -kv[1])
    elo_models = [k for k, v in elo_items]
    elo_vals   = [v for k, v in elo_items]
    cols       = [model_color(m) for m in elo_models]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(range(len(elo_models)), elo_vals, color=cols, alpha=0.88,
           width=0.6, edgecolor="white", linewidth=0.6)
    ax.axhline(1200, color="#888", lw=0.8, ls="--", alpha=0.6)
    for x, v, m in zip(range(len(elo_models)), elo_vals, elo_models):
        ax.text(x, v+4, f"{v:.0f}", ha="center", va="bottom",
                fontproperties=FB(10, bold=True), color="#222")
    ax.set_xticks(list(range(len(elo_models))))
    ax.set_xticklabels(elo_models, fontproperties=FB(11, bold=True))
    ax.set_ylabel("Elo rating", fontproperties=FB(10))
    ax.set_title("Elo ranking  (net attractor advantage, dyadic)",
                 fontproperties=FT(14, bold=True), pad=10)
    plt.tight_layout()
    fig.savefig(outdir / "fig_elo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_elo.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=os.environ.get("RUNS_DIR", "./runs"),
                        help="Directory containing sweep_*.db files.")
    parser.add_argument("--outdir",   default=os.environ.get("OUTDIR", str(Path(__file__).parent)),
                        help="Output directory for figures.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dyadic runs from {runs_dir}...")
    all_H_dy, swap_vals_dy = load_dyadic(runs_dir)

    print(f"Loading triadic runs from {runs_dir}...")
    all_H_tri, tri_by_ts = load_triadic(runs_dir)

    all_H_combined = defaultdict(list)
    for pair, vals in all_H_dy.items():
        all_H_combined[pair].extend(vals)
    for pair, vals in all_H_tri.items():
        all_H_combined[pair].extend(vals)

    swap_dy_mean,   swap_dy_std   = dyadic_swap_matrix(swap_vals_dy, MODELS)
    swap_tri_mean,  swap_tri_std  = triadic_swap_matrix(tri_by_ts, MODELS)
    swap_comb_mean, swap_comb_std = combined_swap_matrix(
        swap_dy_mean, swap_dy_std, swap_tri_mean, swap_tri_std, MODELS)

    datasets = [
        ("dyadic",   all_H_dy,       swap_dy_mean,   swap_dy_std),
        ("triadic",  all_H_tri,      swap_tri_mean,  swap_tri_std),
        ("combined", all_H_combined, swap_comb_mean, swap_comb_std),
    ]

    for tag, H_data, sw_mean, sw_std in datasets:
        label = {"dyadic": "Dyadic", "triadic": "Triadic", "combined": "Combined"}[tag]

        plot_h_matrix(
            H_data, MODELS,
            rf"{label} directed engagement homophily  $H_{{X \to Y}}$",
            outdir / f"fig_h_matrix_{tag}.png",
            swap_mean=sw_mean, swap_std=sw_std,
        )
        plot_h_matrix(
            H_data, MODELS,
            rf"{label} cross-group homophily  $H_{{X \to Y}}$,  $X \neq Y$",
            outdir / f"fig_h_matrix_cross_{tag}.png",
            cross_only=True,
            swap_mean=sw_mean, swap_std=sw_std,
        )

        swap_title = {
            "dyadic":   r"Dyadic swap instability:  mean $|H_{AB} - H_{BA}|$",
            "triadic":  r"Triadic ordering instability:  mean max-$|\Delta H|$",
            "combined": r"Ordering instability:  mean max-$|\Delta H|$  (pooled)",
        }[tag]
        plot_swap(sw_mean, sw_std, MODELS, swap_title,
                  outdir / f"fig_swap_stability_{tag}.png")

        plot_self_pref(H_data, tag, label, outdir)

    plot_elo(all_H_dy, outdir)

    print(f"\nAll figures written to {outdir}")


if __name__ == "__main__":
    main()
