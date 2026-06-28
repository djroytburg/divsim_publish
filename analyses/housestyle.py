"""Shared house style: Palatino (TeX Gyre Pagella) for titles/headers,
Ubuntu Mono for body text and numbers."""
from matplotlib import font_manager as fm

_PAGELLA = "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ["texgyrepagella-regular.otf", "texgyrepagella-bold.otf", "texgyrepagella-italic.otf"]:
    try: fm.fontManager.addfont(f"{_PAGELLA}/{_f}")
    except Exception: pass
from pathlib import Path
for ttf in (Path.home() / "fonts").rglob("*.ttf"):
    try: fm.fontManager.addfont(str(ttf))
    except Exception: pass

# family names once registered
SERIF = "TeX Gyre Pagella"   # Palatino-style, for titles/headers
MONO  = "Ubuntu Mono"        # for body text + numbers

def serif(size, bold=False):
    return fm.FontProperties(family=SERIF, size=size, weight="bold" if bold else "normal")
def mono(size, bold=False):
    return fm.FontProperties(family=MONO, size=size, weight="bold" if bold else "normal")

# per-model palette (consistent across figures)
MODEL_COLOR = {
    "gpt-oss": "#d62728",  # red (attractor)
    "Qwen":    "#1f77b4",  # blue
    "Mag":     "#9467bd",  # purple (repeller)
    "GLM-4":   "#2ca02c",  # green
    "Gemma":   "#ff7f0e",  # orange
}
