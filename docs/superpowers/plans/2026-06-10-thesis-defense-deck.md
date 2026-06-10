# Thesis Defense Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate `presentation/momentum_defense.pptx` — an 11-slide mechanism-forward thesis-defense deck (plus 12 backup slides and speaker notes) built per the Tilburg guidelines, using canonical numbers and figures from the thesis.

**Architecture:** A small `python-pptx` build with three layers: `content.py` holds all canonical numbers and text (single source of truth, guarded by tests against the stale Gadi numbers); `deck_lib.py` holds reusable slide primitives (title, bullets, fitted figure, table, pipeline diagram, stat callout, notes, and a generic `content_slide`); `build_defense_deck.py` orchestrates 11 bespoke/generic main slides + 12 data-driven backup slides and saves the `.pptx`. `rasterize_assets.py` converts the few PDF-only figures (logo + 3 backup charts) to PNG since PowerPoint cannot embed PDF.

**Tech Stack:** Python, `python-pptx`, `Pillow` (image fitting), optional `PyMuPDF`/`sips` (PDF rasterizing), `pytest`. Branch: `presentation/defense-deck` (already checked out).

**Spec:** `docs/superpowers/specs/2026-06-10-thesis-defense-deck-design.md`

---

## File structure

| File | Responsibility |
|------|----------------|
| `presentation/content.py` | Canonical numbers + slide text + backup-slide data; `EXPECTED_*` constants; `FORBIDDEN` stale literals |
| `presentation/deck_lib.py` | pptx primitives: `new_slide`, `title`, `bullets`, `figure_fit`, `table`, `pipeline`, `stat`, `notes`, `content_slide`, palette |
| `presentation/rasterize_assets.py` | `ensure_pngs()` — PDF→PNG for logo + 3 backup charts into `presentation/assets/` |
| `presentation/build_defense_deck.py` | `build_presentation()` (returns `Presentation`) + `main()` (saves); 11 main builders + 12 backup builders |
| `presentation/test_deck.py` | pytest: canonical-number guard, primitives smoke, rasterize output, slide count/titles/notes |
| `presentation/assets/` | generated PNGs (gitignored) |
| `presentation/momentum_defense.pptx` | generated deck (gitignored) |

Each main slide draws figures from `plots/thesis/`; numbers come only from `content.py`.

---

## Task 1: Scaffold, dependency check, gitignore

**Files:**
- Create: `presentation/` (dir), `presentation/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Verify required libraries are importable (do NOT auto-install)**

Run:
```bash
cd /Users/giladgang/momentum_regime && python -c "import pptx, PIL; print('pptx', pptx.__version__, '| PIL ok')"
```
Expected: prints a version (e.g. `pptx 0.6.x | PIL ok`).
If it errors with ModuleNotFoundError: STOP and tell Gilad — per CLAUDE.md, do not `pip install` without updating `requirements.lock`. Both are common (pptx was used by the old deck; PIL ships with the matplotlib stack), so this should pass.

- [ ] **Step 2: Verify a PDF rasterizer is available**

Run:
```bash
cd /Users/giladgang/momentum_regime && (python -c "import fitz; print('pymupdf ok')" 2>/dev/null || (command -v sips >/dev/null && echo "sips ok") || echo "NO RASTERIZER")
```
Expected: `pymupdf ok` or `sips ok` (sips is built into macOS). If `NO RASTERIZER`, STOP and flag.

- [ ] **Step 3: Create the package dir**

Run:
```bash
mkdir -p /Users/giladgang/momentum_regime/presentation/assets && touch /Users/giladgang/momentum_regime/presentation/__init__.py
```

- [ ] **Step 4: Gitignore generated artifacts**

Add these lines to `.gitignore` (append under the `.superpowers/` line):
```
presentation/assets/
presentation/*.pptx
```

- [ ] **Step 5: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add .gitignore presentation/__init__.py && git commit -m "presentation: scaffold deck package + ignore build artifacts

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `content.py` — canonical numbers, text, and the stale-number guard

**Files:**
- Create: `presentation/content.py`
- Test: `presentation/test_deck.py`

- [ ] **Step 1: Write the failing test**

Create `presentation/test_deck.py`:
```python
from presentation import content as C


def test_canonical_headline_numbers():
    assert C.SHARPE_XGB == 1.11
    assert C.FF6_ALPHA == "24.1%"
    assert C.FF6_T == "4.81"
    assert C.PANIC_SHARPE == 1.53
    assert C.CALM_SHARPE == 0.84
    assert C.SHARPE_FIX12 == 0.06
    assert C.MDD_FIX12 == "-72.2%"
    assert C.CLUSTER_SHARPES == [0.92, 0.93, 1.21, 1.82]


def test_stale_gadi_numbers_are_not_used():
    # The old "call with Gadi" deck used a different cut; these must never appear.
    forbidden = ["24.7%", "1.56", "$15.7"]
    blob = " ".join(str(v) for v in vars(C).values() if isinstance(v, (str, int, float, list)))
    for bad in forbidden:
        assert bad not in blob, f"stale value {bad!r} present in content.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` (content.py does not exist yet).

- [ ] **Step 3: Write `presentation/content.py`**

```python
"""Single source of truth for every number and text block in the defense deck.

All numbers are the thesis CANONICAL values, traceable to tables/*.tex and
results/PRODUCTION_METRICS.json (see the design spec, section 8). The stale
"Gadi" cut (alpha 24.7%, panic Sharpe 1.56, terminal wealth $15.7) is forbidden
by test_deck.py::test_stale_gadi_numbers_are_not_used.
"""

# ---- Headline performance (tables/table_performance.tex, table_factor_alphas.tex)
SHARPE_XGB   = 1.11
FF6_ALPHA    = "24.1%"
FF6_T        = "4.81"
CAPM_ALPHA   = "23.4%"
VOL_XGB      = "19.5%"
SHARPE_MARKET = 0.84
MARKET_RET    = "11.9%"
SHARPE_FIX12  = 0.06
MDD_FIX12     = "-72.2%"
SHARPE_DET    = 0.11
SHARPE_LR     = -0.01
SHARPE_NO_PI  = 0.43           # XGB minus the regime signal (>60% drop from 1.11)

# ---- Regime split (tables/table_regime_sharpe.tex)
PANIC_SHARPE = 1.53
CALM_SHARPE  = 0.84
PANIC_N      = 59
CALM_N       = 108

# ---- Mechanism (tables/table_cluster_k4_descriptors.tex, table_shap.tex, table_leg_betas.tex)
CLUSTER_NAMES   = ["calm continuation", "transition", "post-panic recovery", "deep crisis"]
CLUSTER_SHARPES = [0.92, 0.93, 1.21, 1.82]
CLUSTER_PI      = [0.21, 0.34, 0.32, 0.81]
SHAP_PI         = "46%"
LEG_BETA_LONG   = (1.40, 1.65)   # (calm, panic)
LEG_BETA_SHORT  = (1.22, 0.98)

# ---- Robustness (main_results.tex history/international, table_international_results.tex)
CRASH2009_XGB = "+20.5%"
CRASH2009_FIX = "-62.6%"
GFC_XGB       = "+11.1%"
GFC_MKT       = "-13.9%"
DOTCOM_XGB    = "-43.8%"
DOTCOM_MDD    = "-64.8%"
UK_XGB, UK_FIX = 0.72, 0.35
JP_XGB, JP_FIX = 0.57, -0.01
MDD_US, MDD_UK, MDD_JP = "-22.8%", "-21.4%", "-19.8%"

# ---- Sample
TRAIN = "1990-2010"
TEST  = "2011-2024 (167 months)"

# ---- Title block
TITLE = "Momentum Across Market Regimes"
SUBTITLE = "A Machine Learning Approach"
AUTHOR = "Gilad Gang"
SUPERVISOR = "Supervisor: Denis Kojevnikov"
PROGRAMME = "MSc Quantitative Finance and Actuarial Science  -  Tilburg University"
DATE = "June 2026"

# ---- Figure locations (relative to repo root)
FIG = "plots/thesis"

# ---- Backup slides: (title, figure-or-None, [bullets], notes)
# figure is a path under plots/thesis OR a rasterized asset key handled by the builder.
BACKUP_SLIDES = [
    ("B1  HMM specification & Gibbs sampling", "ASSET:gibbs",
     ["Two-state Bayesian HMM; emissions Gaussian, NIW prior; transition Dirichlet(9,1)/(1,9).",
      "Estimated by Gibbs sampling with forward-filtering backward-sampling (FFBS).",
      "Trading signal = forward filter only (no look-ahead)."],
     "Walk the FFBS loop only if asked; the diagram carries it."),
    ("B2  Sampler convergence", "convergence_trace.png",
     ["ESS > 100 on every parameter; Gelman-Rubin R-hat = 1.001 across 5 chains.",
      "200-seed ensemble; downstream Sharpe plateaus by ~30 seeds."],
     "R-hat ~ 1 means every chain found the same two-cluster solution."),
    ("B3  Four-pass HMM feature selection", None,
     ["Candidate pool of 9 stress indicators; DD anchored; cap of 4 features.",
      "Passes: quality screen -> portfolio-Sharpe ranking -> definitive validation -> stability.",
      "Selected set: DD + DISP + REL_N + CS."],
     "Each pass shrinks the set on a different criterion."),
    ("B4  Factor-model alphas", "ASSET:acf",
     ["Alpha survives CAPM, FF3, Carhart, FF5, and FF6 (adds UMD).",
      "Newey-West (6 lags); Ljung-Box supports the truncation.",
      "Six-factor alpha 24.1% (t = 4.81)."],
     "FF6 is the toughest test - it cancels mechanical winner-tilt."),
    ("B5  Risk-aversion trade-off", "ASSET:riskav",
     ["Per-direction log-utility with a volatility penalty gamma in [0,1].",
      "gamma=0.1 keeps Sharpe ~1.0 while drawdown tightens to -18%.",
      "Edge depends on taking volatility; high gamma collapses the spread."],
     "Use if asked about risk control for a risk-averse investor."),
    ("B6  25-year expanding window + reversal failure", None,
     ["Annual retraining 2000-2024; strictly causal, 299 months.",
      "2009 momentum crash: XGB +20.5% vs unconditional -62.6%.",
      "Dot-com bear: XGB -43.8% (the reversal-failure mode); D&M overlay is the complement."],
     "This is the honest worst case; pair with the D&M circuit-breaker idea."),
    ("B7  SHAP shares per horizon by leg", "shap_per_horizon_by_leg.png",
     ["Both legs concentrate on months 8-12 (~65% each).",
      "Consistent with the classical 12-month momentum range."],
     "Direction comes from the z-curves, not these magnitudes."),
    ("B8  Leg-level CAPM betas", None,
     ["Long-leg beta 1.40 (calm) -> 1.65 (panic); short-leg 1.22 -> 0.98.",
      "Long-leg beta > short-leg beta in every regime and every cluster.",
      "Composition (which stocks), not exposure (how much)."],
     "Direct evidence the regime acts on holdings, not sizing."),
    ("B9  International feature selection", None,
     ["UK and Japan both select DD + VOL + REL_N via the same 4-pass procedure.",
      "US-specific CS (BAA-AAA) has no direct local equivalent.",
      "Transplanted US template fails the Pass-1 quality screen in both regions."],
     "Only the upstream HMM features are region-specific; the model transfers."),
    ("B10  Number of states & emission robustness", None,
     ["K=2 chosen: K>=3 adds no downstream Sharpe and destabilises.",
      "Student-t emissions: Normal has lowest BIC; classifications agree >97.8%."],
     "Two states = one stress dimension, the one momentum cares about."),
    ("B11  Transaction costs & turnover", None,
     ["10 bps one-way baseline; XGB monthly one-way turnover ~132%.",
      "XGB stays profitable to 50 bps (Sharpe 0.78); others die at moderate cost."],
     "Value-weighting + NYSE breakpoints justify the 10 bps assumption."),
    ("B12  XGB hyperparameters", None,
     ["500 trees, learning rate 0.05, max depth 4, row/col subsample 0.8.",
      "50-seed ensemble averaged before ranking.",
      "Depth 4 = regime gate (1) + within-regime shape reading (3)."],
     "Performance peaks at depth 4 and falls off either side."),
]

EXPECTED_MAIN = 11
EXPECTED_BACKUP = len(BACKUP_SLIDES)   # 12
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add presentation/content.py presentation/test_deck.py && git commit -m "presentation: canonical content + stale-number guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `deck_lib.py` — slide primitives

**Files:**
- Create: `presentation/deck_lib.py`
- Test: `presentation/test_deck.py` (append)

- [ ] **Step 1: Write the failing test (append to test_deck.py)**

```python
def test_primitives_build_a_titled_slide_with_notes():
    from pptx import Presentation
    from pptx.util import Inches
    from presentation import deck_lib as L
    prs = Presentation()
    prs.slide_width, prs.slide_height = L.SLIDE_W, L.SLIDE_H
    s = L.content_slide(
        prs, title_text="Hello",
        bullets_items=["one", ("two", L.GOOD)],
        table_rows=[["A", "B"], ["1", "2"]],
        takeaway_text="done", notes_text="say hi",
    )
    titles = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame]
    assert "Hello" in titles
    assert s.notes_slide.notes_text_frame.text == "say hi"
    assert len(prs.slides) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_primitives_build_a_titled_slide_with_notes -v`
Expected: FAIL — `ModuleNotFoundError: deck_lib`.

- [ ] **Step 3: Write `presentation/deck_lib.py`**

```python
"""Reusable python-pptx primitives for the defense deck (16:9, clean-minimal)."""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
FONT = "Calibri"

INK    = RGBColor(0x1A, 0x1A, 0x1A)
MUTE   = RGBColor(0x6B, 0x6B, 0x6B)
ACCENT = RGBColor(0x00, 0x2C, 0x5F)   # Tilburg navy
GOOD   = RGBColor(0x1B, 0x7A, 0x3D)
BAD    = RGBColor(0xB3, 0x2A, 0x2A)
PANEL  = RGBColor(0xEE, 0xF1, 0xF6)


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tb.text_frame.word_wrap = True
    return tb


def _set(p, text, size, color=INK, bold=False, italic=False, align=PP_ALIGN.LEFT):
    p.alignment = align
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.name, f.color.rgb, f.bold, f.italic = Pt(size), FONT, color, bold, italic
    return r


def title(slide, text, color=ACCENT):
    tb = _box(slide, Inches(0.6), Inches(0.4), Inches(12.1), Inches(1.0))
    _set(tb.text_frame.paragraphs[0], text, 30, color=color, bold=True)
    return tb


def takeaway(slide, text, color=GOOD):
    tb = _box(slide, Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.5))
    _set(tb.text_frame.paragraphs[0], text, 16, color=color, italic=True)
    return tb


def bullets(slide, items, left=Inches(0.7), top=Inches(1.7), width=Inches(12), size=20, gap=12):
    tf = _box(slide, left, top, width, Inches(5)).text_frame
    for i, item in enumerate(items):
        text, color = item if isinstance(item, tuple) else (item, INK)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        _set(p, "•  " + text, size, color=color)


def figure_fit(slide, path, box_left, box_top, box_w, box_h):
    """Place an image inside a box, preserving aspect ratio, centered."""
    iw, ih = Image.open(str(path)).size
    if iw / ih > box_w / box_h:
        w = box_w; h = int(box_w * ih / iw)
    else:
        h = box_h; w = int(box_h * iw / ih)
    slide.shapes.add_picture(str(path), box_left + (box_w - w) // 2,
                             box_top + (box_h - h) // 2, width=w, height=h)


def table(slide, rows, left, top, width, height, highlight_row=None, font=14):
    nrow, ncol = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(nrow, ncol, left, top, width, height).table
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = gfx.cell(ri, ci)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            align = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.RIGHT
            head = ri == 0
            hi = highlight_row is not None and ri == highlight_row
            _set(p, str(val), font, color=MUTE if head else INK, bold=head or hi, align=align)
    return gfx


def pipeline(slide, stages, top=Inches(1.5), height=Inches(0.85),
             left=Inches(0.9), total_w=Inches(11.5)):
    n = len(stages)
    gap = Inches(0.35)
    box_w = (total_w - gap * (n - 1)) // n
    x = left
    for i, label in enumerate(stages):
        sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, box_w, height)
        sh.fill.solid(); sh.fill.fore_color.rgb = PANEL
        sh.line.color.rgb = ACCENT; sh.line.width = Pt(1.25)
        sh.text_frame.word_wrap = True
        _set(sh.text_frame.paragraphs[0], label, 13, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)
        if i < n - 1:
            ar = _box(slide, x + box_w, top, gap, height)
            _set(ar.text_frame.paragraphs[0], "→", 22, color=MUTE, align=PP_ALIGN.CENTER)
        x += box_w + gap


def stat(slide, number, label, left, top, width=Inches(3.6), color=ACCENT):
    tf = _box(slide, left, top, width, Inches(1.9)).text_frame
    _set(tf.paragraphs[0], number, 44, color=color, bold=True, align=PP_ALIGN.CENTER)
    _set(tf.add_paragraph(), label, 14, color=MUTE, align=PP_ALIGN.CENTER)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def content_slide(prs, *, title_text, figure_path=None, bullets_items=None,
                  table_rows=None, table_highlight=None, takeaway_text=None, notes_text=None):
    """Generic single-figure / bullets / table slide used by most backup slides."""
    s = new_slide(prs)
    title(s, title_text)
    if bullets_items:
        bullets(s, bullets_items, top=Inches(1.6))
    if figure_path:
        if bullets_items:
            figure_fit(s, figure_path, Inches(6.9), Inches(1.7), Inches(6.0), Inches(4.6))
        else:
            figure_fit(s, figure_path, Inches(1.2), Inches(1.6), Inches(11.0), Inches(4.9))
    if table_rows:
        table(s, table_rows, Inches(0.8), Inches(4.8), Inches(7.6), Inches(1.8),
              highlight_row=table_highlight)
    if takeaway_text:
        takeaway(s, takeaway_text)
    if notes_text:
        notes(s, notes_text)
    return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add presentation/deck_lib.py presentation/test_deck.py && git commit -m "presentation: pptx slide primitives

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `rasterize_assets.py` — PDF→PNG for embed-only assets

**Files:**
- Create: `presentation/rasterize_assets.py`
- Test: `presentation/test_deck.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_rasterize_produces_pngs():
    from presentation.rasterize_assets import ensure_pngs
    paths = ensure_pngs()
    assert set(paths) >= {"logo", "gibbs", "acf", "riskav"}
    for key, p in paths.items():
        assert p.exists() and p.stat().st_size > 1000, f"{key} png missing/empty"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_rasterize_produces_pngs -v`
Expected: FAIL — `ModuleNotFoundError: rasterize_assets`.

- [ ] **Step 3: Write `presentation/rasterize_assets.py`**

```python
"""Rasterize the PDF-only figures PowerPoint cannot embed into presentation/assets/.

Tries PyMuPDF (fitz) at 200 DPI; falls back to macOS `sips`. Skips work if a PNG
twin already exists in plots/thesis/ or the asset is already rasterized.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "plots" / "thesis"
OUT = ROOT / "presentation" / "assets"

# key -> source pdf stem in plots/thesis
PDFS = {
    "logo": "tilburg_logo",
    "gibbs": "gibbs_sampling_diagram",
    "acf": "factor_residual_acf_ff6",
    "riskav": "risk_aversion_dual_util",
}


def _render(pdf: Path, png: Path) -> None:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf))
        pix = doc.load_page(0).get_pixmap(dpi=200)
        pix.save(str(png))
        return
    except Exception:
        pass
    subprocess.run(
        ["sips", "-s", "format", "png", "--resampleHeightWidthMax", "2400",
         str(pdf), "--out", str(png)],
        check=True, capture_output=True,
    )


def ensure_pngs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, stem in PDFS.items():
        twin = SRC / f"{stem}.png"          # prefer an existing PNG twin
        if twin.exists():
            result[key] = twin
            continue
        png = OUT / f"{stem}.png"
        if not png.exists():
            _render(SRC / f"{stem}.pdf", png)
        result[key] = png
    return result


if __name__ == "__main__":
    for k, p in ensure_pngs().items():
        print(f"{k}: {p}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_rasterize_produces_pngs -v`
Expected: PASS (PNGs created under `presentation/assets/` or found as twins).

- [ ] **Step 5: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add presentation/rasterize_assets.py presentation/test_deck.py && git commit -m "presentation: rasterize PDF-only figures to PNG

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `build_defense_deck.py` — the 11 main slides + orchestrator

**Files:**
- Create: `presentation/build_defense_deck.py`
- Test: `presentation/test_deck.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_main_deck_structure():
    from presentation.build_defense_deck import build_presentation
    from presentation import content as C
    prs = build_presentation()
    # 11 main + 1 divider + 12 backup
    assert len(prs.slides) == C.EXPECTED_MAIN + 1 + C.EXPECTED_BACKUP
    # every slide has a non-empty title (first text shape) ...
    for i, s in enumerate(prs.slides):
        texts = [sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text]
        assert texts, f"slide {i} has no text"
    # ... and every MAIN slide carries speaker notes
    for i, s in enumerate(list(prs.slides)[:C.EXPECTED_MAIN]):
        assert s.notes_slide.notes_text_frame.text.strip(), f"main slide {i} missing notes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_main_deck_structure -v`
Expected: FAIL — `ModuleNotFoundError: build_defense_deck`.

- [ ] **Step 3: Write `presentation/build_defense_deck.py` (main slides + orchestrator; backup builders added in Task 6)**

```python
"""Build the thesis defense deck -> presentation/momentum_defense.pptx."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN

from presentation import content as C
from presentation import deck_lib as L
from presentation.rasterize_assets import ensure_pngs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / C.FIG
OUT = ROOT / "presentation" / "momentum_defense.pptx"


def _fig(name):
    return FIG / name


# ---------------- Main slides ----------------

def s01_title(prs, A):
    s = L.new_slide(prs)
    L.figure_fit(s, A["logo"], Inches(4.9), Inches(0.7), Inches(3.5), Inches(1.6))
    tf = L._box(s, Inches(0.8), Inches(2.7), Inches(11.7), Inches(2.0)).text_frame
    L._set(tf.paragraphs[0], C.TITLE, 40, color=L.ACCENT, bold=True, align=PP_ALIGN.CENTER)
    L._set(tf.add_paragraph(), C.SUBTITLE, 22, color=L.MUTE, italic=True, align=PP_ALIGN.CENTER)
    tf2 = L._box(s, Inches(0.8), Inches(4.7), Inches(11.7), Inches(2.2)).text_frame
    for line in (C.AUTHOR, C.SUPERVISOR, C.PROGRAMME, C.DATE):
        p = tf2.paragraphs[0] if not tf2.paragraphs[0].runs else tf2.add_paragraph()
        L._set(p, line, 16, color=L.INK, align=PP_ALIGN.CENTER)
    L.notes(s, "Title and framing. Aim: understand momentum across regimes, not beat the market.")


def s02_motivation(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Momentum: most profitable anomaly, most fragile")
    L.bullets(s, [
        "Buying past winners and shorting past losers is one of finance's most robust anomalies.",
        ("Yet it crashes hard at regime turns - losses over 50% in the 2009 rebound.", L.BAD),
    ], top=Inches(1.5), width=Inches(12))
    L.stat(s, C.MDD_FIX12, "fixed 12-mo momentum\nmax drawdown", Inches(1.6), Inches(3.6), color=L.BAD)
    L.stat(s, "0.06", "its Sharpe\n(2011-2024)", Inches(5.5), Inches(3.6), color=L.BAD)
    L.stat(s, "?", "what selects stocks\nin each regime", Inches(9.2), Inches(3.6), color=L.ACCENT)
    L.takeaway(s, "Question: how does the cross-section reorganize between calm and panic, and what selects stocks in each?", color=L.ACCENT)
    L.notes(s, "Set up the puzzle. Existing fixes scale exposure or blend horizons; the selection rule stays unconditional. We ask a different question.")


def s03_literature(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Prior work, and the gap")
    cols = [
        ("Momentum & crashes", ["Jegadeesh-Titman (1993)", "Daniel-Moskowitz (2016):\ncrashes are predictable"]),
        ("Crash management", ["Adjusts EXPOSURE or HORIZON", "Barroso-Santa-Clara; GHM (2023)"]),
        ("Regime + ML", ["Cooper (2004): coarse up/down", "Gu (2020); Beckmeyer-\nWiedemann (2025)"]),
    ]
    x = Inches(0.7)
    for head, items in cols:
        tf = L._box(s, x, Inches(1.6), Inches(3.9), Inches(3.2)).text_frame
        L._set(tf.paragraphs[0], head, 18, color=L.ACCENT, bold=True)
        for it in items:
            L._set(tf.add_paragraph(), it, 14, color=L.INK)
        x += Inches(4.05)
    L.takeaway(s, "Gap: nobody studies how the full momentum term structure reorganizes across regimes, or whether capturing it needs nonlinear regime-conditioned selection.", color=L.ACCENT)
    L.notes(s, "Three buckets. Each treats the regime as a switch on sizing or horizon, never as a feature inside a nonlinear selection rule. That gap is the thesis.")


def s04_methodology_hmm(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Methodology I - the regime signal (Stage 1)")
    L.pipeline(s, ["HMM\n(4 stress features)", "pi : panic\nprobability", "XGBoost", "Long-short\nportfolio"], top=Inches(1.35))
    L.figure_fit(s, _fig("regime_probabilities.png"), Inches(1.2), Inches(2.6), Inches(11.0), Inches(3.9))
    L.takeaway(s, "A 2-state Bayesian HMM turns four stress indicators into a continuous panic probability pi that updates monthly. pi is a context variable, not a return predictor.")
    L.notes(s, "pi spikes at dot-com, GFC, COVID, 2022. Gibbs/FFBS detail is in backup B1. Trained 1990-2010, frozen for 2011-2024.")


def s05_methodology_xgb(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Methodology II - cross-sectional selection (Stage 2)")
    L.bullets(s, [
        "Inputs: 12 momentum horizons (1-12 months) + the regime probability pi.",
        ("XGBoost ranks stocks on PREDICTED forward return, not raw momentum.", L.ACCENT),
        "So the long leg need not hold high-momentum names; pi routes how the whole term structure is used.",
        "Benchmarked against a deterministic formula (DET) and a logistic regression (LR).",
    ], top=Inches(1.6), width=Inches(12))
    L.table(s, [["Method", "Sharpe"],
                ["DET (formula)", f"{C.SHARPE_DET:.2f}"],
                ["LR (linear)", f"{C.SHARPE_LR:.2f}"],
                ["XGB (nonlinear)", f"{C.SHARPE_XGB:.2f}"]],
            Inches(0.8), Inches(4.9), Inches(4.6), Inches(1.7), highlight_row=3)
    L.takeaway(s, "Only the nonlinear model turns these inputs into performance. XGB specifics (depth-4, 50-seed ensemble) are in backup B12.")
    L.notes(s, "Emphasise predicted-return ranking - that is what lets the long leg flip composition by regime. DET and LR share the exact same inputs and collapse.")


def s06_results(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Headline results (2011-2024, out-of-sample)")
    L.figure_fit(s, _fig("cs_performance_regime_shaded.png"), Inches(0.7), Inches(1.5), Inches(7.6), Inches(5.0))
    L.table(s, [["Sharpe", ""],
                ["Market", f"{C.SHARPE_MARKET:.2f}"],
                ["Fixed 12-mo mom.", f"{C.SHARPE_FIX12:.2f}"],
                ["XGB (mom + pi)", f"{C.SHARPE_XGB:.2f}"]],
            Inches(8.7), Inches(1.7), Inches(3.9), Inches(1.9), highlight_row=3)
    tf = L._box(s, Inches(8.7), Inches(4.0), Inches(4.2), Inches(2.4)).text_frame
    L._set(tf.paragraphs[0], f"Six-factor alpha {C.FF6_ALPHA}", 20, color=L.GOOD, bold=True)
    L._set(tf.add_paragraph(), f"t = {C.FF6_T}", 16, color=L.MUTE)
    L._set(tf.add_paragraph(), f"Remove pi -> Sharpe {C.SHARPE_NO_PI:.2f}", 14, color=L.INK)
    L._set(tf.add_paragraph(), f"Fixed mom. MDD {C.MDD_FIX12}", 14, color=L.BAD)
    L.takeaway(s, "XGB is the only row clearing every conventional t-stat threshold; the linear baseline on identical features collapses.")
    L.notes(s, "Credibility first. Alpha survives FF6 (cancels mechanical winner tilt). The advantage concentrates in the shaded panic months - that motivates the mechanism.")


def s07_mechanism_flip(prs, A):
    s = L.new_slide(prs)
    L.title(s, "The mechanism I - the direction flip")
    L.figure_fit(s, _fig("zscore_long_heatmap.png"), Inches(3.3), Inches(1.5), Inches(6.7), Inches(5.0))
    tf = L._box(s, Inches(0.6), Inches(2.0), Inches(2.6), Inches(4.0)).text_frame
    L._set(tf.paragraphs[0], "Calm", 20, color=L.GOOD, bold=True)
    L._set(tf.add_paragraph(), "picks sit ABOVE the cross-sectional mean (winners)", 14, color=L.INK)
    L._set(tf.add_paragraph(), "Panic", 20, color=L.BAD, bold=True)
    L._set(tf.add_paragraph(), "picks sit BELOW at every horizon (the beaten-down)", 14, color=L.INK)
    L.takeaway(s, "'Buy the dip' emerges only after conditioning on pi - no single-horizon momentum rule would select these names.")
    L.notes(s, "Each row is a test month; columns are the 12 horizons. Red below the mean in panic, blue above in calm. Foreshadow the four modes.")


def s08_mechanism_clusters(prs, A):
    s = L.new_slide(prs)
    L.title(s, "The mechanism II - four modes, one market cycle")
    panels = ["zscore_l2_k4_panel_c0.png", "zscore_l2_k4_panel_c1.png",
              "zscore_l2_k4_panel_c2.png", "zscore_l2_k4_panel_c3.png"]
    coords = [(Inches(0.7), Inches(1.6)), (Inches(6.9), Inches(1.6)),
              (Inches(0.7), Inches(4.0)), (Inches(6.9), Inches(4.0))]
    for (name, sh) in zip(panels, C.CLUSTER_SHARPES):
        pass
    for name, (lx, ty), cname, csh in zip(panels, coords, C.CLUSTER_NAMES, C.CLUSTER_SHARPES):
        L.figure_fit(s, _fig(name), lx, ty, Inches(5.6), Inches(2.2))
        cap = L._box(s, lx, ty - Inches(0.05), Inches(5.6), Inches(0.3)).text_frame
        L._set(cap.paragraphs[0], f"{cname}  (Sharpe {csh:.2f})", 12, color=L.ACCENT, bold=True)
    L.takeaway(s, "Calm continuation -> transition -> post-panic recovery -> deep crisis. Momentum where it works, reversal where it crashes.")
    L.notes(s, "The heatmap clusters into four selection modes that trace one cycle. Cluster 4 (deep crisis) earns the highest Sharpe, 1.82, by buying the most beaten-down names.")


def s09_novel(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Why it's novel")
    L.bullets(s, [
        (f"pi is a routing variable: {C.SHAP_PI} of total SHAP, ~6x its 1/13 share.", L.ACCENT),
        f"Nonlinearity is the binding constraint: DET {C.SHARPE_DET:.2f}, LR {C.SHARPE_LR:.2f}, XGB {C.SHARPE_XGB:.2f}.",
        "Composition, not exposure: which stocks are held flips by regime, not how much.",
    ], top=Inches(1.6), width=Inches(12))
    L.table(s, [["CAPM beta", "Calm", "Panic"],
                ["Long leg", f"{C.LEG_BETA_LONG[0]:.2f}", f"{C.LEG_BETA_LONG[1]:.2f}"],
                ["Short leg", f"{C.LEG_BETA_SHORT[0]:.2f}", f"{C.LEG_BETA_SHORT[1]:.2f}"]],
            Inches(0.8), Inches(4.4), Inches(6.0), Inches(1.8))
    L.takeaway(s, "Long-leg beta exceeds short-leg beta in every regime - the strategy rides rebounds instead of being crushed by them, unlike D&M/Barroso exposure scaling.")
    L.notes(s, "This is the contribution slide. Contrast each literature bucket: we change WHICH stocks, not HOW MUCH, and we learn it nonlinearly from a continuous real-time signal.")


def s10_robustness(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Robustness")
    L.bullets(s, [
        (f"Survives the 2009 momentum crash: XGB {C.CRASH2009_XGB} vs unconditional {C.CRASH2009_FIX}.", L.GOOD),
        "Alpha survives all five factor models (CAPM through FF6).",
    ], top=Inches(1.5), width=Inches(12))
    L.table(s, [["Sharpe (2011-2024)", "XGB", "Fixed 12-mo"],
                ["US", f"{C.SHARPE_XGB:.2f}", f"{C.SHARPE_FIX12:.2f}"],
                ["UK", f"{C.UK_XGB:.2f}", f"{C.UK_FIX:.2f}"],
                ["Japan", f"{C.JP_XGB:.2f}", f"{C.JP_FIX:.2f}"]],
            Inches(0.8), Inches(3.6), Inches(6.4), Inches(2.3), highlight_row=0)
    tf = L._box(s, Inches(7.6), Inches(3.6), Inches(5.2), Inches(2.4)).text_frame
    L._set(tf.paragraphs[0], "Max drawdown compresses:", 16, color=L.INK, bold=True)
    L._set(tf.add_paragraph(), f"US {C.MDD_US}  |  UK {C.MDD_UK}  |  JP {C.MDD_JP}", 15, color=L.GOOD)
    L._set(tf.add_paragraph(), "vs fixed-momentum -62% to -72%", 14, color=L.BAD)
    L.takeaway(s, "Works where momentum already works (US, UK) and where it does not (Japan, baseline ~0). Only the HMM features are region-specific.")
    L.notes(s, "Japan is the strongest evidence: baseline momentum is absent there, yet regime-conditioning produces 0.57. Rules out 'just a better momentum implementation'.")


def s11_conclusions(prs, A):
    s = L.new_slide(prs)
    L.title(s, "Conclusions & future work")
    cols = [
        ("Contributions", ["Novel filter + nonlinear model combination",
                            "Concrete stock-level mechanism (4 modes)"]),
        ("Limitation", ["Reversal-failure in a sustained bear",
                        f"dot-com: {C.DOTCOM_XGB} ({C.DOTCOM_MDD} MDD)"]),
        ("Future work", ["Regime forecasting / trajectory",
                         "Neural nets; D&M exposure overlay"]),
    ]
    x = Inches(0.7)
    for head, items in cols:
        tf = L._box(s, x, Inches(1.7), Inches(3.9), Inches(3.5)).text_frame
        L._set(tf.paragraphs[0], head, 18, color=L.ACCENT, bold=True)
        for it in items:
            L._set(tf.add_paragraph(), it, 15, color=L.INK)
        x += Inches(4.05)
    L.takeaway(s, "Momentum's cross-section reorganizes across regimes; a continuous regime signal routes a nonlinear model between winner selection and reversal.")
    L.notes(s, "Close on the two contributions and one future direction. Be honest about the dot-com reversal-failure mode and the D&M circuit-breaker complement.")


MAIN_BUILDERS = [s01_title, s02_motivation, s03_literature, s04_methodology_hmm,
                 s05_methodology_xgb, s06_results, s07_mechanism_flip,
                 s08_mechanism_clusters, s09_novel, s10_robustness, s11_conclusions]

BACKUP_BUILDERS = []   # populated in Task 6


def _divider(prs, text):
    s = L.new_slide(prs)
    tf = L._box(s, Inches(0.8), Inches(3.0), Inches(11.7), Inches(1.5)).text_frame
    L._set(tf.paragraphs[0], text, 32, color=L.ACCENT, bold=True, align=PP_ALIGN.CENTER)


def build_presentation():
    prs = Presentation()
    prs.slide_width, prs.slide_height = L.SLIDE_W, L.SLIDE_H
    A = ensure_pngs()
    for build in MAIN_BUILDERS:
        build(prs, A)
    _divider(prs, "Backup slides  -  for the Q&A")
    for build in BACKUP_BUILDERS:
        build(prs, A)
    return prs


def main():
    prs = build_presentation()
    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
```

Note: delete the stray empty `for (name, sh) in zip(panels, C.CLUSTER_SHARPES): pass` loop in `s08` if you copy literally — it is a no-op; the real loop follows it. (Left out intentionally below in the clean version.)

- [ ] **Step 4: Remove the no-op loop in s08**

Delete these two lines from `s08_mechanism_clusters` (they do nothing):
```python
    for (name, sh) in zip(panels, C.CLUSTER_SHARPES):
        pass
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_main_deck_structure -v`
Expected: PASS — 11 main + 1 divider + 0 backup = 12 slides for now; the assertion uses `EXPECTED_BACKUP` (12), so this test will FAIL until Task 6 adds the backups. Temporarily assert against the partial count: change the test's first assert to `>= C.EXPECTED_MAIN + 1` to confirm main slides build, then restore `==` in Task 6.

Run instead now: `python -c "from presentation.build_defense_deck import build_presentation as b; print(len(b().slides))"`
Expected: prints `12`.

- [ ] **Step 6: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add presentation/build_defense_deck.py && git commit -m "presentation: 11 main slides + orchestrator

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Backup slides B1-B12

**Files:**
- Modify: `presentation/build_defense_deck.py` (populate `BACKUP_BUILDERS`)
- Test: `presentation/test_deck.py` (restore strict count)

- [ ] **Step 1: Restore the strict slide-count assertion**

In `test_main_deck_structure`, ensure the first assertion reads:
```python
    assert len(prs.slides) == C.EXPECTED_MAIN + 1 + C.EXPECTED_BACKUP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py::test_main_deck_structure -v`
Expected: FAIL — got 12, expected 24 (backups not built yet).

- [ ] **Step 3: Populate `BACKUP_BUILDERS` in build_defense_deck.py**

Replace `BACKUP_BUILDERS = []` with a data-driven builder over `C.BACKUP_SLIDES`:
```python
def _backup_builder(entry):
    title_text, figure, bullet_items, notes_text = entry

    def build(prs, A):
        fig_path = None
        if figure and figure.startswith("ASSET:"):
            fig_path = A[figure.split(":", 1)[1]]
        elif figure:
            fig_path = FIG / figure
        L.content_slide(
            prs, title_text=title_text, figure_path=fig_path,
            bullets_items=bullet_items, notes_text=notes_text,
        )
    return build


BACKUP_BUILDERS = [_backup_builder(e) for e in C.BACKUP_SLIDES]
```

Place this block ABOVE `build_presentation()` (so the list exists at call time). Remove the old `BACKUP_BUILDERS = []` line.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py -v`
Expected: PASS (all tests; deck = 24 slides, every main slide has notes).

- [ ] **Step 5: Commit**

```bash
cd /Users/giladgang/momentum_regime && git add presentation/build_defense_deck.py presentation/test_deck.py && git commit -m "presentation: 12 backup slides (data-driven)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Build the deck + verification gate

**Files:**
- Generates: `presentation/momentum_defense.pptx`

- [ ] **Step 1: Build the deck**

Run: `cd /Users/giladgang/momentum_regime && python -m presentation.build_defense_deck`
Expected: `wrote .../presentation/momentum_defense.pptx  (24 slides)`.

- [ ] **Step 2: Run the full test suite**

Run: `cd /Users/giladgang/momentum_regime && python -m pytest presentation/test_deck.py -v`
Expected: PASS (all tests).

- [ ] **Step 3: Verification gate - numbers trace to canonical sources**

Manually cross-check the deck's numbers against the thesis sources (do NOT trust memory):
```bash
cd /Users/giladgang/momentum_regime && grep -E "Sharpe|alpha|24.1|4.81|1.11|1.53|0.84" tables/table_performance.tex tables/table_regime_sharpe.tex tables/table_factor_alphas.tex
```
Confirm: XGB Sharpe 1.11, panic 1.53 / calm 0.84, FF6 alpha 24.1% (t 4.81). If any source value differs from `content.py`, fix `content.py` (the source wins) and rebuild.

- [ ] **Step 4: Visual spot-check**

Run: `open /Users/giladgang/momentum_regime/presentation/momentum_defense.pptx`
Confirm by eye: title slide logo renders; slide 4 pipeline + regime figure; slide 8 shows four cluster panels with captions; no overflowing text past slide edges; backup divider present. Note any figure that is the wrong one or any slide breaching the ~15-line guideline. Fix and rebuild if needed.

- [ ] **Step 5: Final commit**

```bash
cd /Users/giladgang/momentum_regime && git add -A presentation && git commit -m "presentation: build verified 24-slide defense deck

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(The `.pptx` and `assets/` are gitignored; this commits only any remaining source tweaks.)

---

## Self-review (against the spec)

**Spec coverage:** Title (s01), motivation/question (s02), literature & gap (s03), methodology I/II = Ch 3 (s04-s05), headline results (s06), mechanism split into two (s07-s08), why-novel (s09), robustness (s10), conclusions/future (s11) — all 11 present. Backup B1-B12 present (Task 6 / content). Speaker notes on every main slide (test enforces). PowerPoint via python-pptx (Tasks 3,5,6). Clean-minimal + Tilburg logo (s01). PDF→PNG rasterizing (Task 4). Numbers from canonical sources + stale-number guard (Task 2 + Task 7 gate). Gadi files untouched. **No gaps found.**

**Placeholder scan:** No "TBD/TODO/handle edge cases". The one no-op loop in s08 is explicitly removed in Task 5 Step 4. The Task 5 Step 5 note flags the intentional temporary-count situation resolved in Task 6.

**Type/name consistency:** `ensure_pngs()` returns a dict keyed `logo/gibbs/acf/riskav`; `s01` uses `A["logo"]`; backup `ASSET:gibbs|acf|riskav` map to those keys. `build_presentation()` used by tests and `main()`. `content_slide`, `figure_fit`, `pipeline`, `stat`, `table`, `notes`, `_box`, `_set` all defined in deck_lib and called consistently. `EXPECTED_MAIN`/`EXPECTED_BACKUP` defined in content, used in test. Consistent.
