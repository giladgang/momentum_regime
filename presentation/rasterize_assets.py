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
