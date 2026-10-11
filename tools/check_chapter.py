"""Compile one or more chapter files in a harness where every registry label resolves.

    python3 tools/check_chapter.py zh/ch07_systems.tex [zh/ch08_regimes.tex ...] [--lang en]

Writes build/check_<first-file-stem>/check.pdf and contact sheets sheet_*.png (6 pages each),
and prints LaTeX errors, overfull boxes, undefined references and missing glyphs.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def registry() -> list[str]:
    plan = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    block = plan.split("## Label registry", 1)[1].split("```", 2)[1]
    # every figure that exists may be referred to as fig:<file name>
    figs = [f"fig:{p.stem}" for p in sorted((ROOT / "figures_zh").glob("*.pdf"))]
    return block.split() + figs

VTAGS = {"id:div": "V1", "id:stokes": "V2", "id:zeros": "V3", "id:product": "V4", "id:crossdiv": "V5",
         "id:curlcurl": "V6", "id:scalaribp": "V7", "id:curlibp": "V8", "id:path": "V9"}
CHAPTERS = ["intro", "calculus", "electric", "magnetic", "matter", "interfaces", "systems", "regimes",
            "potentials", "tes", "tms", "eddy", "bh", "lumped", "magnets", "femscalar", "femvector",
            "framework", "mri", "coils", "neuro", "rf", "projects", "practice"]

def main(argv: list[str]) -> None:
    lang = "zh"
    if "--lang" in argv:
        i = argv.index("--lang"); lang = argv[i + 1]; del argv[i:i + 2]
    files = [Path(a) for a in argv]
    if not files:
        raise SystemExit(__doc__)
    out = ROOT / "build" / f"check_{lang}_{files[0].stem}"
    out.mkdir(parents=True, exist_ok=True)
    fake = ["\\makeatletter", "\\def\\fakelabel#1#2{\\expandafter\\gdef\\csname r@#1\\endcsname{{#2}{0}{}{}{}}}"]
    for lab in registry():
        if lab.startswith("ch:"):
            num = str(CHAPTERS.index(lab[3:]) + 1) if lab[3:] in CHAPTERS else "0"
        elif lab.startswith("app:"):
            num = {"app:identities": "A", "app:symbols": "B", "app:compare": "C"}.get(lab, "A")
        elif lab in VTAGS:
            num = VTAGS[lab]
        else:
            num = "0.0"
        fake.append(f"\\fakelabel{{{lab}}}{{{num}}}")
    fake.append("\\makeatother")
    (out / "registry.tex").write_text("\n".join(fake), encoding="utf-8")
    m = re.match(r"ch(\d+)", files[0].stem)
    start = int(m.group(1)) - 1 if m else 0
    body = "\n".join(f"\\input{{{(ROOT / f).as_posix()[:-4] if str(f).endswith('.tex') else (ROOT / f).as_posix()}}}" for f in files)
    tex = f"""\\documentclass[11pt,a4paper,oneside,openany]{{book}}
\\newif\\ifChinese
\\Chinese{'true' if lang == 'zh' else 'false'}
\\input{{preamble}}
\\input{{{(out / 'registry').as_posix()}}}
\\begin{{document}}
\\phantomsection\\label{{nav:contents}}\\phantomsection\\label{{nav:compare}}
\\setcounter{{chapter}}{{{start}}}
{body}
\\input{{references_{lang}}}
\\end{{document}}
"""
    (out / "check.tex").write_text(tex, encoding="utf-8")
    for k in range(2):
        p = subprocess.run(["xelatex", "-interaction=nonstopmode", f"-output-directory={out}", str(out / "check.tex")],
                           cwd=ROOT, capture_output=True, text=True)
    log = (out / "check.log").read_text(errors="replace")
    errs = [l for l in log.splitlines() if l.startswith("!")]
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*lines (\d+--\d+)", log)
    undef = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    cites = sorted(set(re.findall(r"Citation `([^']+)' on page", log)))
    missing = sorted(set(re.findall(r"Missing character: There is no (.) ", log)))
    print("PDF:", out / "check.pdf")
    print("errors:", errs[:15] or "none")
    print("overfull hboxes (>1pt):", [o for o in over if float(o[0]) > 1.0][:20] or "none")
    print("undefined references:", undef or "none")
    print("undefined citations:", cites or "none")
    print("missing glyphs:", missing or "none")
    if (out / "check.pdf").exists():
        for f in out.glob("pg-*.png"):
            f.unlink()
        subprocess.run(["pdftoppm", "-r", "55", "-png", str(out / "check.pdf"), str(out / "pg")], check=True)
        from PIL import Image
        pages = sorted(out.glob("pg-*.png"))
        for f in out.glob("sheet_*.png"):
            f.unlink()
        if pages:
            w, h = Image.open(pages[0]).size
            for s in range(0, len(pages), 6):
                chunk = pages[s:s + 6]
                sheet = Image.new("RGB", (3 * w, 2 * h), "white")
                for i, f in enumerate(chunk):
                    sheet.paste(Image.open(f), ((i % 3) * w, (i // 3) * h))
                sheet.save(out / f"sheet_{s // 6 + 1}.png")
            print(f"{len(pages)} pages; contact sheets:", sorted(str(p) for p in out.glob('sheet_*.png')))
            # ink in the side margins (catches displays that overflow inside boxes)
            import numpy as np
            bad = []
            for k, f in enumerate(pages, 1):
                a = np.asarray(Image.open(f).convert("L"))
                h, w = a.shape
                mm = w / 210.0
                rows = slice(int(22 * mm), int(272 * mm))
                left = a[rows, : int(27.5 * mm)]
                right = a[rows, int(182.5 * mm):]
                if (left < 200).sum() > 3 or (right < 200).sum() > 3:
                    bad.append(k)
            print("pages with ink in the side margins:", bad or "none")

if __name__ == "__main__":
    main(sys.argv[1:])
