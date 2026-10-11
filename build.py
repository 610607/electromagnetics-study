"""Build the fifth-edition PDFs (Chinese and/or English) from the supplied sources.

    python build.py                       # both languages, supplied figures
    python build.py --language zh         # Chinese only
    python build.py --regenerate-figures  # redraw 2-D figures (matplotlib) first
    python build.py --regenerate-figures --with-3d   # also re-render 3-D figures (needs Playwright + Chromium)
    python build.py --checks              # run all numerical and bilingual checks first

Both editions are compiled with XeLaTeX (three passes).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"zh": "Electromagnetics_Quasistatic_Learning_Edition_v5_ZH.pdf",
         "en": "Electromagnetics_Quasistatic_Learning_Edition_v5_EN.pdf"}
MAIN = {"zh": "main_zh", "en": "main_en"}


def run(cmd: list[str], log: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
        return
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    if p.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Build failed; inspect {log}\n{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", choices=["en", "zh", "both"], default="both")
    ap.add_argument("--regenerate-figures", action="store_true")
    ap.add_argument("--with-3d", action="store_true", help="also re-render the shaded 3-D figures")
    ap.add_argument("--checks", action="store_true")
    a = ap.parse_args()
    langs = ["zh", "en"] if a.language == "both" else [a.language]
    if a.language == "both" and not any((ROOT / "en").glob("ch*.tex")):
        print("The English chapter text is not available yet (en/ is empty); building the Chinese edition only.")
        langs = ["zh"]
    out = ROOT / "compiled"
    out.mkdir(exist_ok=True)
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    if a.checks:
        for script in ["numerics.py", "concept_checks.py", "extended_checks.py"]:
            run([sys.executable, script])
        for script in sorted((ROOT / "checks").glob("check_*.py")):
            run([sys.executable, str(script)])
        run([sys.executable, "validate_source.py"])
    if not shutil.which("xelatex"):
        raise RuntimeError("xelatex was not found; install a full TeX distribution.")
    for lang in langs:
        if a.regenerate_figures:
            env = os.environ.copy()
            env["EM_LANG"] = lang
            sys.path.insert(0, str(ROOT / "figsrc"))
            names = None
            if not a.with_3d:
                import importlib
                import style  # noqa: F401
                for m in sorted(p.stem for p in (ROOT / "figsrc").glob("fig_*.py")):
                    if m != "fig_3d":
                        importlib.import_module(m)
                names = sorted(style.REGISTRY)
            run([sys.executable, "figsrc/build.py", *(names or [])], env=env)
        main_tex = MAIN[lang]
        for ext in ["aux", "out", "toc", "cmp"]:
            (build / f"{main_tex}.{ext}").unlink(missing_ok=True)
        for k in range(1, 4):
            run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", f"{main_tex}.tex"],
                build / f"{lang}_pass_{k}.txt")
        log = (build / f"{main_tex}.log").read_text(errors="replace")
        problems = [bad for bad in ["There were undefined references", "multiply defined", "Missing character:"] if bad in log]
        if problems:
            print(f"warning ({lang}): {problems}; see build/{main_tex}.log", file=sys.stderr)
        shutil.copy2(build / f"{main_tex}.pdf", out / NAMES[lang])
        print("Written:", out / NAMES[lang])


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        print(e, file=sys.stderr)
        raise SystemExit(1)
