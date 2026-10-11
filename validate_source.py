"""Check that the Chinese and English sources have the same structure.

For every file zh/X.tex there must be en/X.tex with
  * the same labels and citations (as multisets),
  * the same cross-reference targets (as sets),
  * the same figure calls, in the same order, and every figure present in both
    figures_zh/ (Chinese) and figures/ (English),
  * the same sequence of teaching boxes (example, compare, pitfall, ...),
  * the same displayed mathematics, block by block, after removing text inside
    \\text{...}-like commands and whitespace.
Inline mathematics is compared as a multiset and differences are reported as
warnings (a translation may legitimately rephrase a sentence around a symbol).

    python validate_source.py               # all pairs; exit status 1 on any hard mismatch
    python validate_source.py ch05_matter   # only the listed pairs
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DISPLAY = ["equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
           "keyeq", "keyeq*"]
BOXES = ["chapterguide", "guidegoals", "intuition", "example", "compare", "compare*", "pitfall", "deeper",
         "takeaways", "selfcheck", "answers", "table", "figure"]
REFCMDS = ["ref", "eqref", "chref", "secref", "figref", "tabref", "exref", "appref", "pgref", "nameref",
           "pageref", "autoref", "hyperref"]
TEXTCMDS = ["text", "textup", "textrm", "textsf", "textit", "mbox", "mathrm", "operatorname", "ZE", "intertext",
            "shortintertext", "tag"]


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def brace_arg(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' ; return (content, index after closing brace)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def drop_text(s: str) -> str:
    """Replace \\text{...} and friends (all their brace arguments) by a token."""
    out, i = [], 0
    pat = re.compile(r"\\(" + "|".join(TEXTCMDS) + r")\b\s*")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        n = 0
        while j < len(s) and s[j] == "{" and n < 2:
            _, j = brace_arg(s, j)
            n += 1
            while j < len(s) and s[j] in " \t":
                j += 1
        out.append("T")
        i = j
    return "".join(out)


def normalize_math(s: str) -> str:
    s = drop_text(s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\(nonumber|notag)\b", "", s)
    s = re.sub(r"[\u3000-\u303f\uff00-\uffef\u4e00-\u9fff]+", "T", s)   # stray CJK
    s = re.sub(r"\\[,;:! ]|~", "", s)                                     # spacing
    s = re.sub(r"\\(quad|qquad)\b", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[.,;]+(?=(\\\\|&|$))", "", s)                            # trailing punctuation
    s = re.sub(r"[.,;]+$", "", s)
    return s


def args_of(text: str, cmds: list[str]) -> list[str]:
    pat = re.compile(r"\\(" + "|".join(cmds) + r")\*?(?:\[[^\]]*\])?\{([^}]*)\}")
    return [m.group(2) for m in pat.finditer(text)]


def cites(text: str) -> list[str]:
    out = []
    for a in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", text):
        out += [k.strip() for k in a.split(",") if k.strip()]
    return out


def figures(text: str) -> list[str]:
    calls = re.findall(r"\\fig(?:H)?\*?(?:\[[^\]]*\])?\{([^}]+)\}", text)
    calls += [Path(p).stem for p in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)]
    return calls


def env_sequence(text: str, names: list[str]) -> list[str]:
    pat = re.compile(r"\\begin\{(" + "|".join(re.escape(n) for n in names) + r")\}")
    return [m.group(1) for m in pat.finditer(text)]


def display_blocks(text: str) -> list[tuple[str, str]]:
    pat = re.compile(r"\\begin\{(" + "|".join(re.escape(n) for n in DISPLAY) + r")\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{\1\}")
    return [(m.group(1), m.group(2)) for m in pat.finditer(text)]


def inline_math(text: str) -> list[str]:
    t = re.sub(r"\\begin\{(" + "|".join(re.escape(n) for n in DISPLAY) + r")\}[\s\S]*?\\end\{\1\}", "", text)
    t = t.replace(r"\$", "")
    return [normalize_math(m) for m in re.findall(r"\$([^$]+)\$", t) if normalize_math(m)]


def compare_pair(zh: Path, en: Path) -> tuple[list[str], list[str], dict]:
    errors, warnings = [], []
    a, b = strip_comments(zh.read_text(encoding="utf-8")), strip_comments(en.read_text(encoding="utf-8"))
    name = zh.name
    la, lb = Counter(args_of(a, ["label"])), Counter(args_of(b, ["label"]))
    if la != lb:
        errors.append(f"{name}: labels differ; zh only {sorted((la - lb).elements())}, en only {sorted((lb - la).elements())}")
    ca, cb = Counter(cites(a)), Counter(cites(b))
    if set(ca) != set(cb):
        errors.append(f"{name}: citations differ; zh only {sorted(set(ca) - set(cb))}, en only {sorted(set(cb) - set(ca))}")
    ra, rb = set(args_of(a, REFCMDS)), set(args_of(b, REFCMDS))
    if ra != rb:
        errors.append(f"{name}: reference targets differ; zh only {sorted(ra - rb)}, en only {sorted(rb - ra)}")
    fa, fb = figures(a), figures(b)
    if fa != fb:
        errors.append(f"{name}: figure calls differ\n   zh {fa}\n   en {fb}")
    for f in fa:
        for d in ["figures_zh", "figures"]:
            if not (ROOT / d / f"{f}.pdf").exists() and not (ROOT / d / f"{f}.png").exists():
                errors.append(f"{name}: figure file {d}/{f}.pdf is missing")
    sa, sb = env_sequence(a, BOXES), env_sequence(b, BOXES)
    if sa != sb:
        errors.append(f"{name}: teaching-box sequence differs\n   zh {sa}\n   en {sb}")
    da, db = display_blocks(a), display_blocks(b)
    if len(da) != len(db):
        errors.append(f"{name}: {len(da)} displayed equations in zh but {len(db)} in en")
    for k, ((ea, xa), (eb, xb)) in enumerate(zip(da, db), 1):
        if ea != eb or normalize_math(xa) != normalize_math(xb):
            errors.append(f"{name}: displayed block {k} differs ({ea} vs {eb})\n   zh {normalize_math(xa)[:160]}\n   en {normalize_math(xb)[:160]}")
    ia, ib = Counter(inline_math(a)), Counter(inline_math(b))
    only_a, only_b = ia - ib, ib - ia
    n = max(sum(ia.values()), 1)
    diff = sum(only_a.values()) + sum(only_b.values())
    if diff:
        warnings.append(f"{name}: inline math differs in {diff} of {n} items; e.g. zh {list(only_a)[:4]}, en {list(only_b)[:4]}")
    row = {"file": name, "labels": sum(la.values()), "citations": sum(ca.values()), "figures": len(fa),
           "boxes": len(sa), "display_blocks": len(da), "inline_math": n, "inline_math_differences": diff}
    return errors, warnings, row


def main(argv: list[str]) -> int:
    """With file stems as arguments (e.g. ch05_matter), only those pairs are compared."""
    errors, warnings, rows, pending = [], [], [], []
    only = {Path(a).stem for a in argv}
    zh_files = sorted(p for p in (ROOT / "zh").glob("*.tex") if not only or p.stem in only)
    for zh in zh_files:
        en = ROOT / "en" / zh.name
        if not en.exists():
            pending.append(zh.name)
            continue
        e, w, r = compare_pair(zh, en)
        errors += e
        warnings += w
        rows.append(r)
    for en in sorted((ROOT / "en").glob("*.tex")):
        if only:
            break
        if not (ROOT / "zh" / en.name).exists():
            errors.append(f"English file en/{en.name} has no Chinese counterpart")
    # both main files must input the same chapters in the same order
    inputs = {"zh": [], "en": []}
    for lang in ([] if only else ["zh", "en"]):
        t = strip_comments((ROOT / f"main_{lang}.tex").read_text(encoding="utf-8"))
        inputs[lang] = [Path(p).name for p in re.findall(r"\\input(?:chapter)?\{(?:zh|en)/([^}]+)\}", t)]
    if inputs["zh"] != inputs["en"]:
        errors.append(f"main files input different chapters:\n   zh {inputs['zh']}\n   en {inputs['en']}")
    out = ROOT / "build"
    out.mkdir(exist_ok=True)
    (out / "bilingual_structure_checks.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for w in warnings:
        print("warning:", w)
    if pending:
        print(f"English translation pending for {len(pending)} file(s): {', '.join(pending)}")
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1
    print(f"All {len(rows)} bilingual file pairs match: labels, citations, references, figures, "
          f"teaching boxes and displayed mathematics.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
