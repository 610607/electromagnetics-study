# 准静态仿真电磁学 · Electromagnetics for Quasistatic Simulation

**Fifth edition, learning edition**

**Status (October 2026):** the Chinese edition is complete (`compiled/Electromagnetics_Quasistatic_Learning_Edition_v5_ZH.pdf`). The English edition is prepared but not yet translated: every figure already exists in English (`figures/`), and the English title page, part pages and bibliography are in place, but the chapter text in `en/` is still to be written. `validate_source.py` lists the files still waiting for translation, and `build.py` builds only the Chinese edition until `en/` has chapters.

状态（2026 年 10 月）：中文版已完成；英文版的插图、标题页、各部分导言与参考文献已就绪，正文尚待翻译。

A question-driven course that starts at Maxwell's equations and builds the physical intuition, approximations, potentials and numerical methods behind quasistatic simulation: transcranial electrical and magnetic stimulation (TES, TMS), eddy currents, ferromagnetic materials, lumped magnetic circuits, permanent magnets and yokes, finite elements, MRI gradient and shim fields, coil design, neural response, and the boundary where RF and SAR stop being quasistatic.

本书从 Maxwell 方程出发，以问题为线索，逐步建立准静态仿真所需的物理直觉、近似、势函数与数值方法，覆盖 TES/TMS、涡流、铁磁材料、集总磁路、永磁体与磁轭、有限元、MRI 梯度与匀场、线圈设计、神经响应，以及射频与 SAR 何时不再是准静态问题。

## How the book is built / 全书结构

| Part | Chapters |
|---|---|
| I 场的语言 · The language of fields | 1 A question that leads to the whole course · 2 The geometry of fields |
| II 电与磁 · Electricity and magnetism | 3 Charge, field, potential · 4 Current, magnetic field, induction · 5 Fields in matter · 6 Interfaces · 7 Four standard systems |
| III 时间、近似与表示 · Time, approximation, representation | 8 Static, quasistatic and waves · 9 Potentials |
| IV 核心场问题 · Core field problems | 10 TES · 11 TMS · 12 Eddy currents · 13 Ferromagnetic materials · 14 Lumped magnetic circuits · 15 Magnets, yokes and MRI magnets |
| V 有限元 · Finite elements | 16 Scalar FEM · 17 Vector FEM and nonlinear solves |
| VI 应用 · Applications | 18 Common framework · 19 MRI field observables · 20 Inverse coil design · 21 From field to neural response · 22 RF and SAR |
| VII 综合与验证 · Synthesis and validation | 23 Integrated projects · 24 Verification, uncertainty, decisions |
| Appendices | A Vector identities · B Symbols and units · C Index of comparisons |

Every chapter follows the same learning path: a guiding question → intuition → derivation → worked examples → common pitfalls → side-by-side comparison boxes ("same / different / one sentence") → takeaways → self-check with answers. Appendix C collects all comparison boxes. Colours carry meaning throughout text and figures (blue electric, orange magnetic, green currents, purple potentials, warm grey materials).

每章遵循同一学习路径：核心问题 → 直觉 → 推导 → 例题 → 常见误区 → 对照框（相同／不同／一句话）→ 本章要点 → 自测与答案。附录 C 汇总全书对照框。颜色在正文与插图中含义一致（蓝：电；橙：磁；绿：电流；紫：势；暖灰：材料）。

## Repository layout / 文件结构

```
main_zh.tex, main_en.tex     entry points (title page, parts, chapter order)
preamble.tex                 shared layout: fonts, colours, teaching boxes, cross-reference helpers
zh/, en/                     chapters, front matter and appendices (same file names, same labels)
references_zh.tex, _en.tex   bibliography (real, linked sources)
figures_zh/, figures/        figure PDFs (Chinese / English)
figsrc/                      figure sources: style.py (palette, helpers), fig_chNN.py, fig_3d.py
figsrc/render3d.py, three/   shaded 3-D figures: geometry in Python, rendered by three.js in headless Chromium
checks/check_chNN.py         every number quoted in a chapter, recomputed and asserted
numerics.py, concept_checks.py, extended_checks.py   shared teaching computations (P1 FEM, B–H curve, …)
validate_source.py           Chinese/English structure check (labels, citations, figures, boxes, displayed maths)
tools/check_chapter.py       compile one chapter in isolation and report layout problems
build.py                     builds both PDFs (optionally regenerating figures and running all checks)
docs/                        authoring notes: chapter plan and label registry (PLAN.md), style guide (STYLE.md),
                             and the briefs for reviewing and translating chapters (agent_prompts/)
compiled/                    the built PDF(s)
```

## Build / 编译

Requirements:
- TeX Live 2023 or newer with **XeLaTeX** (both editions use XeLaTeX), including `fontspec`, `unicode-math`, `xeCJK`, `tcolorbox`, `titlesec`, `titletoc`, `fancyhdr`, `booktabs`, `tabularx`, `hyperref`, `tikz`.
- Fonts: STIX Two Text and STIX Two Math, Inter, Noto Serif CJK SC and Noto Sans CJK SC (Chinese edition). Fallbacks are built in (TeX Gyre Termes/Heros, Latin Modern Math, Fandol); no fonts are distributed here.
- Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `fonttools` (see `requirements.txt`) to regenerate 2-D figures and run the checks.
- Only for re-rendering the shaded 3-D figures: the Python package `playwright` and its Chromium (`python -m playwright install chromium`). three.js (revision 187dev) is vendored in `figsrc/three/` under its MIT licence.

```bash
python build.py                         # PDFs from the supplied figures → compiled/ (Chinese only until en/ has chapters)
python build.py --language zh           # Chinese only
python build.py --checks                # run all numerical checks and the bilingual structure check first
python build.py --regenerate-figures    # redraw 2-D figures (add --with-3d to re-render 3-D figures)
python checks/check_ch14.py             # one chapter's numbers
python tools/check_chapter.py zh/ch14_lumped.tex   # compile one chapter and report layout problems
```

## What the numerical checks do and do not establish / 数值检查的范围

The scripts recompute every number quoted in the text from stated teaching inputs: field and potential examples, a P1 finite-element model of a conducting square with an inclusion (TES and TMS on one mesh), electrode models, eddy-current shell time constants, a synthetic B–H curve, lumped magnetic circuits checked against 2-D nonlinear FEM, demagnetizing factors from Osborn's closed forms, gradient and concomitant fields, cable-equation and strength–duration results, and SAR quadratic forms and averages. They are teaching calculations: they do not replace measured B–H data, validated anatomical models, a production 3-D or full-wave solver, or clinical and regulatory safety assessment.

脚本根据书中给出的教学输入重算正文中引用的每一个数值。它们是教学计算，不能代替实测 B–H 数据、经过验证的解剖模型、生产级三维或全波求解器，也不能代替临床与法规层面的安全评估。

See `REVISION_NOTES.md` for what changed from the fourth edition.
