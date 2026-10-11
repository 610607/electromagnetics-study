# Writing and figure style guide (fifth edition, Chinese master)

Read `zh/ch05_matter.tex` (finished exemplar) and its PDF `compiled/样章_第5章_物质中的电与磁.pdf`
before writing. Match its voice, density, box usage and figure quality.

## 1. Goal: the reader must LEARN, not read a catalogue
The author said: “目的是让人学会而非罗列”. Concretely:
- **Question first.** Every chapter opens with a concrete puzzle (the `\guidequestion`) and a short
  opening section that sets up a physical situation. Every section starts with *why* we need it.
- **Just-in-time concepts.** Introduce a quantity when the story needs it; never a list of
  definitions. Prefer: situation → intuition/picture → derivation → worked numbers → what can go wrong.
- **Build on running examples** (capacitor, two-layer conductor, long solenoid, gapped core,
  conducting square with inclusion, two-element bar). Reuse the same numbers across chapters and refer
  back instead of re-deriving.
- **Similarities and differences are a first-class topic.** Whenever two concepts could be
  confused or are analogous, add a `compare` box (typically 2–4 per chapter). Rows: 3–6 aspects; end with
  `\cmpsum{...}` — one memorable sentence.
- **Consolidate caveats.** The old edition hedges in almost every sentence. Keep every caveat that
  matters physically, but gather them: one `pitfall` box, or one “适用范围” sentence at the end of a
  section, or a `deeper` box. Do not drop correctness; drop repetition.
- **Explain, don't just assert.** Show the one-line reason (“因为旋度的散度为零”) for each step.
  Use plain words before symbols. Use concrete numbers with units.
- **Progressive depth.** Advanced technical material (FEM details, convention subtleties, function
  spaces) that a first-time reader can skip goes into `deeper` boxes, not the main line.
- End each chapter with a sentence that hands over to the next chapter's question.

## 2. Chapter skeleton (Chinese file `zh/chNN_slug.tex`)
```latex
\chapter{标题}\label{ch:slug}
\begin{chapterguide}
\guidequestion{一个具体问题？}
\begin{guidegoals}
\item 3–5 个可检验的能力（动词开头：推导/区分/计算/判断/解释）
\end{guidegoals}
\guideprereq{\chref{...}；\secref{...}}
\end{chapterguide}
\section{打开问题的情景}\label{sec:slug-...}
...
\begin{intuition}[可选小标题] ... \end{intuition}          % 1–3 per chapter
\begin{keyeq} ... \label{eq:...}\end{keyeq}                 % the 2–5 most important equations
\begin{example}{标题}\label{ex:...} 题目 \solution 解答 \end{example}   % ≥1 per chapter
\begin{compare}{标题}{A 列标题}{B 列标题}
\cmprow{方面}{A}{B}
\cmpsum{一句话区分}
\end{compare}
\begin{pitfall}[误区的说法] ... \end{pitfall}
\begin{deeper}[可选标题] ... \end{deeper}
\fig{figname}{图注}{fig:figname}        % [H]-placed variant: \figH ; optional width: \fig[.8]{...}
...
\begin{takeaways}\begin{itemize}\item ...\end{itemize}\end{takeaways}   % 4–6 bullets
\begin{selfcheck}\begin{enumerate}\item ...\end{enumerate}\end{selfcheck} % 4–7 questions
\begin{answers}\item ...\end{answers}  % one \item per question, concise
```
- Sections: `\section` (numbered, in the TOC). Sub-headings: `\subsection` (printed without a
  number; use sparingly). Run-in headings: `\subsubsection{...}`.
- No `\clearpage`, no manual vertical spacing, no `\newpage`.
- Tables: `table` float with `\caption` above, `tabularx` with `booktabs` rules, column type `Y`.

## 3. Language and notation
- 简体中文，讲解语气（可以用“我们”），避免翻译腔和长串“不是……而是……”。One idea per paragraph.
- Key terms in **bold** at first appearance, optionally with English in parentheses:
  `\textbf{磁准静态}（magnetoquasistatic, MQS）`.
- Full-width Chinese punctuation “，。；：（）“”” ; math stays in `$...$`.
- Vectors `\vb{E}`; tensors `\sig`, `\eps`, `\boldsymbol\mu`; `\dd`, `\dV`, `\dS`, `\jj`;
  phasors `\widetilde{\vb E}` with the **peak** convention `e^{\jj\omega t}`; units upright:
  `$100\,\mathrm{V/m}$`. Identity references: `\eqref{id:div}` prints “(V1)”.
- Cross references only through the helpers: `\chref{}`, `\secref{}`, `\figref{}`, `\tabref{}`,
  `\exref{}`, `\eqref{}`, `\appref{}`. Only labels from the registry in `docs/PLAN.md` or labels
  defined in your own chapters.
- Citations: only keys that exist in `references_zh.tex`:
  haus, itisdielectric, itislf, simnibs, cem, comsolbh, femm, mfem, getdp, rf, concomitant,
  bfieldtheory, bfieldsoftware, rattay, davids, polarization, magnetization, vectorpoisson,
  vectorboundary, demagnetizing, qslimits, mfemem, simnibsfields, griffiths, purcell, fritsch, nedelec, oberkampf.
  Approved new keys (the editor adds the entries): `fitzgerald` (A. E. Fitzgerald, C. Kingsley Jr.,
  S. D. Umans, *Electric Machinery*, 7th ed., McGraw-Hill, 2014 — Ch. 1 magnetic circuits, inductance,
  energy, permanent magnets) and `osborn` (J. A. Osborn, “Demagnetizing factors of the general
  ellipsoid”, Phys. Rev. 67, 351–357, 1945). Do not invent any other reference. If you truly need a new
  one, it must be real and verifiable; put the full entry in your final report instead of the file.
- Numbers: every number in the text must either come from the old edition (same inputs, same
  results) or be computed by a script you add under `checks/check_chNN.py` that asserts it
  (run it). No invented tissue values, device specs or thresholds. Teaching inputs must be labelled as
  such once (not in every sentence).
- The author requires methods traceable to sources: do not invent algorithms or empirical rules; when a
  model needs a correction factor (e.g. gap fringing), say it must be calibrated against a field
  solution or measurement.

## 4. Figures
- One Python module per chapter: `figsrc/fig_chNN.py` (look at `figsrc/fig_matter.py` and
  `figsrc/style.py`). Each figure is a function decorated `@figure("name")` that returns a Figure.
  Build and look at them:
  `EM_LANG=zh python3 figsrc/build.py name1 name2` (from the repository root) and the same with
  `EM_LANG=en`. Outputs: `figures_zh/name.{pdf,png,svg}` and `figures/…`.
- **Every text label uses `T("中文", "English")`** so both editions get native labels.
- Colour semantics (never deviate): electric quantities `ELEC` blue; magnetic `MAG` vermilion;
  currents/charge flow `CURR` green; potentials `POT` purple; materials/regions light fills
  (`MATL_T` tissue-like sand, `IRON_T` iron/yoke grey); structure/accent `ACCENT`; muted text `MUTED`.
  Light fills via `tint(color, 0.1–0.2)`.
- Size: width `FIG_W` (6.0 in = text width) so 8.5–10 pt figure text matches the body; height as
  small as the content allows (2.2–3.6 in typical). Use `canvas()` for diagrams, `plt.subplots` with
  `clean_axes()` for plots.
- Field pictures: compute from exact formulas or `numerics.py`; prefer exact field lines via
  flux/stream functions (`flux_lines`), or `streamplot` with explicit `start_points`; magnitude as a
  light sequential colour map (`seq_cmap(ELEC)`), direction with normalised arrows. Put a white
  `halo()` behind labels/arrows that sit on top of lines.
- Diagrams: rounded `rbox`/`card`, `arrow`, `label`, `pill`. Consistent arrow style. No 3-D
  matplotlib unless essential. No figure titles except short panel titles. Avoid text collisions;
  keep ≥ 0.1 in of white space from other elements.
- Captions (in the .tex): first sentence tells the reader what to look at; then what it shows;
  mention normalisation/teaching nature briefly at the end.
- Computed figures must import from `numerics.py` (repo root) or compute with NumPy/SciPy only.
- After building, open every PNG you made with the Read tool and fix overlaps, clipping, tiny text,
  wrong arrow directions. Check both languages (English strings are often longer).

## 5. Checking your chapter
`python3 tools/check_chapter.py zh/chNN_slug.tex [more files]` compiles your chapter(s) in a
harness where all registry labels resolve, writes `build/check_<name>/…pdf` and a contact sheet
`build/check_<name>/sheet_*.png`. Look at the sheets (Read tool) and at least two pages at full
resolution (`pdftoppm -r 110 -f N -l N`). Fix: LaTeX errors, overfull boxes, undefined references
(other than registry labels not yet written), figures too large/small, boxes splitting badly.

## 6. Terminology (use these Chinese terms consistently)
电场强度/电场 E；电位移 D；极化强度 P；磁通密度 B；磁场强度 H；磁化强度 M；电标势 φ；磁矢势 A；
磁标势 ψ；（表面电流）流函数 ψ_s；自由电荷/自由电流；束缚电荷/束缚电流（磁化电流）；位移电流；
电导率 σ；介电常数 ε；磁导率 μ；磁阻率 ν；电极化率 χ_e；磁化率 χ_m；复导纳率 Y；
磁通 Φ_B；磁链 λ；电感 L；互感；磁动势 𝓕（`\mathcal F`）；磁阻 ℛ（`\mathcal R`）；磁导 𝒫（`\mathcal P`）；
边缘效应（fringing）；漏磁（leakage）；剩磁 B_r；矫顽力 H_cB / 内禀矫顽力 H_cJ；回复线、回复磁导率 μ_rec；
负载线；磁能积 (BH)_max；退磁场、退磁因子/退磁张量 N_d；割线磁导率、微分磁导率、增量磁阻率张量；
静态、电准静态（EQS）、磁准静态（MQS）、全波；单向耦合；趋肤深度 δ；相量（峰值约定）；
规范、库仑规范；本质边界条件、自然边界条件；弱形式；测试函数；自由度；刚度矩阵；
棱单元（Nédélec）；牛顿法、Picard 迭代、延拓；比吸收率 SAR；外周神经刺激 PNS；激活函数；
基强度、时值；完整电极模型（CEM）；导联场；数值验证（verification）与物理确认（validation）。

## 7. Shaded 3-D figures
Some ideas are much clearer in 3-D. The editor added an off-screen renderer: `figsrc/render3d.py`
(three.js in headless Chromium via Playwright) and examples in `figsrc/fig_3d.py`
(solenoid_3d, coil_block_3d, div_curl_3d, interface_probes_3d, a_circulation_3d, multipillar_3d,
edge_element_3d, gradient_3d, stream_cylinder_3d). Python computes all geometry; labels are added by
matplotlib (`put`, `put_leader`) so fonts match. Do not edit fig_3d.py; if you need a new 3-D figure,
create it in your own module with the same API (Scene → render → trim → compose/compose_row → put).
Use 3-D only where depth/geometry genuinely helps; keep 2-D vector figures for plots and diagrams.

## 8. Notes from finished chapters
- Avoid maths in `\section{}` titles; if unavoidable use `\texorpdfstring{$...$}{text}`.
- Finished Chinese chapters you can read for continuity (and whose registry labels you can cite):
  zh/ch04_magnetic.tex, zh/ch05_matter.tex, zh/ch06_interfaces.tex, zh/ch07_systems.tex
  (more appear as authors finish; check `ls zh/`).
- Ch 7 already defines a teaching gapped-core example (ℓ_y = 0.2 m, g = 1 mm, A = 4 cm², μ_r = 1000,
  N = 200, I = 2 A; see ex:systems-core) — reuse it rather than inventing a different one where it fits.
