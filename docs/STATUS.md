# Project status

Updated 2026-10-11.

## Chinese edition — complete
- Front matter, Chapters 1–24, Appendices A (vector identities), B (symbols and units) and C (index of comparisons); 383 pages.
- `python3 build.py --language zh` builds it with no LaTeX errors, overfull boxes, undefined references or missing glyphs.
- Every chapter has `checks/check_chNN.py`; all of them, plus `numerics.py`, `concept_checks.py` and `extended_checks.py`, pass.
- Each chapter was written from the plan in `PLAN.md` and then reviewed by an independent reader (physics, teaching order, cross-chapter consistency, language); the reviewers' fixes are in the text and their asserts in the check scripts.
- 96 figures (87 two-dimensional, 9 shaded 3-D), each in Chinese (`figures_zh/`) and English (`figures/`).

## English edition — not yet translated (deferred at the author's request)
- Ready: all English figures, `main_en.tex` (title page, part pages and part introductions), `references_en.tex`, the `\ZE{}{}` English headings of every teaching box.
- To do: the chapter text in `en/` (same file names as `zh/`, identical labels, citations, figure calls, box order and displayed mathematics).
- How: use the brief in `agent_prompts/translate_template.txt`, a few chapters per translator, then run `python3 validate_source.py` (it lists files still waiting for translation and checks the structure of every translated pair) and `python3 build.py`.

## Possible follow-ups noted by the reviewers (not errors)
- Ch 14's gapped-core example uses different parameters from Ch 7's `ex:systems-core`; Ch 14 says so explicitly. Aligning them would change numbers in Ch 14 and Ch 18 and one figure.
- Ch 6 mentions total normal current continuity before phasors are introduced in Ch 8 (now written in the time domain).
- Mansfield & Chapman (1986, active screening) and Turner (1986, target-field method) would be natural primary references for Ch 20; they are not cited because their bibliographic details could not be verified from the publishers' pages during writing.
