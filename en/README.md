# English chapter text

This folder will hold the English edition: one file per Chinese file in `../zh/`, with the same file name and identical labels, citations, figure calls, teaching-box order and displayed mathematics. The translation has not been started yet; the English figures (`../figures/`), title page, part pages and bibliography are ready.

Brief for translators: `../docs/agent_prompts/translate_template.txt`. Check each translated file with `python3 validate_source.py <stem>` and `python3 tools/check_chapter.py en/<file>.tex --lang en`.
