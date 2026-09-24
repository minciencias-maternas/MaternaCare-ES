# Paper Writing Rules

The current manuscript and bibliography are `papers/oficial/MaternaCare_ES_with_QLoRA_table.tex` and `papers/oficial/references.bib`.

## Non-negotiable rules

- Never invent citations.
- Every citation must be traceable to a real DOI, arXiv page, ACL Anthology page, IEEE/ACM/Springer/Elsevier page, or official documentation.
- Never invent experimental results.
- Use placeholders when evidence is missing.
- Separate claims from evidence.
- Preserve LaTeX compilation.
- Do not rewrite the whole paper unless explicitly requested.
- Prefer small, reviewable edits.

## Bibliography workflow

When adding references:
1. Verify the source using a reliable academic index or publisher page; prefer DOI-based metadata from Crossref, Semantic Scholar, OpenAlex, or arXiv.
2. Update `papers/oficial/references.bib`; do not invent missing fields.
3. Use BibTeX keys in the format `lastnameYearKeyword`.
4. Every citation used in `.tex` must exist in the bibliography, and every bibliography entry must be traceable to a reliable academic source.
