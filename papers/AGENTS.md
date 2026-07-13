# Paper Writing Rules

This project is an academic paper project.

## Non-negotiable rules

- Never invent citations.
- Every citation must be traceable to a real DOI, arXiv page, ACL Anthology page, IEEE/ACM/Springer/Elsevier page, or official documentation.
- Never invent experimental results.
- Use placeholders when evidence is missing.
- Separate claims from evidence.
- Preserve LaTeX compilation.
- Do not rewrite the whole paper unless explicitly requested.
- Prefer small, reviewable edits.
- Maintain a changelog of major paper revisions.

## Bibliography workflow

When adding references:
1. Use paper-search-mcp to search academic sources.
2. Prefer DOI-based metadata from Crossref, Semantic Scholar, OpenAlex, or arXiv.
3. Update `references.bib`.
4. Do not invent missing fields.
5. Use BibTeX keys in the format `lastnameYearKeyword`.
6. Every citation used in `.tex` must exist in `references.bib`.
7. Every entry in `references.bib` must correspond to a paper found through a reliable academic source.
