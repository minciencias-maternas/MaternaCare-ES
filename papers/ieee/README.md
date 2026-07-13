# IEEE Draft for MaternaQA-es

This folder contains the first IEEE-style two-column manuscript draft for the MaternaQA-es dataset paper.

## Files

- `main.tex` — IEEE conference-style LaTeX manuscript.
- `references.bib` — BibTeX references used by the draft.

## Intended framing

The paper is a **dataset creation paper / data descriptor**, not a software-pipeline paper. It focuses on the dataset, its construction methodology, source traceability, quality assessment, intended uses, and limitations.

## Compile

If `IEEEtran.cls` is installed:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If `IEEEtran.cls` is missing, install the IEEE LaTeX template / TeX Live publishers package or compile in Overleaf using the IEEE Conference Template.

## Before submission

- Replace placeholder author block.
- Verify final dataset URLs and licenses.
- Decide whether to keep or move the AI assistance disclosure depending on venue policy.
- Re-run dataset statistics if the source dataset changes.
- Add clinician validation results if available.
