# ODD task: IEEE / ISCMI 2026 paper revision

## Goal
Revise `papers/oficial/MaternaCare_ES_with_QLoRA_table.tex` and rebuild the PDF for ISCMI 2026, preserving IEEEtran conference format, research content, and unrelated user changes.

## Previous IEEEtran pass
The earlier compliance pass and paragraph/float investigation are recorded in Engram observations #2645 and #2647. Baseline was a 9-page IEEEtran conference paper; build had no errors, package warnings, citation/reference errors, or overfull boxes. Do not chase every natural two-column paragraph split: forcing them creates whitespace. Preserve the existing `\needspace` fix for the harmful Table II reference/float collision.

## Accepted scope
- Replace the existing three-panel QLoRA figure with `papers/oficial/figures/ft-vs-lora.pdf` as a single-column figure, and align its caption and nearby prose with its actual full-fine-tuning-vs-LoRA content.
- Remove the existing MaternaQA corpus-construction figure and redundant dataset-creation explanation. Retain the MaternaQA-es citation and benchmark-relevant provenance, split, selected test-subset, and retrieval-corpus facts.
- Remove the `img-006.png` runtime screenshot from the paper and remove screenshot-specific utilization observations and cross-reference. Retain concise hardware facts relevant to reproducibility if independent of the screenshot.
- Set author order/details: Nicolás Hoyos-Giraldo (ORCID `0009-0001-9462-1136`), then Jhon Hander Mejía-Muñoz (ORCID `0009-0008-7768-4072`), then two explicit placeholders for the other authors. Shared affiliation: Facultad de Ingeniería, Institución Universitaria de Envigado, Envigado, Colombia. Keep name/ORCID association clear in IEEE conference author style.
- In the follow-up revision, remove the existing RAG-architecture variants figure (retain its methodology prose) and place `papers/oficial/figures/ragas-eval-workflow.pdf` in the Evaluation Framework section. Add a concise in-text bridge that describes the workflow and clarifies that the diagram shows response-quality metric flow, while Context Precision/Recall remain retrieval-specific metrics. Do not imply the diagram covers more than it depicts.
- Remove from the paper only; do not delete unselected image assets from the codebase.

## Constraints and acceptance criteria
- Preserve experimental values, claims, interpretation, IEEEtran conference class, and MaternaQA-es dataset citation.
- Figure/caption readable and not clipped in final two-column PDF. No missing graphics, undefined citations/references, overfull boxes, LaTeX errors, or BibTeX warnings.
- Rebuild and record page count and warnings. ISCMI's official guidance: one regular registration includes up to 5 pages; extra pages are chargeable. Do not alter IEEE margins/type/spacing or crop content to hit 5 pages.
- TDD is not applicable to manuscript/layout changes. Functional check: clean LaTeX/BibTeX build plus rendered-PDF inspection.
- Branch `feat/rag-benchmark-wip`; no commit without explicit user authorization. Preserve all unrelated worktree changes.

## Tasks
- [x] `ISCMI-1` Update author block and refactor figure/text scope in the LaTeX source.
- [x] `ISCMI-2` Independently rebuild and inspect the PDF, references, warnings, page count, and visual layout.
- [x] `ISCMI-3` Report results, including page-fee implications and deferred RAGAS decision.
- [ ] `ISCMI-4` Create a work-unit commit only if the user explicitly authorizes it.
- [x] `ISCMI-5` Replace the RAG architecture figure with the RAGAS evaluation workflow and clarify its scope in surrounding prose.
- [x] `ISCMI-6` Rebuild and inspect the updated PDF, warnings, page count, and figure placement.
- [x] `ISCMI-7` Report the updated result and ISCMI page implications.

## Correction: metric-provenance claim in the RAGAS paragraph (post-verification)
The first RAGAS-figure revision inserted a claim that Context Precision and Context Recall are "separately evaluated retrieval metrics". That was factually wrong and was corrected after reading the code.

Verified evidence:
- `scripts/rag_benchmark/metrics.py` `build_metrics()` constructs **all six** metrics from `ragas.metrics.collections`: `ContextPrecision(llm)`, `ContextRecall(llm)`, `Faithfulness(llm)`, `AnswerRelevancy(llm, embeddings)`, `AnswerCorrectness(llm, embeddings)`, `SemanticSimilarity(embeddings)`.
- `scripts/evaluate_model_predictions.py` imports the same RAGAS collection classes and builds `semantic_similarity` as `stack["SemanticSimilarity"]`.
- The only `sentence_transformers` usage in `scripts/` is `scripts/rag_benchmark/retrieval.py` (BGE-M3 dense-retrieval embeddings), not metric computation.

So every retained metric is computed by the single RAGAS evaluator; `semantic_similarity` is a RAGAS metric that uses embeddings instead of the LLM judge. The manuscript sentence was reworded three times: (1) remove the false "separately evaluated" claim, (2) remove "omitted" (read as if those metrics were unused), and (3) simplify to one short sentence after the user rejected the redundant second `es2024ragas` citation. Final wording: "Figure~\ref{fig:ragas-eval-workflow} illustrates the RAGAS evaluation workflow, showing how each part contributes to scoring the responses." Rebuild: 8 pages, no final LaTeX/reference/citation/package/BibTeX warnings, 6 underfull warnings (5 hbox, 1 vbox). No commit.

Writing lessons: (a) when a figure covers only part of a set, do not say the rest are "omitted" — a reader infers those metrics are unused; (b) do not re-cite a source already cited earlier in the same paragraph; (c) the user prefers a short sentence over a fuller explanation.

## RAG variants figure (user-requested addition)
The user supplied `papers/oficial/figures/rag-types.pdf` (landscape, 1180 x 673 pt, ~1.75:1) to illustrate the three retrieval techniques. It is a redesigned version of the removed `img-003` content: (A) no retrieval, (B) Hybrid (Corpus -> Dense + Sparse -> RRF fusion -> Model), (C) HyDE (Generator -> Hypothetical doc. -> Dense retrieval). It restores the RAG methodology illustration without duplicating existing figures.

Decision: user explicitly chose SINGLE-COLUMN placement despite the legibility warning. Implemented as `figure` at `\columnwidth`, label `fig:rag-types`, in `Retrieval-Augmented Generation Protocol`, with the one-line in-text reference "Fig.~\ref{fig:rag-types} summarizes the three variants."

Result: still 8 pages (no extra chargeable page). Final pass: no LaTeX/citation/reference/package/BibTeX warnings; 7 underfull warnings (5 hbox, 2 vbox). Figure numbering updated automatically to Fig. 1 ft-vs-lora, Fig. 2 rag-types, Fig. 3 RAGAS; no stale cross-references.

Legibility finding (reported to the user): at `\columnwidth` the main labels (NO RETRIEVAL/HYBRID/HYDE, Corpus, Dense, Sparse, RRF fusion, Model, Answer, Generator, Hypothetical doc., Dense retrieval) remain readable, but the small sublabels ("clinical query (ES)", "Model Base Knowledge", "model + context", "grounded + cited", "LLM / SLM") fall below IEEE's ~8 pt figure-label guidance. Switching to a full-width `figure*` at `\textwidth` would render them at ~12 pt and likely still fit 8 pages.

## Text reduction pass (user-approved: groups 1-3)
The user reviewed the over-explanation analysis and approved applying all three groups: (1) re-explanations of cited standard techniques, (2) repeated caveats/hedges, (3) the internal process note.

Criterion used: re-explaining how a cited standard technique works is a candidate for compression; the authors' own formulas, parameters, configurations, and data stay.

Sites to change (line numbers from the pre-edit source):
1. L523-535 `RAG Benchmark Evaluator`: drop the six per-metric definitions (cited RAGAS + now Fig. 3) and point to the figure.
2. L320-326 `Grounded QLoRA`: the concept is stated three times and shown by Fig. 1; keep one statement plus figure/table references.
3. L465-482 `HyDE`: the "retrieved chunks, not the hypothetical text" statement appears twice; merge the two paragraphs and keep it once. Convert the internal process note about the request-side output limit into an explicit limitation sentence.
4. L404-405 RAG protocol intro: remove the third restatement of the same claim.
5. L440-451 `Dense and Sparse Retrieval`: drop the BM25 manual definition.
6. L453-463 `Canonical Hybrid Retrieval`: stop explaining RRF mechanics; keep the `rank_constant` and the `k_c` equation.
7. L776-777 Discussion: drop the repeated "evidence-available / completely unseen" caveat (stated in Methods).
8. L801-802 Conclusions: drop the repeated "evidence-available" and scenario-confound restatements; keep the actionable conclusion.
9. L573-577 Reproducibility: compress the scenario-confound re-explanation to the design commitment (within-scenario comparisons).
10. L781-783 Discussion: delete the standalone paragraph that repeats the Methods scenario description.
11. L544-559 `Analysis Plan`: remove two hedge restatements repeated from the Abstract/Results.
12. L597-598 Results: drop the repeated descriptiveness reminder.

Untouched by design: `clinical_score` equations and symbols, `k_c` and `rank_constant` values, Table II, Inference and Prompt Controls, the Reproducibility list, corpus/dedup facts, Discussion interpretations, and the Abstract's single hedge (the right place for it).

Tasks: `ISCMI-11` apply the 12 sites, `ISCMI-12` rebuild and inspect, `ISCMI-13` report.

### Result of the reduction pass
All 12 edits applied; no anchor failed. Source went from 827 to 788 lines (-39). Verified counts after the pass: "completely unseen" 3 -> 1, "question composition" 3 -> 0, "confounded" 1 (kept in Methods, the right place), "descriptive" 6 -> 4, "exploratory" 3 (Abstract, Analysis Plan, Conclusions), "hypothetical text" 3 -> 1, "BM25 supplies" 1 -> 0.

Rendered checks confirmed: the RAG-evaluator subsection now points to Fig. 3 instead of defining six metrics; the HyDE subsection states the evidence rule once and the internal process note became an explicit limitation ("the request-side output limit could not be recovered from the retained run records"); the Results reminder, the Discussion's repeated scenario paragraph, and the Conclusions' repeated caveats are gone while the actionable close remains.

Build: 8 pages, UNCHANGED. The reduction tightened prose but did not save a page, because ~39 source lines is less than one two-column page (~104 lines). Still 3 pages over ISCMI's included 5. Saving a page would require a structural cut (e.g., moving Table III or Table II detail to a repository/appendix), not further prose trimming.

Warnings: no undefined references/citations; 6 underfull hbox and 2 underfull vbox.

## Table style unification (user-requested)
The user compared the three tables and asked for Tables I and II to match Table III's style.

Diagnosis: Table III uses `\scriptsize` + `tabular*` with `\extracolsep{\fill}` (evenly spread columns, tight rows). Table II was the real outlier at `\small` (larger text, taller rows, larger note). Table I shared Table III's font size but used `tabularx` (left-packed, prose column absorbing the width).

The IEEE Word template's own sample table sets no cell justification or centering, so it does not prescribe alignment; the criterion here is internal visual consistency.

Changes applied:
- Table II: `\small` -> `\scriptsize` (now matches Table III's text and note size).
- Table I: `tabularx` `@{}c l l l X@{}` -> `tabular*` `@{\extracolsep{\fill}}c l l l p{0.40\textwidth}@{}`, so the first four columns spread evenly like Table III while the prose column keeps a fixed width.
- Removed the now-unused `\usepackage{tabularx}` line (dead after Table I stopped using it).

Result: all three tables now share `\scriptsize` + `tabular*` + `\extracolsep{\fill}`. Verified visually on pages 3, 4 and 6. Final build: 8 pages (unchanged), no Overfull hbox, no undefined references or citations.

## Font audit (user-reported "different font")
The user reported that the `MaternaQA-es Subset` text looked like a different font than `Analysis Plan`, and asked to "fix Analysis Plan". Both sections are on page 5 of the current PDF, so the comparison was measurable directly.

Measured result — the body text is uniform:
- Both sections, and in fact all body prose in the PDF, use `NimbusRomNo9L` (Times) at 10 pt with 11.8-12.1 pt leading. Identical leading is the decisive test: leading scales with font size.
- Per-span size reported by PyMuPDF wobbles between 9.86 and 10.06 **line by line inside the same paragraph**, equally in both sections. Cause: `microtype` automatic font expansion is active (log: "Automatic font expansion enabled (level 2), stretch: 20, shrink: 20, step: 1"), which scales each line horizontally by up to +/-2% to improve justification. This is deliberate and invisible. Do NOT disable it: it would not unify anything and would worsen justification.
- `NimbusMonL` (monospace) in body text is legitimate: it is the `\code{}` technical-identifier command.

Real extra typefaces — all inside the three figures, never in the prose:
- `Poppins-Bold`, `Poppins-SemiBold`, `Poppins-ExtraBold` (geometric sans, from the HTML/Skia figure exports).
- Type 3 fallbacks `NotoSans_700wght`, `NotoSansJP_700wght`, `STIXTwoText-Italic`, substituted because Poppins lacks glyphs such as subscript zero, multiplication sign, arrow, mu, and prime.
- All Type 3 usage is figure-internal: page 3 (ft-vs-lora: `W0`, `d x k`), page 5 (ragas-eval-workflow: arrows, `mu`).
- So the figures carry four families against one in the text. That contrast is the likely source of the perceived inconsistency, especially on page 5 where the RAGAS figure sits beside Times prose.

No change was made for this report. The user chose to point at a concrete location for a targeted measurement instead of a global change.

Diagnostic method worth reusing: PyMuPDF `get_text('dict')` per span for family/size, `pdffonts` for the embedded font inventory, and line `bbox` y-deltas to derive leading.

## Font audit follow-up: measured the exact point the user flagged
The user pointed at page 5, `I. Analysis Plan` vs `B. MaternaQA-es Subset`, and described the difference as a perceptible weight/size difference "not tiny but noticeable while reading". Measured that exact spot at 400 dpi.

What is IDENTICAL (so it is not the cause):
- Family: `NimbusRomNo9L` (Times) in both, single embedded font object.
- Vertical size: line band height 50 px (= 9.00 pt of ink at 400 dpi) in both, and lowercase x-height median 25-26 px in both. A real size change would move these, and would also move leading; leading is ~12 pt in both.

What IS measurably different (the actual cause of the perception):
1. **Justification looseness.** Interword gaps (per justified line, gaps >1.5 pt): Analysis Plan mean 3.86 pt, median 3.78, p10 3.06, p90 4.48. MaternaQA-es mean 4.15 pt, median 3.96, p10 2.45, p90 5.94. So the MaternaQA paragraph averages ~7.5% looser and its loose lines are up to ~33% looser. Normal 10 pt Times interword space is 2.5 pt, so both are stretched, but the MaternaQA one much more unevenly.
2. **Lining-numeral density.** That paragraph is full of cap-height figures -- (0.668), (0.813), (0.811), 0.129, 0.047, 0.053, 0.057, 0.212, 0.109 -- plus capitalized metric names and one monospace `no_rag`. Rigid, unhyphenatable tokens reduce break opportunities, which is exactly what forces looser word spacing, and tall numerals also read visually heavier than lowercase prose.

Conclusion: there is no font, size or weight defect to fix. The lever is justification, not typography. Disabling microtype would make it WORSE, because expansion is what relieves word spacing. Raising expansion limits (e.g. stretch/shrink 20 -> 40) would let the engine trade slightly more glyph width for tighter, more uniform word spacing; the effect is measurable with the same gap statistics before/after. Not yet applied -- global typography change, needs user approval.

## Progress
Authorized by the user after clarification. Writer updated author block, replaced the old QLoRA multi-panel figure with `ft-vs-lora.pdf`, removed the MaternaQA construction and runtime figures, retained dataset references, and deferred RAGAS. A follow-up removed a duplicated 2,268-chunk statement and screenshot-adjacent shared-memory capacity details while retaining the RTX 5070 Ti Laptop GPU and 12 GB dedicated VRAM. Independent rebuild and inspection confirmed 8 letter-size pages, no final-pass undefined citations/references or package/BibTeX warnings, 4 underfull hboxes and 2 underfull vboxes, and a legible uncropped replacement figure. The official ISCMI guidance includes up to 5 pages per regular registration, so 3 additional pages are chargeable. No commit was made; it remains pending explicit user authorization. Writer build produced 8 letter-size pages with no final citation/reference warnings or errors, 6 underfull hboxes, 1 underfull vbox, and the standard IEEEtran last-page balancing reminder. Independent inspection confirmed the RAG architecture figure is gone, methods prose remains, and RAGAS is legible and unclipped. The intro/citation is on page 4 and IEEEtran floats the tall single-column figure to the top of page 5; this is normal float placement. ISCMI still exceeds the five included pages by 3 chargeable pages. No commit was made.

Route: delegated direct writer for source plus generated PDF; one writer, no parallel writes. Exact build command used:
`mkdir -p /tmp/maternacare-ragas && latexmk -pdf -interaction=nonstopmode -halt-on-error -cd -outdir=/tmp/maternacare-ragas papers/oficial/MaternaCare_ES_with_QLoRA_table.tex && cp /tmp/maternacare-ragas/MaternaCare_ES_with_QLoRA_table.pdf papers/oficial/MaternaCare_ES_with_QLoRA_table.pdf`
