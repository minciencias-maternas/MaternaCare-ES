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

## Progress
Authorized by the user after clarification. Writer updated author block, replaced the old QLoRA multi-panel figure with `ft-vs-lora.pdf`, removed the MaternaQA construction and runtime figures, retained dataset references, and deferred RAGAS. A follow-up removed a duplicated 2,268-chunk statement and screenshot-adjacent shared-memory capacity details while retaining the RTX 5070 Ti Laptop GPU and 12 GB dedicated VRAM. Independent rebuild and inspection confirmed 8 letter-size pages, no final-pass undefined citations/references or package/BibTeX warnings, 4 underfull hboxes and 2 underfull vboxes, and a legible uncropped replacement figure. The official ISCMI guidance includes up to 5 pages per regular registration, so 3 additional pages are chargeable. No commit was made; it remains pending explicit user authorization. Writer build produced 8 letter-size pages with no final citation/reference warnings or errors, 6 underfull hboxes, 1 underfull vbox, and the standard IEEEtran last-page balancing reminder. Independent inspection confirmed the RAG architecture figure is gone, methods prose remains, and RAGAS is legible and unclipped. The intro/citation is on page 4 and IEEEtran floats the tall single-column figure to the top of page 5; this is normal float placement. ISCMI still exceeds the five included pages by 3 chargeable pages. No commit was made.

Route: delegated direct writer for source plus generated PDF; one writer, no parallel writes. Exact build command used:
`mkdir -p /tmp/maternacare-ragas && latexmk -pdf -interaction=nonstopmode -halt-on-error -cd -outdir=/tmp/maternacare-ragas papers/oficial/MaternaCare_ES_with_QLoRA_table.tex && cp /tmp/maternacare-ragas/MaternaCare_ES_with_QLoRA_table.pdf papers/oficial/MaternaCare_ES_with_QLoRA_table.pdf`
