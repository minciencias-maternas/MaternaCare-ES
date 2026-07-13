# MaternaQA-es Paper Planning Notes

This directory documents the strategy for the **first paper** associated with the project now named **MaternaQA-es**.

The intended manuscript is **not** a software-pipeline paper and should not be framed as "we built scripts to generate data." It should be framed as a **dataset creation paper / data descriptor / resource paper**:

> We created, documented, and assessed a synthetic Spanish clinical question-answering dataset focused on pregnancy, maternal health, and maternity care, derived from curated medical documents.

## Working Dataset Name

**MaternaQA-es**

Rationale:

- `Materna` signals maternal health, pregnancy, maternity care, and perinatal context.
- `QA` signals question-answering.
- `es` signals Spanish-language content.
- The name is more accurate than a generic obstetrics/gynecology label because the dataset emphasis should be pregnancy and maternal-health knowledge rather than the full gynecology domain.

## Recommended Working Title

**MaternaQA-es: A Synthetic Spanish Clinical Question-Answering Dataset for Pregnancy and Maternal Health from Curated Medical Documents**

Alternative titles are documented in [`07-title-abstract-and-contribution-drafts.md`](./07-title-abstract-and-contribution-drafts.md).

## Scope of the First Paper

The first paper should explain:

1. **What was created**: a Spanish synthetic clinical QA dataset focused on pregnancy and maternal health.
2. **Why it was needed**: limited availability of specialized Spanish QA resources for maternal-health NLP.
3. **How it was created**: source selection, document processing, chunking, synthetic QA generation, dataset variants, and quality assessment.
4. **How quality was assessed**: source traceability, document-level splits, RAGAS faithfulness/relevancy estimates, grounding heuristics, and manual/clinical-review recommendations.
5. **How the dataset should and should not be used**: research, fine-tuning, RAG, evaluation, education; not direct clinical decision-making.

## Relationship to the Second Paper

The project has two research artifacts:

| Artifact | Main question | Output | Paper type |
|---|---|---|---|
| **Artifact 1: MaternaQA-es dataset** | How can we construct and document a Spanish maternal-health QA dataset from curated medical documents? | Dataset, documentation, quality analysis, Hugging Face release | Dataset paper / data descriptor / resource paper |
| **Artifact 2: fine-tuning benchmark** | How useful is MaternaQA-es for adapting and evaluating open LLMs? | Fine-tuned models, QLoRA experiments, model comparison, error analysis | Experimental fine-tuning / benchmark paper |

The first paper may mention fine-tuning as an **intended downstream use**, but it should not make model adaptation the central contribution.

## Files in This Directory

| File | Purpose |
|---|---|
| [`01-paper-positioning-and-scope.md`](./01-paper-positioning-and-scope.md) | Defines the manuscript identity, novelty, audience, and boundaries. |
| [`02-argumentation-map.md`](./02-argumentation-map.md) | Maps the claims we want to defend to evidence, project artifacts, and papers. |
| [`03-related-work-and-citation-strategy.md`](./03-related-work-and-citation-strategy.md) | Lists the papers to cite and where to use them. |
| [`04-manuscript-section-blueprint.md`](./04-manuscript-section-blueprint.md) | Provides a detailed section-by-section outline for the paper. |
| [`05-methodology-and-dataset-documentation.md`](./05-methodology-and-dataset-documentation.md) | Documents the dataset construction methodology in paper-ready terms. |
| [`06-quality-validation-ethics-and-limitations.md`](./06-quality-validation-ethics-and-limitations.md) | Covers validation, risk, ethics, limitations, and intended/non-intended uses. |
| [`07-title-abstract-and-contribution-drafts.md`](./07-title-abstract-and-contribution-drafts.md) | Draft titles, abstract skeletons, contribution statements, and wording blocks. |

## Current Project Statistics to Re-check Before Submission

These values are taken from the current repository documentation and should be verified against final artifacts before submission:

| Item | Current value |
|---|---:|
| Processed PDFs | 63 |
| Extracted pages | 5,856 |
| Kept clean pages | 5,176 |
| Final chunks after deduplication | 2,268 |
| LM dataset chunks in publication splits | 2,223 |
| Synthetic QA pairs | 5,727 |
| Train QA pairs | 5,093 |
| Validation QA pairs | 306 |
| Test QA pairs | 328 |
| Average grounding overlap | 0.6836 |
| Low-grounding pairs | 27 / ~5,000+ reported as 0.54% |
| RAGAS train sample | faithfulness 0.7726; relevancy 0.6466 |
| RAGAS validation sample | faithfulness 0.7826; relevancy 0.6812 |
| RAGAS test sample | faithfulness 0.7132; relevancy 0.5583 |

## Core Positioning Sentence

> MaternaQA-es is a Spanish synthetic clinical question-answering dataset for pregnancy and maternal-health research, constructed from curated medical documents with explicit source traceability, document-level data splits, and quality assessment procedures designed to support downstream research in retrieval-augmented QA, instruction tuning, and clinical NLP evaluation.
