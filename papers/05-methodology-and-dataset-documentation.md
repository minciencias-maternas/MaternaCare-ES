# Methodology and Dataset Documentation

This file rewrites the project methodology in paper-ready terms for the MaternaQA-es dataset paper.

## 1. Methodological Philosophy

The methodology should be presented as a **dataset construction process**, not merely a software pipeline.

The key methodological principles are:

1. **Curated source grounding**: generated answers should be traceable to curated maternal-health documents.
2. **Conservative preprocessing**: preserve clinically relevant language while removing extraction noise.
3. **Document-level split integrity**: avoid leakage by ensuring that all chunks from a document remain in the same split.
4. **Multiple publication views**: release variants for different research uses.
5. **Transparent quality assessment**: report automated metrics while acknowledging their limits.

## 2. Source Selection

### Source types

The corpus may include:

- clinical practice guidelines;
- maternal-health protocols;
- pregnancy-care manuals;
- prenatal-care documents;
- labor and delivery guidance;
- postpartum-care documents;
- fetal monitoring documents;
- academic chapters;
- clinical review articles;
- institutional documents.

### Inclusion criteria

Suggested criteria for the paper:

| Criterion | Rationale |
|---|---|
| Spanish-language or Spanish-relevant clinical content | Aligns with MaternaQA-es language scope. |
| Pregnancy, maternal health, perinatal, or obstetric relevance | Maintains domain coherence. |
| Medical or institutional origin | Reduces uncontrolled web noise. |
| Extractable text | Enables reliable QA generation and traceability. |
| Sufficient clinical density | Ensures chunks contain answerable content. |

### Exclusion criteria

| Exclusion reason | Rationale |
|---|---|
| Needs OCR and no OCR applied | Avoid unreliable empty/noisy extraction. |
| Too short | Insufficient context for QA generation. |
| Reference-heavy | Bibliographic sections are poor QA sources. |
| Fragmented extraction | Risk of incoherent generated answers. |
| Non-clinical section | Not aligned with dataset purpose. |
| Table-of-contents/index | Low semantic value for QA. |

## 3. Text Extraction

Paper-ready description:

> Text was extracted at the page level from the source PDF collection. The extraction stage prioritized retaining page-level provenance, including source document identifiers and page ranges, so that downstream chunks and QA pairs could be traced back to their origin. A primary PDF text extraction backend was used, with a fallback extractor for pages with low initial text yield. Pages with insufficient recoverable text were flagged rather than automatically corrected, preserving transparency about OCR limitations.

## 4. Cleaning and Filtering

Paper-ready description:

> Cleaning was designed to remove document artifacts without rewriting clinical content. Repeated headers and footers, page numbers, administrative labels, reference-heavy sections, short fragments, and non-clinical pages were removed or flagged. Each page retained audit metadata describing whether it was kept, the reason for exclusion when applicable, and extraction diagnostics.

## 5. Chunk Construction

### Purpose

Chunks are the bridge between documents and QA generation.

They need to be:

- long enough to contain clinically meaningful context;
- short enough to fit model prompts and QA generation constraints;
- traceable to document and page metadata;
- suitable for document-level split assignment.

### Metadata to document

| Field | Purpose |
|---|---|
| `chunk_id` | Stable identifier. |
| `source_pdf` | Provenance. |
| `pages` | Page-level traceability. |
| `text` | Cleaned clinical content. |
| `section_type` | Approximate document section. |
| `content_role` | Role such as definition, recommendation, explanation, etc. |
| `clinical_score` | Heuristic signal of clinical density. |
| `topics` | Maternal-health topic tags. |
| `token_estimate` | Size control and analysis. |

## 6. Split Strategy

### Why document-level split?

Randomly splitting QA pairs can leak similar or identical content across train/validation/test when multiple examples come from the same document or chunk. MaternaQA-es should emphasize that splits are assigned at the document level.

Paper-ready wording:

> To reduce leakage, dataset splits were assigned at the document level rather than at the QA-pair level. All chunks and QA pairs derived from a given PDF were assigned to the same split. This design prevents near-identical content from the same source document from appearing simultaneously in training and evaluation subsets.

## 7. Synthetic QA Generation

### Generation unit

Input: accepted source chunk.

Output: one or more QA pairs with metadata.

### Question types

| Type | Purpose |
|---|---|
| Factual | Recall of clinically relevant facts. |
| Definition | Explanation of terms or conditions. |
| Comparison | Distinguish concepts, interventions, risks, or states. |
| Reasoning | Explain why a recommendation or phenomenon holds. |
| Application | Apply source knowledge to a scenario. |
| Hypothetical | Scenario-based reasoning with constraints. |

### Prompting constraints to describe

The prompt should require:

- Spanish output;
- self-contained questions;
- clinically precise answers;
- no unsupported claims;
- direct grounding in the source chunk;
- no references such as "according to the text";
- rejection/avoidance of chunks with insufficient clinical substance.

### Important paper argument

> The dataset uses LLMs to generate QA formulations, but the intended factual basis is the curated source document. The synthetic step transforms source-grounded clinical text into instruction-like QA examples; it does not invent clinical knowledge.

## 8. Dataset Variants

### `sft_closed_book`

Purpose:

- train or evaluate a model using question -> answer pairs;
- test whether the model can internalize domain-specific response patterns;
- useful for Paper 2 fine-tuning experiments.

### `sft_grounded`

Purpose:

- train or evaluate context + question -> answer behavior;
- simulate evidence-grounded clinical QA;
- align with RAG-style systems and source-conditioned answering.

### `qa_flat_jsonl`

Purpose:

- transparent inspection;
- dataset analysis;
- paper tables;
- audit and reproducibility.

## 9. Dataset Documentation Checklist

Inspired by Datasheets for Datasets, Data Statements for NLP, and Data Cards, the paper or appendix should answer:

| Documentation item | Answer to include |
|---|---|
| Motivation | Why MaternaQA-es was created. |
| Composition | Number of PDFs, pages, chunks, QA pairs, topics, splits. |
| Collection process | How PDFs were selected and processed. |
| Preprocessing | Extraction, cleaning, filtering, chunking, deduplication. |
| Labeling/generation | How QA pairs were generated and constrained. |
| Quality | RAGAS, grounding, split leakage, manual review plans. |
| Intended uses | Research, SFT, RAG, evaluation. |
| Non-intended uses | Direct clinical care, diagnosis, treatment decisions. |
| Distribution | Hugging Face dataset, GitHub methodology/code. |
| Maintenance | Versioning, future updates, issue reporting. |
| Ethics | Synthetic data risks, clinical safety, licensing, source provenance. |

## 10. Reproducibility Artifacts

The paper should point to:

- GitHub repository for code and methodology;
- Hugging Face dataset card for data release;
- schema documentation;
- train/validation/test files;
- evaluation reports;
- scripts or notebooks to compute statistics;
- prompt templates if publishable.
