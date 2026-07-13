# Manuscript Section Blueprint

This file proposes the full structure of the first MaternaQA-es paper.

## Recommended Paper Structure

```text
Title
Abstract
Keywords
1. Introduction
2. Related Work
3. Dataset Design Goals
4. Source Corpus
5. Corpus Processing and Chunk Construction
6. Synthetic QA Generation Methodology
7. Dataset Variants and Format
8. Quality Assessment
9. Dataset Statistics and Analysis
10. Intended Uses and Release
11. Limitations and Ethical Considerations
12. Conclusion and Future Work
References
Appendix A. Prompting and Generation Details
Appendix B. Dataset Schema
Appendix C. Additional Statistics
Appendix D. Datasheet / Data Statement
```

## Title

Recommended:

> MaternaQA-es: A Synthetic Spanish Clinical Question-Answering Dataset for Pregnancy and Maternal Health from Curated Medical Documents

## Abstract

The abstract should include:

1. Gap: limited Spanish maternal-health QA datasets.
2. Resource: MaternaQA-es.
3. Method: curated PDFs, text extraction, cleaning, chunking, synthetic QA generation.
4. Outputs: QA pairs, closed-book and grounded variants, flat audit format.
5. Quality: document-level splits, source traceability, grounding, RAGAS estimates.
6. Availability and use: research, RAG, fine-tuning, evaluation.
7. Caution: not for direct clinical decision-making.

## 1. Introduction

### Goals

- Establish the need for Spanish maternal-health QA resources.
- Explain why pregnancy and maternal health are high-impact domains.
- Explain why QA datasets are useful for clinical NLP, RAG, and fine-tuning.
- State the dataset contribution.

### Include

- Biomedical QA datasets exist, but many are English-centric or broad-domain.
- Maternal health requires specialized terminology and careful grounding.
- Spanish clinical NLP needs reusable, documented resources.
- MaternaQA-es addresses this by creating a source-traceable QA dataset.

### Possible contribution bullets

> The contributions of this work are:
>
> 1. We introduce MaternaQA-es, a synthetic Spanish clinical QA dataset focused on pregnancy and maternal health.
> 2. We describe a reproducible dataset construction methodology from curated medical documents, including extraction, cleaning, chunking, QA generation, and publication variants.
> 3. We release closed-book, grounded, and flat-audit dataset variants to support instruction tuning, evidence-grounded QA, and dataset analysis.
> 4. We provide quality assessment through document-level split controls, grounding analysis, RAGAS faithfulness/relevancy estimates, and explicit limitations.

## 2. Related Work

### 2.1 Biomedical and clinical QA datasets

Discuss PubMedQA, MedQuAD, MedMCQA, BioASQ, RealMedQA.

Purpose:

- Locate MaternaQA-es in biomedical QA.
- Show the gap: Spanish + pregnancy/maternal-health + synthetic source-grounded QA.

### 2.2 Synthetic instruction and QA data generation

Discuss Self-Instruct, Alpaca, WizardLM/Evol-Instruct, PubMedQA artificial subset, RealMedQA LLM generation.

Purpose:

- Show that synthetic QA/instruction data is a recognized method.
- Emphasize the need for filtering and documentation.

### 2.3 Dataset documentation and responsible release

Discuss Datasheets for Datasets, Data Statements for NLP, Data Cards.

Purpose:

- Justify detailed documentation of motivation, composition, sources, preprocessing, uses, distribution, maintenance, and limitations.

### 2.4 Evidence-grounded QA and downstream adaptation

Discuss RAG, FLAN, LIMA, LoRA/QLoRA briefly.

Purpose:

- Explain why grounded variants and SFT-ready formats matter.
- Keep fine-tuning as downstream, not central.

## 3. Dataset Design Goals

### Design goals

| Goal | Explanation |
|---|---|
| Pregnancy/maternal-health focus | Keep the dataset clinically coherent and domain-specific. |
| Spanish language | Address language-resource scarcity. |
| Source traceability | Preserve link from QA examples to source chunks and PDFs. |
| Evidence grounding | Enable grounded QA and auditing. |
| Training-ready variants | Support downstream SFT experiments. |
| Leakage control | Split by document, not by random QA pair. |
| Transparent limitations | Avoid overclaiming clinical validity. |

## 4. Source Corpus

### Include

- Types of documents: clinical guidelines, protocols, manuals, academic chapters, articles, institutional documents.
- Source location: `pdfs/obstetrics/` currently, but final name may be updated.
- Inclusion/exclusion criteria.
- Language and topic coverage.
- Current corpus statistics.

### Table to include

| Metric | Value |
|---|---:|
| Processed PDFs | 63 |
| Extracted pages | 5,856 |
| Kept clean pages | 5,176 |
| Discarded pages | 660 |

## 5. Corpus Processing and Chunk Construction

### Include

- PDF extraction using PyMuPDF with pdfplumber fallback.
- Page-level cleaning.
- Dropping pages requiring OCR, too short pages, fragmented text, references-heavy pages, TOC/index pages, non-clinical sections.
- Chunk construction with metadata.
- Clinical score and topic inference.
- Deduplication and near-duplicate checks.
- Document-level splits.

### Key argument

> The objective of preprocessing was not aggressive normalization, but the construction of stable, clinically meaningful text units suitable for QA generation while preserving source traceability.

## 6. Synthetic QA Generation Methodology

### Include

- Generation from accepted chunks.
- Structured output constraints.
- Question types: factual, definition, comparison, reasoning, application, hypothetical.
- Requirement that questions be self-contained.
- Requirement that answers be grounded in source context.
- Avoiding phrases such as "according to the text".
- Metadata retained for auditing.

### Important distinction

The method generates **synthetic QA pairs**, not synthetic clinical facts. The factual basis should come from the curated documents.

## 7. Dataset Variants and Format

| Variant | Input | Output | Intended use |
|---|---|---|---|
| `sft_closed_book` | Question | Answer | Supervised fine-tuning without explicit context; tests parametric adaptation. |
| `sft_grounded` | Context + question | Answer | Evidence-conditioned QA and grounded SFT. |
| `qa_flat_jsonl` | Flat records with metadata | Audit/exploration | Dataset analysis, documentation, paper tables. |

### Include schema

Document all fields in an appendix:

- question;
- answer;
- source context;
- source PDF;
- pages;
- split;
- question type;
- topics;
- token estimates;
- evaluation fields if present.

## 8. Quality Assessment

### Include

- RAGAS faithfulness and answer relevancy.
- Grounding overlap analysis.
- Document-level leakage control.
- Deduplication.
- POC generator/evaluator comparison.
- Limitations of automated evaluation.

### Quality table

| Split | Sample evaluated | Faithfulness | Relevancy |
|---|---:|---:|---:|
| Train | 300 / 5,093 | 0.7726 | 0.6466 |
| Validation | 100 / 306 | 0.7826 | 0.6812 |
| Test | 100 / 328 | 0.7132 | 0.5583 |

## 9. Dataset Statistics and Analysis

### Include

- QA count by split.
- Chunks and PDFs by split.
- Question type distribution.
- Topic coverage.
- Token length distributions.
- Source-document distribution.
- Low-grounding examples and error types.

### QA split table

| Split | QA pairs | Source chunks | Source PDFs |
|---|---:|---:|---:|
| Train | 5,093 | 1,744 | 52 |
| Validation | 306 | 101 | 2 |
| Test | 328 | 108 | 3 |
| Total | 5,727 | 1,953 | 57 |

## 10. Intended Uses and Release

### Intended uses

- clinical NLP research;
- Spanish biomedical QA experimentation;
- supervised fine-tuning;
- RAG evaluation;
- prompt and evaluation studies;
- educational/research prototypes.

### Non-intended uses

- direct patient care;
- diagnosis;
- treatment recommendation without clinician oversight;
- replacing clinical guidelines;
- claiming model safety after fine-tuning without independent validation.

## 11. Limitations and Ethical Considerations

### Include

- Synthetic QA is not equivalent to clinician-authored QA.
- Automated metrics are not clinical validation.
- PDF extraction may lose tables, figures, or layout-dependent information.
- OCR-needed pages were not recovered.
- Source documents may contain outdated or local recommendations.
- Coverage may be uneven across maternal-health topics.
- License and redistribution rights must be checked for source PDFs and derived data.

## 12. Conclusion and Future Work

### Include

- Restate MaternaQA-es as a documented resource.
- Emphasize source traceability and variants.
- Future work: clinician validation, more sources, OCR, more languages, benchmark paper, human error analysis.

## Appendix Ideas

| Appendix | Content |
|---|---|
| A | Prompt templates and structured output schema. |
| B | Full JSONL schema. |
| C | Dataset statistics and per-topic distributions. |
| D | Datasheet/Data Statement for MaternaQA-es. |
| E | Examples of accepted QA pairs and rejected/low-quality cases. |
