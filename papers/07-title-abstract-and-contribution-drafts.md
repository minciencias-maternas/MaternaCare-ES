# Title, Abstract, and Contribution Drafts

This file provides paper-ready wording blocks in English.

## 1. Preferred Title

**MaternaQA-es: A Synthetic Spanish Clinical Question-Answering Dataset for Pregnancy and Maternal Health from Curated Medical Documents**

## 2. Alternative Titles

1. **MaternaQA-es: A Source-Traceable Spanish Question-Answering Dataset for Pregnancy and Maternal Health**
2. **MaternaQA-es: Constructing a Synthetic Clinical QA Dataset for Spanish Maternal-Health NLP**
3. **MaternaQA-es: A Curated Synthetic QA Resource for Spanish Pregnancy and Maternity Care Research**
4. **A Synthetic Spanish Clinical QA Dataset for Pregnancy and Maternal Health from Curated Medical Documents**
5. **Building MaternaQA-es: A Documented Synthetic Question-Answering Resource for Spanish Maternal-Health NLP**

## 3. Short Abstract Draft

> Spanish clinical NLP lacks open, well-documented question-answering resources focused on pregnancy and maternal health. We present MaternaQA-es, a synthetic Spanish clinical QA dataset constructed from curated medical documents covering pregnancy, prenatal care, labor, delivery, postpartum care, fetal monitoring, and related maternal-health topics. The dataset was created through a source-traceable process involving PDF text extraction, clinical-content filtering, chunk construction, synthetic QA generation, document-level data splitting, and post-hoc quality assessment. MaternaQA-es includes closed-book, grounded, and flat-audit variants to support supervised fine-tuning, evidence-conditioned QA, dataset analysis, and retrieval-augmented generation research. We report dataset statistics, source coverage, split composition, grounding analysis, and RAGAS-based estimates of faithfulness and answer relevancy. The dataset is intended as a research resource and should not be used for direct clinical decision-making without independent expert validation.

## 4. Extended Abstract Skeleton

```text
Background:
Biomedical question-answering datasets have enabled progress in clinical NLP, but Spanish resources focused on pregnancy and maternal health remain limited.

Objective:
We introduce MaternaQA-es, a synthetic Spanish clinical QA dataset designed to support maternal-health NLP research, instruction tuning, and evidence-grounded QA.

Methods:
We curated a collection of Spanish medical documents related to pregnancy and maternal care, extracted and cleaned page-level text, constructed clinically meaningful chunks with source metadata, generated synthetic QA pairs using structured LLM outputs, and produced multiple publication variants.

Results:
The current release contains 5,727 QA pairs derived from 1,953 source chunks and 57 source PDFs, with train/validation/test splits assigned at the document level. Post-hoc quality assessment used grounding analysis and RAGAS faithfulness and answer-relevancy metrics on stratified samples.

Conclusion:
MaternaQA-es provides a documented, source-traceable dataset for Spanish maternal-health QA research. Its intended uses include dataset analysis, supervised fine-tuning, and grounded QA experimentation; it is not intended for direct clinical decision-making.
```

## 5. Contribution Bullets

Use these in the Introduction:

> This work makes the following contributions:
>
> 1. We introduce **MaternaQA-es**, a synthetic Spanish clinical QA dataset focused on pregnancy and maternal health.
> 2. We document a source-traceable dataset construction methodology from curated medical PDFs, including extraction, cleaning, chunking, synthetic QA generation, and document-level split assignment.
> 3. We release multiple dataset variants: a closed-book SFT format, a grounded SFT format, and a flat JSONL format for audit and analysis.
> 4. We provide quality assessment through source-traceability checks, grounding analysis, RAGAS-based faithfulness and relevancy estimates, and explicit discussion of limitations and responsible use.
> 5. We position the dataset as a research resource for Spanish maternal-health NLP, retrieval-augmented QA, and downstream fine-tuning studies.

## 6. Dataset Description Paragraph

> MaternaQA-es is a Spanish synthetic clinical question-answering dataset focused on pregnancy and maternal health. Each QA pair is derived from a curated medical-document chunk and retains metadata that supports source tracing, split assignment, and downstream analysis. The dataset is distributed in three complementary variants: a closed-book supervised fine-tuning format, a grounded context-question-answer format, and a flat JSONL audit format containing metadata for inspection and research.

## 7. Methodology Paragraph

> The dataset was constructed through a multi-stage process designed to preserve provenance and reduce leakage. First, source PDFs were extracted at the page level and filtered to remove low-quality or non-clinical content. Second, retained pages were grouped into clinically meaningful chunks enriched with document, page, topic, and token metadata. Third, QA pairs were generated from accepted chunks using structured LLM outputs under constraints requiring Spanish, self-contained questions, clinically precise answers, and grounding in the source context. Finally, dataset variants were produced using document-level splits so that all content from the same PDF remained in a single partition.

## 8. Quality Paragraph

> Quality assessment combined structural and semantic checks. Structurally, examples retained source metadata and were split at the document level to reduce leakage. Semantically, we estimated grounding through context-answer overlap and evaluated stratified samples using RAGAS faithfulness and answer-relevancy metrics. These metrics provide evidence of source consistency and answer relevance, but they are not interpreted as clinical validation. We therefore report the dataset as a research resource and identify clinician-led review as an important future step.

## 9. Responsible Use Paragraph

> MaternaQA-es is intended for research in Spanish clinical NLP, including supervised fine-tuning, evidence-grounded QA, retrieval-augmented generation, and dataset analysis. It is not intended for direct patient care, diagnosis, treatment recommendation, or replacement of clinical guidelines. Models trained or evaluated with this dataset require independent validation and clinical oversight before any high-stakes use.

## 10. Bridge to the Second Paper

Use this to mention future fine-tuning without making it central:

> Although MaternaQA-es was designed to support downstream fine-tuning and benchmarking of open language models, the present paper focuses on the dataset itself: its motivation, construction, documentation, quality assessment, and responsible release. A separate study will evaluate the impact of the dataset on parameter-efficient adaptation and evidence-grounded QA performance.

## 11. Strong Opening Sentence Options

1. > Pregnancy and maternal health are clinically important domains in which Spanish question-answering resources remain limited.
2. > The development of safe and useful clinical language models requires not only model architectures, but also well-documented datasets with clear provenance, scope, and limitations.
3. > While biomedical QA benchmarks have advanced medical NLP, few resources focus on Spanish maternal-health content with source-traceable synthetic question-answer pairs.

## 12. Strong Closing Sentence Options

1. > By documenting both the dataset and its construction process, MaternaQA-es aims to support reproducible research in Spanish maternal-health NLP while making its limitations explicit.
2. > The dataset provides a foundation for future work on clinically grounded QA, parameter-efficient fine-tuning, and expert-validated maternal-health language technologies.
3. > MaternaQA-es should be understood as a research resource: useful for experimentation and benchmarking, but not a substitute for clinical validation or professional medical judgment.
