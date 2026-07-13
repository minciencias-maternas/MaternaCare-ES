# Paper Positioning and Scope

## 1. What This Paper Is

This manuscript should be positioned as a **dataset creation paper**, also known as a **data descriptor**, **resource paper**, or **dataset paper**.

The paper is not primarily about software engineering. The software pipeline is part of the reproducibility story, but the scientific contribution is the dataset and the methodological reasoning behind its construction.

### Correct framing

> This paper presents MaternaQA-es, a synthetic Spanish clinical question-answering dataset focused on pregnancy and maternal health. The paper documents the motivation, source selection, corpus construction, QA generation methodology, dataset variants, quality assessment, limitations, and intended uses of the resource.

### Framing to avoid

> This paper presents a pipeline to generate QA data.

That framing is too narrow. It makes the work sound like tooling rather than a scientific dataset contribution.

## 2. Dataset Theme

The manuscript should emphasize:

- pregnancy;
- maternal health;
- maternity care;
- prenatal care;
- labor and delivery;
- postpartum care;
- fetal monitoring;
- hypertensive disorders of pregnancy;
- gestational diabetes;
- obstetric emergencies;
- newborn/perinatal care when connected to pregnancy or delivery.

The project may include obstetrics and gynecology source material, but the paper title and central framing should lean toward **pregnancy and maternal health**, not broad gynecology.

## 3. Working Name

**MaternaQA-es**

Recommended expansion:

> MaternaQA-es: A Spanish synthetic clinical question-answering dataset for pregnancy and maternal health.

## 4. Main Research Gap

A strong gap statement should combine three shortages:

1. **Language gap**: few open Spanish clinical QA resources compared with English biomedical QA datasets.
2. **Domain gap**: maternal-health and pregnancy-specific QA resources are limited relative to general biomedical or medical-exam datasets.
3. **Documentation gap**: many datasets provide examples but do not fully document source provenance, design rationale, generation decisions, quality checks, and limitations.

Possible wording:

> Despite growing interest in biomedical question answering and clinical language models, Spanish resources focused on pregnancy and maternal health remain limited. Existing biomedical QA datasets are often English-centric, broad-domain, or oriented toward exam-style or research-abstract QA rather than maternal-health educational and clinical-document contexts. MaternaQA-es addresses this gap by providing a documented synthetic QA resource built from curated Spanish medical documents with explicit provenance and quality assessment.

## 5. Main Contribution

The contribution is the **resource and its construction methodology**, not merely the code.

### Contribution statement

> We contribute MaternaQA-es, a Spanish synthetic clinical QA dataset focused on pregnancy and maternal health, together with a reproducible construction methodology, document-level data splits, QA variants for closed-book and grounded supervised fine-tuning, and quality assessment reports based on source traceability, grounding analysis, and RAGAS metrics.

## 6. Boundary Between Paper 1 and Paper 2

### Paper 1: dataset creation paper

Focus:

- dataset motivation;
- source corpus;
- construction methodology;
- QA generation design;
- quality assessment;
- dataset schema;
- availability;
- limitations;
- intended uses.

Model training appears only as a downstream use.

### Paper 2: fine-tuning benchmark paper

Focus:

- TRL/PEFT/bitsandbytes;
- QLoRA;
- model selection;
- baseline vs fine-tuned models;
- closed-book vs grounded comparison;
- metrics;
- error analysis;
- clinical usefulness of adaptation.

## 7. Recommended Manuscript Category

Potential submission categories:

- dataset paper;
- resource paper;
- data descriptor;
- NLP resource paper;
- clinical NLP dataset paper;
- applied AI in health data resource.

Potential venues to inspect later:

- Scientific Data;
- Data in Brief;
- LREC-COLING style resource papers;
- ACL/EMNLP Findings resource papers;
- AMIA for clinical NLP/data resources;
- BMC Medical Informatics and Decision Making;
- Journal of Biomedical Informatics.

## 8. One-Sentence Thesis

> Carefully documented synthetic QA generation from curated maternal-health documents can produce a reusable Spanish clinical NLP resource whose value lies not only in its examples, but in its traceability, dataset design, quality controls, and explicit limitations.
