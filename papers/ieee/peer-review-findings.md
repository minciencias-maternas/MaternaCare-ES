# MaternaQA-es — Hallazgos del Pair Review

Documento generado como parte de la revisión de pares del manuscrito IEEE
`main.tex`. Cada sección plantea preguntas, problemas detectados, severidad y
recomendaciones accionables.

---

## 1. Problemas Críticos (amenazan conclusiones principales)

### 1.1 Sin baseline experimental

**Pregunta:** ¿El dataset es usable? ¿Representa un desafío no trivial para
modelos actuales?

**Problema:** El paper difiere todos los experimentos de fine-tuning a un
estudio downstream separado. SQuAD, PubMedQA, MedMCQA y BioASQ incluyeron al
menos un baseline en su paper de presentación. Un dataset paper sin evidencia
de que el dataset funciona para la tarea que propone es difícil de defender.

**Severidad:** Crítica.

**Recomendación:** Agregar al menos un baseline mínimo:
- Zero-shot o few-shot sobre el test set con un LLM español (ej. Gemma 4,
  Llama 3 Spanish)
- O un retrieval baseline simple (BM25 / Sentence-BERT) sobre la variante grounded

**Sección afectada:** Nueva sección de Baseline / Preliminary Experiments.

---

### 1.2 Modelo generador no identificado

**Pregunta:** ¿Qué LLM generó los pares QA? ¿Con qué parámetros?

**Problema:** El paper dice "LLMs" y "the generation prompt" sin especificar
modelo, versión, temperatura ni proveedor. Esto es un problema de
reproducibilidad. El script `generate_synthetic_qa.py` (línea 991) revela que
el default es `gpt-5.4-mini` de OpenAI (modelos activos a mayo 2026), con
parámetros de temperatura no explicitados en el paper.

**Severidad:** Crítica.

**Recomendación:** Declarar explícitamente en la sección de metodología:
- Modelo generador: `gpt-5.4-mini` (OpenAI, mayo 2026)
- Temperatura, max_tokens si aplica
- Prompt template resumido o referencia al repositorio

**Sección afectada:** `\section{Synthetic QA Generation Methodology}`.

---

### 1.3 Answer relevancy de 0.5583 en test sin análisis

**Pregunta:** ¿Por qué casi la mitad de las respuestas del test set son
juzgadas como poco relevantes a sus preguntas? ¿Es ruido del evaluador o
problema real de calidad?

**Problema:** El test set tiene answer relevancy de 0.5583, muy por debajo de
train (0.6466) y validation (0.6812). El paper reporta el número sin análisis
cualitativo ni explicación. Un revisor va a exigir saber:
- ¿Son ciertos tipos de pregunta los que bajan el score?
- ¿Hay tópicos específicos con peor desempeño?
- ¿El evaluador RAGAS tiene sesgo en el test set?
- ¿Los chunks del test set son inherentemente más difíciles?

**Severidad:** Crítica.

**Recomendación:** Agregar una subsección de análisis cualitativo de los peores
casos, mostrando ejemplos con baja relevancy y explicando si el problema es del
dataset o del evaluador.

**Sección afectada:** `\subsection{RAGAS Evaluation}`.

---

### 1.4 Fallos de scoring RAGAS no explicados

**Pregunta:** ¿Qué significa que una llamada de scoring "falle"? ¿Los pares no
evaluados son sistemáticamente distintos?

**Problema:** El paper menciona que faithfulness se calculó sobre 299/300 train
y 99/100 validation porque "one scoring call failed". Pero no se explica si
fue error de API, timeout, respuesta vacía, o rechazo del modelo. Peor aún, no
se analiza si los ejemplos excluidos son diferentes (más largos, más
complejos) de los evaluados, lo que introduciría sesgo de cobertura.

**Severidad:** Crítica.

**Recomendación:** Documentar la naturaleza exacta de los fallos y verificar si
los ejemplos no puntuados difieren sistemáticamente. Incluir esta información
en la sección RAGAS.

**Sección afectada:** `\subsection{RAGAS Evaluation}`.

---

## 2. Problemas Importantes (afectan interpretación)

### 2.1 Gap piloto→final (~0.99 → ~0.77) sin explicación

**Pregunta:** Si el piloto logró faithfulness 0.9924 y relevancy 0.9947, ¿por
qué el RAGAS final da 0.77 y 0.64? ¿Son métricas comparables?

**Problema:** El gap es enorme. La explicación actual ("se deshabilitó el
evaluador intermedio por costo") no justifica por qué la evaluación post-hoc
también bajó drásticamente. Posibles causas no exploradas:
- El piloto usaba el mismo LLM como juez que como generador (self-evaluation
  bias)
- El piloto fue sobre datos cherry-picked
- Las métricas del piloto no son RAGAS sino un custom judge diferente
- El RAGAS con `gpt-4o-mini` como juez es más estricto que el evaluador del
  piloto

**Severidad:** Importante.

**Recomendación:** Explicar claramente qué métrica usó el piloto, qué modelo
juez, y por qué no son comparables con RAGAS. O eliminar la mención del piloto
si induce a confusión.

**Sección afectada:** `\section{Quality Assessment}`.

---

### 2.2 Cero ejemplos cualitativos del dataset

**Pregunta:** ¿Cómo son los pares QA reales? ¿El español es clínicamente
correcto? ¿Hay alucinaciones?

**Problema:** El paper describe el dataset exhaustivamente con números, pero no
muestra un solo ejemplo de pregunta-respuesta. El lector no puede formarse un
juicio sobre la calidad lingüística o clínica del contenido.

**Severidad:** Importante.

**Recomendación:** Agregar una tabla con 4-6 ejemplos representativos
cubriendo distintos tipos de pregunta, tópicos y calidad (incluir uno con baja
faithfulness como muestra de honestidad). Traducir los ejemplos al inglés en
notas al pie o en una columna separada si la venue lo requiere.

**Sección afectada:** Nueva subsección o tabla en Quality Assessment.

---

### 2.3 Distribución de tipos de pregunta no reportada

**Pregunta:** ¿El dataset realmente cubre los 6 tipos declarados? ¿Qué
porcentaje es factual vs. razonamiento?

**Problema:** El script de generación produce factual, definición, comparación,
razonamiento, aplicación e hipotético, pero el paper no reporta cuántos pares
hay de cada tipo. Sin este dato no se sabe si el dataset es mayormente factual
(trivial para LLMs) o tiene proporción significativa de razonamiento clínico.

**Datos reales del dataset (extraídos 2026-05-24):**

| Tipo | Train | % | Val | % | Test | % | Total | % |
|---|---|---|---|---|---|---|---|---|
| factual | 1,458 | 28.6 | 82 | 26.8 | 101 | 30.8 | 1,641 | 28.7 |
| aplicacion | 1,523 | 29.9 | 85 | 27.8 | 94 | 28.7 | 1,702 | 29.7 |
| razonamiento | 1,147 | 22.5 | 67 | 21.9 | 85 | 25.9 | 1,299 | 22.7 |
| definicion | 586 | 11.5 | 42 | 13.7 | 28 | 8.5 | 656 | 11.5 |
| hipotetico | 234 | 4.6 | 18 | 5.9 | 15 | 4.6 | 267 | 4.7 |
| comparacion | 145 | 2.8 | 12 | 3.9 | 5 | 1.5 | 162 | 2.8 |

Dominan factual + aplicación (~58%). Comparación e hipotético son minoritarios
(~7.5%). La distribución es razonablemente consistente entre splits.

**Severidad:** Importante.

**Recomendación:** Incluir esta tabla en el paper (sección Dataset Statistics
o QA Generation).

**Sección afectada:** `\section{Synthetic QA Generation Methodology}` o
`\section{Dataset Statistics and Analysis}`.

---

### 2.4 Sin validación humana

**Pregunta:** ¿Algún experto clínico revisó aunque sea una muestra pequeña?

**Problema:** Para un dataset clínico, incluso 50-100 pares revisados por un
obstetra/ginecólogo serían transformadores. El paper reconoce esta limitación
pero no mitiga el riesgo con ninguna validación experta. La combinación de
datos sintéticos + sin validación humana + dominio clínico es una debilidad
importante.

**Severidad:** Importante.

**Recomendación:** Si no es factible ahora, planificar explícitamente una
ronda de validación con al menos un clínico y reportar acuerdo inter-anotador
sobre una muestra estratificada de 100 pares. Agregar esto como "ongoing work"
concreto, no solo "future work".

**Sección afectada:** `\section{Limitations and Ethical Considerations}`.

---

### 2.5 El "grounding analysis" es demasiado vago

**Pregunta:** ¿Cómo se mide el grounding? ¿Qué threshold define "bajo
grounding"?

**Problema:** El paper menciona "grounding analysis" pero no explica el método.
El README del repo reporta "average context-answer overlap 0.6836" sin detalle
de la métrica (¿Jaccard? ¿ROUGE? ¿BERTScore?). Los 27 pares (0.54%) con "bajo
grounding" no se analizan por split, tópico ni tipo.

**Severidad:** Importante.

**Recomendación:** Especificar la métrica de grounding usada, reportar la
distribución por split, y caracterizar brevemente los pares con bajo grounding.

**Sección afectada:** `\section{Quality Assessment}`.

---

### 2.6 Modelo juez RAGAS no declarado

**Pregunta:** ¿Qué LLM se usó como juez en RAGAS? ¿Qué embeddings?

**Problema:** Los scores de RAGAS varían significativamente según el modelo
juez. El script `evaluate_qa_with_ragas.py` (líneas 138-147) revela que se usó
`gpt-4o-mini` como LLM judge y `text-embedding-3-small` para embeddings, pero
el paper no lo menciona. `gpt-4o-mini` fue deprecado en febrero 2026, lo cual
es relevante para reproducibilidad futura.

**Severidad:** Importante.

**Recomendación:** Declarar el modelo juez, el modelo de embeddings y la
versión de RAGAS usada. Notar que `gpt-4o-mini` fue deprecado.

**Sección afectada:** `\subsection{RAGAS Evaluation}`.

---

### 2.7 Sin intervalos de confianza en scores RAGAS

**Pregunta:** ¿La diferencia entre train (0.77) y test (0.71) en faithfulness
es estadísticamente significativa?

**Problema:** Se reportan point estimates (0.7726, 0.7826, 0.7132) sobre
muestras de 100-300 pares sin intervalos de confianza. Con tan pocas muestras,
la incertidumbre puede ser grande. La aparente degradación en test podría ser
ruido.

**Severidad:** Importante.

**Recomendación:** Reportar intervalos de confianza bootstrap al 95% para cada
métrica y split, o al menos desviación estándar.

**Sección afectada:** `\subsection{RAGAS Evaluation}`.

---

## 3. Problemas Menores

### 3.1 Tabla de design goals redundante

**Problema:** Table I es traducción textual de los 7 goals listados en prosa.
No agrega información. O se elimina o se condensa en una lista.

**Severidad:** Menor.

### 3.2 "Clinical-score heuristic" no definida

**Problema:** El paper menciona que los chunks se filtran por "a
clinical-score heuristic" sin explicar qué es ni cómo se calcula.

**Severidad:** Menor.

### 3.3 80-token overlap en chunking no explicado

**Problema:** ¿Los chunks adyacentes comparten 80 tokens de texto? Esto podría
generar pares QA semánticamente redundantes entre chunks consecutivos.

**Severidad:** Menor.

### 3.4 Validation split muy pequeño

**Problema:** 306 pares de solo 2 PDFs. Baja representatividad. Cualquier
métrica sobre validation tiene alta varianza.

**Severidad:** Menor.

### 3.5 Falta tabla comparativa de datasets relacionados

**Problema:** Related Work menciona SQuAD, PubMedQA, MedQuAD, MedMCQA, BioASQ,
RealMedQA en prosa pero sin tabla estructurada. Una tabla comparativa con
columnas: idioma, dominio, tamaño, tipos QA, splits, validación humana,
variantes es estándar en dataset papers.

**Severidad:** Menor.

### 3.6 Sin identificador persistente del dataset

**Problema:** El README menciona `JhonHander/obstetrics-qa-synthetic-es` en
Hugging Face, pero el paper no cita el HF dataset ID, ni un DOI de Zenodo, ni
otro identificador persistente.

**Severidad:** Menor.

### 3.7 Diferencia entre splits LM (54/3/3) y QA (52/2/3) no explicada

**Problema:** El paper reporta 57 PDFs totales con splits QA de 52/2/3. Pero
el corpus LM tiene 54/3/3. ¿Por qué 2 PDFs del train de LM no generaron QA?

**Severidad:** Menor.

---

## 4. Elementos Visuales y Tablas Recomendados

### 4.1 Diagrama de pipeline (Figura 1)

Flujo visual: PDF Collection → Extraction (PyMuPDF/pdfplumber) → Cleaning →
Chunking → QA Generation (gpt-5.4-mini) → Variants (closed-book, grounded,
flat) → Quality Assessment (RAGAS).

### 4.2 Tabla de ejemplos cualitativos

4-6 pares QA con: pregunta, respuesta, contexto fuente, tipo, split, y nota de
calidad. Incluir al menos uno con baja faithfulness.

### 4.3 Distribución de tópicos por split

Bar chart horizontal mostrando frecuencia de los 11+ tópicos principales en
train/validation/test.

### 4.4 Distribución de tipos de pregunta por split

Tabla con los 6 tipos y sus conteos/porcentajes por split.

### 4.5 Tabla comparativa de datasets

Columnas: Dataset, Idioma, Dominio, Tamaño, Tipos QA, Estrategia de split,
Validación humana, Variantes, Año.

### 4.6 Distribución de scores RAGAS

Box plots o histogramas de faithfulness y answer relevancy por split,
mostrando dispersión además de medias.

### 4.7 Diagrama Sankey de flujo de corpus

63 PDFs → 5,856 páginas → 5,176 kept → 2,268 chunks → 5,727 QA pairs, con
razones de descarte en cada etapa.

---

## 5. Resumen de Acciones Prioritarias

1. [x] Verificar DOIs de referencias (completado en sesión previa)
2. [ ] Declarar modelo generador (`gpt-5.4-mini`) y modelo juez RAGAS
   (`gpt-4o-mini`) en el paper
3. [ ] Agregar tabla de ejemplos cualitativos (4-6 pares QA reales)
4. [ ] Agregar tabla de distribución de tipos de pregunta
5. [ ] Analizar y documentar la baja answer relevancy en test (0.5583)
6. [ ] Agregar tabla comparativa de datasets relacionados
7. [ ] Explicar el gap piloto→final o eliminar la mención del piloto
8. [ ] Agregar intervalos de confianza a scores RAGAS
9. [ ] Documentar naturaleza de fallos de scoring RAGAS
10. [ ] Agregar baseline experimental mínimo
11. [ ] Agregar diagrama de pipeline
12. [ ] Agregar distribución de tópicos por split
13. [ ] Especificar métrica de grounding y reportar por split

---

*Documento generado como parte del pair review del manuscrito IEEE
MaternaQA-es. Fecha: 2026-05-24.*
