# Documentación técnica de MaternaCare-ES

Documentación por tema del proyecto de fine-tuning QLoRA y benchmark RAG sobre QA clínico en español.

| Documento | Contenido |
|-----------|-----------|
| [rag.md](./rag.md) | Cómo funciona el pipeline RAG: corpus, recuperación (BM25, densa, híbrida, HyDE), generación y evaluación con RAGAS. |
| [training-qlora.md](./training-qlora.md) | Cómo se entrenó el QLoRA: cuantización 4-bit, adapters LoRA, variantes del dataset e hiperparámetros. |
| [configuracion.md](./configuracion.md) | Cómo se configura: variables de entorno, rutas por defecto, registro de modelos y argumentos de los scripts. |
| [modificacion.md](./modificacion.md) | Cómo se modifica: agregar modelos, estrategias, métricas o pasos al pipeline. |
| [estructura-y-datos.md](./estructura-y-datos.md) | Cómo se ve todo: árbol del repositorio, esquemas de datos (JSONL), salidas y reportes. |

El punto de partida general sigue siendo el [README principal](../README.md).