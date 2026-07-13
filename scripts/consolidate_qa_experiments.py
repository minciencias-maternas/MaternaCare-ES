#!/usr/bin/env python3
"""Consolidate synthetic QA experiment reports into JSON and Markdown."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


QA_DIR = Path("datasets/obstetrics/qa")
OUT_JSON = QA_DIR / "qa_experiments_consolidated.json"
OUT_MD = QA_DIR / "qa_experiments_consolidated.md"

META_ANSWER_PATTERNS = [
    r"\bel fragmento\b",
    r"\bel contexto\b",
    r"\bel texto\b",
    r"\bseg[uú]n el\b",
    r"\bde acuerdo con\b",
    r"\bcon base en\b",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_raw(path: Path) -> Dict[str, Any]:
    rows = load_jsonl(path)
    answer_words = [len(r["respuesta"].split()) for r in rows]
    question_words = [len(r["pregunta"].split()) for r in rows]
    meta_answer_refs = sum(
        any(re.search(p, r["respuesta"].lower()) for p in META_ANSWER_PATTERNS)
        for r in rows
    )
    question_with_y = sum(" y " in r["pregunta"].lower() for r in rows)
    return {
        "raw_file": path.name,
        "avg_question_words": round(mean(question_words), 2) if question_words else 0.0,
        "avg_answer_words": round(mean(answer_words), 2) if answer_words else 0.0,
        "min_answer_words": min(answer_words) if answer_words else 0,
        "max_answer_words": max(answer_words) if answer_words else 0,
        "meta_answer_refs": meta_answer_refs,
        "questions_with_y": question_with_y,
        "types": dict(Counter(r["tipo"] for r in rows)),
        "difficulty": dict(Counter(r["dificultad"] for r in rows)),
    }


def main() -> None:
    experiments: List[Dict[str, Any]] = []
    for report_path in sorted(QA_DIR.glob("report_*.json")):
        report = load_json(report_path)
        suffix = report_path.name.removeprefix("report_").removesuffix(".json")
        raw_path = QA_DIR / f"raw_{suffix}.jsonl"
        raw_summary = summarize_raw(raw_path) if raw_path.exists() else {}
        experiments.append(
            {
                "experiment_id": suffix,
                "report_file": report_path.name,
                **report,
                **raw_summary,
            }
        )

    OUT_JSON.write_text(
        json.dumps({"experiments": experiments}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# QA Synthetic Experiments Consolidated Report",
        "",
        "| Experimento | Generador | Verificador | QA | Acceptance | Faith | Rel | Roundtrip | Overlap | Avg answer words | Meta refs en respuestas |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in experiments:
        q = e.get("quality", {})
        g = e.get("grounding", {})
        models = e.get("models", {})
        lines.append(
            f"| {e['experiment_id']} | {models.get('generator_model')} | "
            f"{models.get('verifier_model')} | {e.get('qa_pairs')} | "
            f"{q.get('acceptance_rate', 0):.1%} | {q.get('avg_faithfulness', 0):.3f} | "
            f"{q.get('avg_answer_relevancy', 0):.3f} | {q.get('avg_roundtrip_consistency', 0):.3f} | "
            f"{g.get('avg_context_answer_overlap', 0):.3f} | {e.get('avg_answer_words', 0):.1f} | "
            f"{e.get('meta_answer_refs', 0)} |"
        )
    lines.extend(["", "## Notas", "- `Meta refs en respuestas` cuenta respuestas que mencionan explícitamente el fragmento/contexto/texto."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
