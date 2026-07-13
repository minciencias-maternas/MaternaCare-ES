#!/usr/bin/env python3
"""Convert model evaluation JSON/JSONL outputs to CSV for academic paper analysis.

Produces three CSV files in a target directory:
1. master_eval.csv — one row per (model, qa_id) with all metrics and metadata
2. model_summary.csv — one row per model with aggregate scores
3. wide_comparison.csv — one row per qa_id, side-by-side answers and scores per model

Usage:
    python scripts/convert_eval_to_csv.py --outputs-dir outputs/ --out-dir csv_outputs/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert evaluation JSONL to CSV")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing model subdirectories with test_eval.jsonl files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="csv_outputs",
        help="Directory to write the resulting CSV files",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_json_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_record(record: dict, model_label: str) -> dict:
    """Flatten a single JSONL record into a flat dict suitable for CSV."""
    flat = {}

    # Model identification
    flat["model_label"] = model_label
    flat["model_name"] = record.get("model_name", "")
    flat["model_role"] = record.get("model_role", "")
    flat["dataset_variant"] = record.get("dataset_variant", "")
    flat["adapter_dir"] = record.get("adapter_dir", "")

    # QA core fields
    flat["qa_id"] = record.get("qa_id", "")
    flat["question"] = record.get("question", "")
    flat["generated_answer"] = record.get("generated_answer", "")
    flat["reference_answer"] = record.get("reference_answer", "")
    flat["source_context"] = record.get("source_context", "")

    # Metrics
    flat["faithfulness"] = record.get("faithfulness", "")
    flat["answer_relevancy"] = record.get("answer_relevancy", "")
    flat["answer_correctness"] = record.get("answer_correctness", "")
    flat["semantic_similarity"] = record.get("semantic_similarity", "")

    # Metadata (nested dict)
    meta = record.get("metadata", {})
    flat["source_pdf"] = meta.get("source_pdf", "")
    flat["tipo"] = meta.get("tipo", "")
    flat["dificultad"] = meta.get("dificultad", "")
    flat["chunk_id"] = meta.get("chunk_id", "")
    flat["section"] = meta.get("section", "")
    flat["section_type"] = meta.get("section_type", "")
    flat["content_role"] = meta.get("content_role", "")
    flat["clinical_score"] = meta.get("clinical_score", "")
    flat["token_estimate"] = meta.get("token_estimate", "")
    flat["split"] = meta.get("split", "")
    flat["pages"] = ";".join(str(p) for p in meta.get("pages", [])) if isinstance(meta.get("pages"), list) else meta.get("pages", "")
    flat["topics"] = ";".join(meta.get("topics", [])) if isinstance(meta.get("topics"), list) else meta.get("topics", "")

    # Error flag
    flat["has_error"] = bool(record.get("error"))

    return flat


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_master_csv(models_data: dict[str, list[dict]], out_dir: Path) -> None:
    """Build the long-format master CSV: one row per (model, qa_id)."""
    fieldnames = [
        "model_label",
        "model_name",
        "model_role",
        "dataset_variant",
        "adapter_dir",
        "qa_id",
        "question",
        "generated_answer",
        "reference_answer",
        "source_context",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "semantic_similarity",
        "tipo",
        "dificultad",
        "source_pdf",
        "chunk_id",
        "section",
        "section_type",
        "content_role",
        "clinical_score",
        "token_estimate",
        "split",
        "pages",
        "topics",
        "has_error",
    ]

    rows = []
    for model_label, records in models_data.items():
        for record in records:
            rows.append(flatten_record(record, model_label))

    out_path = out_dir / "master_eval.csv"
    write_csv(out_path, rows, fieldnames)
    print(f"[csv] master_eval.csv: {len(rows)} rows written to {out_path}")


def build_summary_csv(summaries: dict[str, dict], out_dir: Path) -> None:
    """Build the model summary CSV: one row per model with aggregate scores."""
    fieldnames = [
        "model_label",
        "model_name",
        "model_role",
        "dataset_variant",
        "adapter_dir",
        "predictions_evaluated",
        "predictions_with_error",
        "avg_faithfulness",
        "avg_answer_relevancy",
        "avg_answer_correctness",
        "avg_semantic_similarity",
    ]

    rows = []
    for model_label, summary in summaries.items():
        s = summary.get("summary", {})
        meta = summary.get("metadata", {})
        rows.append({
            "model_label": model_label,
            "model_name": meta.get("model_name", ""),
            "model_role": meta.get("model_role", ""),
            "dataset_variant": meta.get("dataset_variant", ""),
            "adapter_dir": meta.get("adapter_dir", ""),
            "predictions_evaluated": s.get("predictions_evaluated", ""),
            "predictions_with_error": s.get("predictions_with_error", ""),
            "avg_faithfulness": s.get("avg_faithfulness", ""),
            "avg_answer_relevancy": s.get("avg_answer_relevancy", ""),
            "avg_answer_correctness": s.get("avg_answer_correctness", ""),
            "avg_semantic_similarity": s.get("avg_semantic_similarity", ""),
        })

    out_path = out_dir / "model_summary.csv"
    write_csv(out_path, rows, fieldnames)
    print(f"[csv] model_summary.csv: {len(rows)} rows written to {out_path}")


def build_wide_csv(models_data: dict[str, list[dict]], out_dir: Path) -> None:
    """Build a wide-format CSV: one row per qa_id, columns per model."""
    # Index all records by qa_id for each model
    by_qa: dict[str, dict[str, dict]] = defaultdict(dict)
    for model_label, records in models_data.items():
        for record in records:
            qa_id = record.get("qa_id", "")
            if qa_id:
                by_qa[qa_id][model_label] = record

    if not by_qa:
        print("[csv] wide_comparison.csv: no data to write")
        return

    model_labels = sorted(models_data.keys())

    # Build fieldnames dynamically
    base_fields = ["qa_id", "question", "reference_answer", "source_context", "tipo", "dificultad", "source_pdf", "section"]
    model_fields = []
    for ml in model_labels:
        suffix = ml.replace("-", "_")
        model_fields.extend([
            f"gen_{suffix}",
            f"faith_{suffix}",
            f"rel_{suffix}",
            f"corr_{suffix}",
            f"sem_{suffix}",
        ])

    fieldnames = base_fields + model_fields

    rows = []
    for qa_id in sorted(by_qa.keys()):
        models_for_qa = by_qa[qa_id]
        # Use the first available model to pull base fields
        first_record = next(iter(models_for_qa.values()))
        meta = first_record.get("metadata", {})
        row = {
            "qa_id": qa_id,
            "question": first_record.get("question", ""),
            "reference_answer": first_record.get("reference_answer", ""),
            "source_context": first_record.get("source_context", ""),
            "tipo": meta.get("tipo", ""),
            "dificultad": meta.get("dificultad", ""),
            "source_pdf": meta.get("source_pdf", ""),
            "section": meta.get("section", ""),
        }
        for ml in model_labels:
            suffix = ml.replace("-", "_")
            rec = models_for_qa.get(ml, {})
            row[f"gen_{suffix}"] = rec.get("generated_answer", "")
            row[f"faith_{suffix}"] = rec.get("faithfulness", "")
            row[f"rel_{suffix}"] = rec.get("answer_relevancy", "")
            row[f"corr_{suffix}"] = rec.get("answer_correctness", "")
            row[f"sem_{suffix}"] = rec.get("semantic_similarity", "")
        rows.append(row)

    out_path = out_dir / "wide_comparison.csv"
    write_csv(out_path, rows, fieldnames)
    print(f"[csv] wide_comparison.csv: {len(rows)} rows written to {out_path}")


def main() -> int:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)

    if not outputs_dir.exists():
        print(f"Error: outputs directory not found: {outputs_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover model subdirectories
    model_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print(f"Error: no model subdirectories found in {outputs_dir}", file=sys.stderr)
        return 1

    models_data: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for model_dir in sorted(model_dirs):
        model_label = model_dir.name
        jsonl_path = model_dir / "test_eval.jsonl"
        json_path = model_dir / "test_eval.json"

        if not jsonl_path.exists():
            print(f"[skip] {model_label}: {jsonl_path} not found")
            continue

        records = load_jsonl(jsonl_path)
        if not records:
            print(f"[skip] {model_label}: no records in {jsonl_path}")
            continue

        models_data[model_label] = records
        print(f"[load] {model_label}: {len(records)} records from {jsonl_path}")

        summary = load_json_summary(json_path)
        if summary:
            # Enrich summary with metadata from first record for consistency
            first = records[0]
            summary["metadata"] = {
                "model_name": first.get("model_name", ""),
                "model_role": first.get("model_role", ""),
                "dataset_variant": first.get("dataset_variant", ""),
                "adapter_dir": first.get("adapter_dir", ""),
            }
            summaries[model_label] = summary
        else:
            print(f"[warn] {model_label}: summary JSON not found at {json_path}")

    if not models_data:
        print("Error: no evaluation data loaded.", file=sys.stderr)
        return 1

    build_master_csv(models_data, out_dir)
    build_summary_csv(summaries, out_dir)
    build_wide_csv(models_data, out_dir)

    print(f"\nAll CSV files written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
