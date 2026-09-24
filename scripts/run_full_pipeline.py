from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from utils import (
    default_artifacts_dir,
    default_corpus_dir,
    default_datasets_dir,
    default_metadata_dir,
    default_pdfs_dir,
    default_reports_dir,
    default_tables_dir,
    project_root,
)


def parse_args() -> argparse.Namespace:
    artifacts_dir = default_artifacts_dir()
    datasets_dir = default_datasets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Spanish obstetrics LM corpus pipeline. "
            "Add PDFs to the input directory, run this script, and the final JSONL files are rebuilt."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_pdfs_dir())
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--datasets-dir", type=Path, default=datasets_dir)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-clinical-score", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Base model name; used to select the matching tokenizer automatically.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=100,
        help="Number of real tokenizer tokens to overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Optional tokenizer override for chunk overlap.",
    )
    parser.add_argument("--samples-per-pdf", type=int, default=5)
    # Phase 7-8 optional steps
    parser.add_argument(
        "--extract-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run table extraction step (requires pdfplumber).",
    )
    parser.add_argument("--table-strategy", choices=["lines", "text", "both"], default="lines")
    parser.add_argument("--table-min-rows", type=int, default=2)
    parser.add_argument("--table-min-cols", type=int, default=2)
    parser.add_argument(
        "--generate-qa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run synthetic QA generation step (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--qa-dry-run",
        action="store_true",
        help="Show QA generation cost estimate without calling API.",
    )
    return parser.parse_args()


def run_step(label: str, command: List[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    corpus_dir = default_corpus_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "corpus"
    metadata_dir = default_metadata_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "metadata"
    reports_dir = default_reports_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "reports"
    tables_dir = default_tables_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "tables"
    datasets_dir = args.datasets_dir
    lm_dir = datasets_dir / "lm"
    qa_dir = datasets_dir / "qa"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    raw_pages = corpus_dir / "raw_pages.jsonl"
    inventory = metadata_dir / "inventory.json"
    clean_pages = corpus_dir / "clean_pages.jsonl"
    cleaning_report = reports_dir / "cleaning_report.json"
    tables = tables_dir / "tables.jsonl"
    table_report = reports_dir / "table_extraction_report.json"
    chunks = corpus_dir / "chunks.jsonl"
    train = lm_dir / "train_lm.jsonl"
    validation = lm_dir / "validation_lm.jsonl"
    test = lm_dir / "test_lm.jsonl"
    build_report = reports_dir / "build_report.json"
    audit_report = reports_dir / "audit_report.json"

    run_step(
        "Extract PDFs",
        [
            sys.executable,
            "scripts/extract_pdfs.py",
            "--input-dir",
            str(args.input_dir),
            "--output",
            str(raw_pages),
            "--inventory-output",
            str(inventory),
            "--recursive" if args.recursive else "--no-recursive",
        ],
    )
    run_step(
        "Clean Pages",
        [
            sys.executable,
            "scripts/clean_text.py",
            "--input",
            str(raw_pages),
            "--output",
            str(clean_pages),
            "--report-output",
            str(cleaning_report),
            "--inventory",
            str(inventory),
        ],
    )
    if args.extract_tables:
        run_step(
            "Extract Tables",
            [
                sys.executable,
                "scripts/extract_tables.py",
                "--input-dir",
                str(args.input_dir),
                "--output",
                str(tables),
                "--report-output",
                str(table_report),
                "--strategy",
                str(args.table_strategy),
                "--min-rows",
                str(args.table_min_rows),
                "--min-cols",
                str(args.table_min_cols),
                "--recursive" if args.recursive else "--no-recursive",
            ],
        )
    build_command = [
            sys.executable,
            "scripts/build_lm_dataset.py",
            "--input",
            str(clean_pages),
            "--inventory",
            str(inventory),
            "--chunks-output",
            str(chunks),
            "--train-output",
            str(train),
            "--validation-output",
            str(validation),
            "--test-output",
            str(test),
            "--build-report-output",
            str(build_report),
            "--validation-ratio",
            str(args.validation_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--min-clinical-score",
            str(args.min_clinical_score),
            "--min-tokens",
            str(args.min_tokens),
            "--max-tokens",
            str(args.max_tokens),
            "--model-name",
            str(args.model_name),
            "--overlap-tokens",
            str(args.overlap_tokens),
        ]
    if args.tokenizer_name:
        build_command.extend(["--tokenizer-name", str(args.tokenizer_name)])
    run_step("Build LM Dataset", build_command)
    run_step(
        "Audit Dataset",
        [
            sys.executable,
            "scripts/audit_dataset.py",
            "--raw-pages",
            str(raw_pages),
            "--clean-pages",
            str(clean_pages),
            "--chunks",
            str(chunks),
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--test",
            str(test),
            "--inventory",
            str(inventory),
            "--output",
            str(audit_report),
            "--table-report",
            str(table_report),
            "--samples-per-pdf",
            str(args.samples_per_pdf),
            "--seed",
            str(args.seed),
        ],
    )

    # Phase 7: optional synthetic QA generation
    if args.generate_qa:
        qa_cmd = [
            sys.executable,
            "scripts/generate_synthetic_qa.py",
            "--input",
            str(train),
            "--model",
            "gpt-5.4-mini",
        ]
        if args.qa_dry_run:
            qa_cmd.append("--dry-run")
        run_step("Generate Synthetic QA", qa_cmd)

    print("\nPipeline complete.", flush=True)
    print(f"Train JSONL: {train}", flush=True)
    print(f"Validation JSONL: {validation}", flush=True)
    print(f"Test JSONL: {test}", flush=True)
    if args.extract_tables:
        print(f"Tables JSONL: {tables}", flush=True)
    print(f"Audit report: {audit_report}", flush=True)
    if args.extract_tables:
        print(f"Table report: {table_report}", flush=True)
    if args.generate_qa:
        print(f"QA output: {qa_dir / 'synthetic_qa_sft.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
