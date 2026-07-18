from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from utils import (
    default_artifacts_dir,
    default_corpus_dir,
    default_datasets_dir,
    default_metadata_dir,
    default_pdfs_dir,
    default_reports_dir,
    dedupe_chunks,
    enrich_chunks,
    filter_lm_chunks,
    project_root,
    read_jsonl,
    slugify,
    split_train_validation_test_by_document,
    to_lm_record,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    artifacts_dir = default_artifacts_dir()
    datasets_dir = default_datasets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally add new or changed obstetrics PDFs to the LM corpus. "
            "Only new/changed PDFs are extracted and cleaned; existing artifacts are reused."
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
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Process all discovered PDFs incrementally, replacing their previous records.",
    )
    parser.add_argument(
        "--keep-temp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep artifacts/obstetrics/_incremental for debugging.",
    )
    return parser.parse_args()


def run_step(label: str, command: List[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)


def discover_pdfs(input_dir: Path, recursive: bool) -> List[Path]:
    globber = input_dir.rglob("*.pdf") if recursive else input_dir.glob("*.pdf")
    return sorted(path.resolve() for path in globber)


def source_path_for(pdf_path: Path) -> str:
    root = project_root()
    try:
        return str(pdf_path.relative_to(root))
    except ValueError:
        return str(pdf_path.resolve())


def pdf_fingerprint(pdf_path: Path) -> Dict[str, Any]:
    stat = pdf_path.stat()
    return {
        "source_pdf": pdf_path.name,
        "source_path": source_path_for(pdf_path),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path, pdfs: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    if path.exists():
        data = load_json(path)
        records = data.get("pdfs", {})
        if isinstance(records, dict):
            return {str(k): v for k, v in records.items() if isinstance(v, dict)}

    # Bootstrap from existing artifacts so the first incremental run does not reprocess old PDFs.
    raw_pages = (
        path.parent.parent / "corpus" / "raw_pages.jsonl"
        if path.parent.name == "metadata"
        else path.parent / "raw_pages.jsonl"
    )
    if raw_pages.exists():
        existing_paths = {str(row.get("source_path", "")) for row in read_jsonl(raw_pages)}
        manifest: Dict[str, Dict[str, Any]] = {}
        by_source_path = {source_path_for(pdf): pdf for pdf in pdfs}
        for source_path in existing_paths:
            pdf = by_source_path.get(source_path)
            if pdf and pdf.exists():
                manifest[source_path] = pdf_fingerprint(pdf)
        return manifest

    return {}


def changed_pdfs(pdfs: Sequence[Path], manifest: Dict[str, Dict[str, Any]], force: bool) -> List[Path]:
    changed: List[Path] = []
    for pdf in pdfs:
        fp = pdf_fingerprint(pdf)
        old = manifest.get(fp["source_path"])
        if force or old is None:
            changed.append(pdf)
            continue
        if old.get("file_size") != fp["file_size"] or old.get("mtime_ns") != fp["mtime_ns"]:
            changed.append(pdf)
    return changed


def read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def row_source_key(row: Dict[str, Any]) -> Tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    source_path = str(row.get("source_path") or metadata.get("source_path") or "")
    source_pdf = str(row.get("source_pdf") or metadata.get("source_pdf") or "")
    return source_path, source_pdf


def remove_sources(rows: Iterable[Dict[str, Any]], source_paths: set[str], source_pdfs: set[str]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in rows:
        source_path, source_pdf = row_source_key(row)
        if source_path and source_path in source_paths:
            continue
        if not source_path and source_pdf in source_pdfs:
            continue
        if source_pdf in source_pdfs:
            continue
        kept.append(row)
    return kept


def normalized_chunk_key(row: Dict[str, Any]) -> str:
    text = str(row.get("text", ""))
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", text)
    return " ".join(text.split()[:240])


def append_new_chunks(existing_chunks: List[Dict[str, Any]], new_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_keys = {normalized_chunk_key(row) for row in existing_chunks if row.get("text")}
    counters: Counter[str] = Counter()
    for row in existing_chunks:
        chunk_id = str(row.get("chunk_id", ""))
        match = re.match(r"(.+)_(\d{5})$", chunk_id)
        if match:
            counters[match.group(1)] = max(counters[match.group(1)], int(match.group(2)))

    accepted_new: List[Dict[str, Any]] = []
    for row in dedupe_chunks(new_chunks):
        key = normalized_chunk_key(row)
        if not key or key in existing_keys:
            continue
        existing_keys.add(key)
        source_pdf = str(row.get("source_pdf", "document"))
        pdf_id = slugify(source_pdf)
        counters[pdf_id] += 1
        new_row = dict(row)
        new_row["chunk_id"] = f"{pdf_id}_{counters[pdf_id]:05d}"
        accepted_new.append(new_row)

    return existing_chunks + accepted_new


def write_manifest(path: Path, pdfs: Sequence[Path], previous: Dict[str, Dict[str, Any]]) -> None:
    updated = dict(previous)
    for pdf in pdfs:
        fp = pdf_fingerprint(pdf)
        updated[fp["source_path"]] = fp
    write_json(
        path,
        {
            "mode": "incremental",
            "pdf_count": len(updated),
            "pdfs": dict(sorted(updated.items())),
        },
    )


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    corpus_dir = default_corpus_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "corpus"
    metadata_dir = default_metadata_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "metadata"
    reports_dir = default_reports_dir() if args.artifacts_dir == default_artifacts_dir() else artifacts_dir / "reports"
    datasets_dir = args.datasets_dir
    lm_dir = datasets_dir / "lm"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    raw_pages = corpus_dir / "raw_pages.jsonl"
    clean_pages = corpus_dir / "clean_pages.jsonl"
    chunks = corpus_dir / "chunks.jsonl"
    train = lm_dir / "train_lm.jsonl"
    validation = lm_dir / "validation_lm.jsonl"
    test = lm_dir / "test_lm.jsonl"
    manifest_path = metadata_dir / "processed_pdfs_manifest.json"
    incremental_dir = artifacts_dir / "_incremental"

    pdfs = discover_pdfs(args.input_dir, args.recursive)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {args.input_dir}")

    manifest = load_manifest(manifest_path, pdfs)
    to_process = changed_pdfs(pdfs, manifest, force=args.force)
    if not to_process:
        write_manifest(manifest_path, [], manifest)
        print("No new or changed PDFs detected. Nothing to append.")
        print(f"Tracked PDFs: {len(manifest)}")
        return

    if incremental_dir.exists():
        shutil.rmtree(incremental_dir)
    incremental_dir.mkdir(parents=True, exist_ok=True)

    new_raw = incremental_dir / "raw_pages_new.jsonl"
    new_inventory = incremental_dir / "inventory_new.json"
    new_clean = incremental_dir / "clean_pages_new.jsonl"
    new_cleaning_report = incremental_dir / "cleaning_report_new.json"
    new_chunks = incremental_dir / "chunks_new.jsonl"
    new_train = incremental_dir / "train_new.jsonl"
    new_validation = incremental_dir / "validation_new.jsonl"
    new_test = incremental_dir / "test_new.jsonl"
    new_build_report = incremental_dir / "build_report_new.json"

    pdf_args: List[str] = []
    for pdf in to_process:
        pdf_args.extend(["--pdf", str(pdf)])

    run_step(
        "Extract New Or Changed PDFs",
        [
            sys.executable,
            "scripts/extract_pdfs.py",
            "--output",
            str(new_raw),
            "--inventory-output",
            str(new_inventory),
            *pdf_args,
        ],
    )
    run_step(
        "Clean New Pages",
        [
            sys.executable,
            "scripts/clean_text.py",
            "--input",
            str(new_raw),
            "--output",
            str(new_clean),
            "--report-output",
            str(new_cleaning_report),
            "--inventory",
            str(new_inventory),
        ],
    )
    build_command = [
            sys.executable,
            "scripts/build_lm_dataset.py",
            "--input",
            str(new_clean),
            "--inventory",
            str(new_inventory),
            "--chunks-output",
            str(new_chunks),
            "--train-output",
            str(new_train),
            "--validation-output",
            str(new_validation),
            "--test-output",
            str(new_test),
            "--build-report-output",
            str(new_build_report),
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
    run_step("Build New Chunks", build_command)

    processed_paths = {source_path_for(pdf) for pdf in to_process}
    processed_pdfs = {pdf.name for pdf in to_process}

    merged_raw = remove_sources(read_jsonl_if_exists(raw_pages), processed_paths, processed_pdfs)
    merged_raw.extend(read_jsonl(new_raw))
    write_jsonl(raw_pages, merged_raw)

    merged_clean = remove_sources(read_jsonl_if_exists(clean_pages), processed_paths, processed_pdfs)
    merged_clean.extend(read_jsonl(new_clean))
    write_jsonl(clean_pages, merged_clean)

    existing_chunks = remove_sources(read_jsonl_if_exists(chunks), processed_paths, processed_pdfs)
    combined_chunks = append_new_chunks(existing_chunks, read_jsonl(new_chunks))
    write_jsonl(chunks, combined_chunks)

    enriched_combined_chunks = enrich_chunks(combined_chunks)
    exportable_chunks = filter_lm_chunks(enriched_combined_chunks)
    new_train_chunks, new_validation_chunks, new_test_chunks = split_train_validation_test_by_document(
        exportable_chunks,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by="doc_type",
    )
    write_jsonl(chunks, enriched_combined_chunks)
    write_jsonl(train, [to_lm_record({**chunk, "split": "train"}) for chunk in new_train_chunks])
    write_jsonl(validation, [to_lm_record({**chunk, "split": "validation"}) for chunk in new_validation_chunks])
    write_jsonl(test, [to_lm_record({**chunk, "split": "test"}) for chunk in new_test_chunks])

    write_manifest(manifest_path, to_process, manifest)

    run_step(
        "Audit Combined Dataset",
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
            "--output",
            str(reports_dir / "audit_report.json"),
            "--samples-per-pdf",
            str(args.samples_per_pdf),
            "--seed",
            str(args.seed),
        ],
    )

    print("\nIncremental pipeline complete.", flush=True)
    print(f"PDFs discovered: {len(pdfs)}", flush=True)
    print(f"PDFs processed this run: {len(to_process)}", flush=True)
    print(f"Total chunks: {len(combined_chunks)}", flush=True)
    print(f"Train JSONL: {train}", flush=True)
    print(f"Validation JSONL: {validation}", flush=True)
    print(f"Test JSONL: {test}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    if not args.keep_temp and incremental_dir.exists():
        shutil.rmtree(incremental_dir)


if __name__ == "__main__":
    main()
