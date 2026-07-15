"""Canonical benchmark dataset and retrieval-corpus loaders."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree


DatasetMode = Literal["sample10", "maternaqa_test"]
XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class BenchmarkSample:
    """Normalized single-turn benchmark sample."""

    qa_id: str
    question: str
    reference: str
    source_context: str = ""
    reference_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusChunk:
    """Retrievable corpus unit."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference)
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_reference}")
    value = 0
    for letter in letters.group(0):
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = {"x": XLSX_MAIN_NS}
    return ["".join(node.text or "" for node in item.findall(".//x:t", namespace)) for item in root]


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    namespace = {"x": XLSX_MAIN_NS, "r": XLSX_REL_NS}
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id: str | None = None
    for sheet in workbook.findall("x:sheets/x:sheet", namespace):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{{{XLSX_REL_NS}}}id")
            break
    if relationship_id is None:
        raise ValueError(f"XLSX sheet not found: {sheet_name}")

    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relationship.get("Id") == relationship_id:
            target = str(relationship.get("Target") or "").lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError(f"XLSX relationship not found for sheet: {sheet_name}")


def _read_xlsx_records(path: Path, sheet_name: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"XLSX dataset not found: {path}")

    namespace = {"x": XLSX_MAIN_NS}
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        worksheet = ElementTree.fromstring(archive.read(_worksheet_path(archive, sheet_name)))

    rows: list[list[str]] = []
    for row in worksheet.findall("x:sheetData/x:row", namespace):
        values: dict[int, str] = {}
        for cell in row.findall("x:c", namespace):
            index = _column_index(str(cell.get("r") or ""))
            cell_type = cell.get("t")
            value_node = cell.find("x:v", namespace)
            raw = value_node.text if value_node is not None and value_node.text else ""
            if cell_type == "s" and raw:
                value = shared[int(raw)]
            elif cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//x:t", namespace))
            else:
                value = raw
            values[index] = value.strip()
        width = max(values, default=-1) + 1
        rows.append([values.get(index, "") for index in range(width)])

    if not rows:
        return []
    headers = rows[0]
    return [
        {header: row[index] if index < len(row) else "" for index, header in enumerate(headers)}
        for row in rows[1:]
    ]


def load_sample10(path: Path) -> list[BenchmarkSample]:
    """Load the canonical ten question/reference pairs from DATA_GT."""

    records = _read_xlsx_records(path, "DATA_GT")
    required = {"question", "ground_truth"}
    if records and not required.issubset(records[0]):
        raise ValueError(f"DATA_GT must contain columns: {sorted(required)}")
    if len(records) != 10:
        raise ValueError(f"sample10 must contain exactly 10 rows, found {len(records)}")

    samples = []
    for index, row in enumerate(records, start=1):
        question = row["question"].strip()
        reference = row["ground_truth"].strip()
        if not question or not reference:
            raise ValueError(f"sample10 row {index} has an empty question or ground_truth")
        samples.append(
            BenchmarkSample(
                qa_id=f"sample10_{index:03d}",
                question=question,
                reference=reference,
                metadata={"dataset_mode": "sample10", "row_number": index + 1},
            )
        )
    return samples


def load_maternaqa_test(path: Path) -> list[BenchmarkSample]:
    """Load the canonical 328-row flat MaternaQA test split."""

    if not path.exists():
        raise FileNotFoundError(f"MaternaQA test split not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "qa_id",
        "pregunta",
        "respuesta",
        "contexto_fuente",
        "chunk_id",
        "source_pdf",
        "pages",
        "section",
    }
    samples: list[BenchmarkSample] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"MaternaQA row {line_number} is missing: {sorted(missing)}")
        qa_id = str(row["qa_id"])
        if qa_id in seen:
            raise ValueError(f"Duplicate MaternaQA qa_id: {qa_id}")
        seen.add(qa_id)
        samples.append(
            BenchmarkSample(
                qa_id=qa_id,
                question=str(row["pregunta"]).strip(),
                reference=str(row["respuesta"]).strip(),
                source_context=str(row["contexto_fuente"]).strip(),
                reference_chunk_id=str(row["chunk_id"]),
                metadata={
                    "chunk_id": str(row["chunk_id"]),
                    "source_pdf": row["source_pdf"],
                    "pages": row["pages"],
                    "section": row["section"],
                },
            )
        )
    if len(samples) != 328:
        raise ValueError(f"maternaqa_test must contain exactly 328 rows, found {len(samples)}")
    return samples


def load_dataset(mode: DatasetMode, sample10_path: Path, maternaqa_path: Path) -> list[BenchmarkSample]:
    if mode == "sample10":
        return load_sample10(sample10_path)
    if mode == "maternaqa_test":
        return load_maternaqa_test(maternaqa_path)
    raise ValueError(f"Unsupported dataset mode: {mode}")


def load_corpus(path: Path) -> list[CorpusChunk]:
    """Load retrieval chunks without deriving content from evaluation rows."""

    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus not found: {path}")
    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "").strip()
        if not chunk_id or not text:
            raise ValueError(f"Corpus row {line_number} requires chunk_id and text")
        if chunk_id in seen:
            raise ValueError(f"Duplicate corpus chunk_id: {chunk_id}")
        seen.add(chunk_id)
        metadata = {key: value for key, value in row.items() if key != "text"}
        chunks.append(CorpusChunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks


def validate_reference_chunks(samples: list[BenchmarkSample], corpus: list[CorpusChunk]) -> None:
    """Prove every canonical test reference chunk is available for retrieval."""

    references = {sample.reference_chunk_id for sample in samples if sample.reference_chunk_id}
    if references and len(references) != 108:
        raise ValueError(f"maternaqa_test must reference exactly 108 chunks, found {len(references)}")
    corpus_ids = {chunk.chunk_id for chunk in corpus}
    missing = sorted(references - corpus_ids)
    if missing:
        raise ValueError(f"Corpus is missing {len(missing)} reference chunk IDs: {missing[:10]}")

