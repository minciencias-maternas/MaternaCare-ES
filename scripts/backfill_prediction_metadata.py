from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENARIOS = {
    "gemma4-grounded": {
        "model_role": "grounded",
        "dataset_variant": "sft_grounded",
    },
    "medgemma-grounded": {
        "model_role": "grounded",
        "dataset_variant": "sft_grounded",
    },
}

PROVENANCE_FIELDS = ("model_role", "model_name", "adapter_dir", "dataset_variant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill model provenance fields in prediction and evaluation JSONL files."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing model output subdirectories.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(DEFAULT_SCENARIOS),
        help="Scenario to update. Can be passed more than once. Defaults to all known scenarios.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files.",
    )
    return parser.parse_args()


def read_base_model_name(adapter_config: Path) -> str:
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    model_name = data.get("base_model_name_or_path")
    if not model_name:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_config}")
    return str(model_name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def missing_or_empty(value: Any) -> bool:
    return value is None or value == ""


def fill_row(row: dict[str, Any], values: dict[str, str]) -> int:
    changed = 0
    for field in PROVENANCE_FIELDS:
        if missing_or_empty(row.get(field)):
            row[field] = values[field]
            changed += 1
    return changed


def update_jsonl(path: Path, values: dict[str, str], dry_run: bool) -> dict[str, int]:
    rows = read_jsonl(path)
    changed_rows = 0
    changed_fields = 0
    for row in rows:
        row_changes = fill_row(row, values)
        if row_changes:
            changed_rows += 1
            changed_fields += row_changes
    if changed_fields and not dry_run:
        write_jsonl(path, rows)
    return {"rows": len(rows), "changed_rows": changed_rows, "changed_fields": changed_fields}


def scenario_values(outputs_dir: Path, scenario: str) -> dict[str, str]:
    scenario_dir = outputs_dir / scenario
    config = DEFAULT_SCENARIOS[scenario]
    return {
        "model_role": config["model_role"],
        "model_name": read_base_model_name(scenario_dir / "adapter_config.json"),
        "adapter_dir": str(scenario_dir),
        "dataset_variant": config["dataset_variant"],
    }


def update_scenario(outputs_dir: Path, scenario: str, dry_run: bool) -> dict[str, dict[str, int]]:
    scenario_dir = outputs_dir / scenario
    values = scenario_values(outputs_dir, scenario)
    files = {
        "predictions": scenario_dir / "test_predictions.jsonl",
        "evaluation": scenario_dir / "test_eval.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")
    return {name: update_jsonl(path, values, dry_run) for name, path in files.items()}


def main() -> None:
    args = parse_args()
    scenarios = args.scenario or list(DEFAULT_SCENARIOS)
    report = {
        scenario: update_scenario(args.outputs_dir, scenario, args.dry_run)
        for scenario in scenarios
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
