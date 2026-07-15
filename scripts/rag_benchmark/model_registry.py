"""Answer-model registry and local/remote adapter resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    base_model_id: str
    adapter_id: str | None = None
    local_adapter_candidates: tuple[Path, ...] = ()

    @property
    def is_adapter(self) -> bool:
        return self.adapter_id is not None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gemma4_base": ModelSpec(
        key="gemma4_base",
        model_id="google/gemma-4-E2B-it",
        base_model_id="google/gemma-4-E2B-it",
    ),
    "gemma4_qlora": ModelSpec(
        key="gemma4_qlora",
        model_id="iue-edu/MaternaCare-ES-gemma4-qlora",
        base_model_id="google/gemma-4-E2B-it",
        adapter_id="iue-edu/MaternaCare-ES-gemma4-qlora",
        local_adapter_candidates=(Path("outputs/gemma4-grounded"), Path("outputs/gemma4-qlora")),
    ),
    "medgemma_base": ModelSpec(
        key="medgemma_base",
        model_id="google/medgemma-1.5-4b-it",
        base_model_id="google/medgemma-1.5-4b-it",
    ),
    "medgemma_qlora": ModelSpec(
        key="medgemma_qlora",
        model_id="iue-edu/MaternaCare-ES-medgemma-qlora",
        base_model_id="google/medgemma-1.5-4b-it",
        adapter_id="iue-edu/MaternaCare-ES-medgemma-qlora",
        local_adapter_candidates=(Path("outputs/medgemma-grounded"), Path("outputs/medgemma-qlora")),
    ),
}


def _is_adapter_directory(path: Path) -> bool:
    has_config = (path / "adapter_config.json").is_file()
    has_weights = (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    return has_config and has_weights


def resolve_adapter_source(spec: ModelSpec, explicit_path: Path | None = None) -> str | None:
    """Use an explicit valid local adapter; otherwise retain the registry HF ID."""

    if not spec.is_adapter:
        return None
    if explicit_path is not None:
        if not _is_adapter_directory(explicit_path):
            raise FileNotFoundError(f"Adapter path has no config and weights: {explicit_path}")
        return str(explicit_path.resolve())
    return spec.adapter_id
