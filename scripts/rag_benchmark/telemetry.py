"""Generation usage and observed wall-clock latency accounting."""

from __future__ import annotations

import math
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationMeasurement:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_latency_seconds: float
    output_tokens_per_second: float
    prompt_preparation_latency_seconds: float | None = None
    model_generation_latency_seconds: float | None = None

    @classmethod
    def from_counts(cls, input_tokens: int, output_tokens: int, latency_seconds: float) -> "GenerationMeasurement":
        """Build measured telemetry; rate is observed output / wall time, not local throughput."""
        total_tokens = input_tokens + output_tokens
        rate = output_tokens / latency_seconds if latency_seconds > 0 else 0.0
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            generation_latency_seconds=latency_seconds,
            output_tokens_per_second=rate,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def combine_system_measurements(
    answer: GenerationMeasurement,
    retrieval_latency_seconds: float,
    hypothetical: GenerationMeasurement | None = None,
    *,
    hypothetical_cache_hit: bool = False,
) -> dict[str, int | float]:
    """Aggregate current-run answer and optional current-run HyDE work into system fields."""

    stages = [answer] + ([hypothetical] if hypothetical is not None and not hypothetical_cache_hit else [])
    input_tokens = sum(stage.input_tokens for stage in stages)
    output_tokens = sum(stage.output_tokens for stage in stages)
    generation_latency = sum(stage.generation_latency_seconds for stage in stages)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "retrieval_latency_seconds": retrieval_latency_seconds,
        "generation_latency_seconds": generation_latency,
        "end_to_end_latency_seconds": retrieval_latency_seconds + generation_latency,
        "output_tokens_per_second": output_tokens / generation_latency if generation_latency > 0 else 0.0,
    }


class ResourceSampler:
    """Best-effort resource sampling with optional single-device NVML telemetry."""

    def __init__(
        self,
        interval_seconds: float = 0.5,
        gpu_device_index: int | None = None,
        gpu_device_identity: tuple[str, int] | None = None,
    ) -> None:
        self.enabled = interval_seconds > 0
        self.interval_seconds = max(0.05, interval_seconds)
        self.gpu_device_index = gpu_device_index
        self._nvml_device_index: int | None = None
        self._nvml_dll_handles: list[Any] = []
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = self._process = self._nvml = None
        self._nvml_initialized = False
        self._cpu_baseline_pending = False
        self._cpu_baseline_time: float | None = None
        if self.enabled:
            try:
                import psutil
                self._psutil = psutil
            except Exception:
                self._psutil = None
            if self._psutil is not None:
                try:
                    self._process = self._psutil.Process()
                except Exception:
                    self._process = None
                self._cpu_baseline_pending = True
            if gpu_device_index is not None:
                pynvml = None
                try:
                    if platform.system() == "Windows":
                        self._nvml_dll_handles = _open_windows_nvml_dll_directories()
                    import pynvml
                    self._nvml = pynvml
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                    self._nvml_device_index = resolve_nvml_device_index(
                        gpu_device_index,
                        pynvml,
                        os.environ.get("CUDA_VISIBLE_DEVICES"),
                        gpu_device_identity,
                    )
                except Exception:
                    if pynvml is not None:
                        try:
                            pynvml.nvmlShutdown()
                        except Exception:
                            pass
                    self._nvml = None
                    self._close_nvml_dll_directories()

    def _sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {"_sample_time_monotonic": time.monotonic()}
        def record(name: str, read: Any) -> None:
            try:
                result[name] = read()
            except Exception:
                result[name] = None

        psutil = self._psutil
        if psutil is not None:
            sample_time = result["_sample_time_monotonic"]
            if self._cpu_baseline_pending:
                record("system_cpu_percent", lambda: psutil.cpu_percent(interval=None))
                if self._process is not None:
                    record("process_cpu_percent", lambda: self._process.cpu_percent(interval=None))
                else:
                    result["process_cpu_percent"] = None
                self._cpu_baseline_pending = False
                self._cpu_baseline_time = sample_time
                result["system_cpu_percent"] = None
                result["process_cpu_percent"] = None
            elif self._cpu_baseline_time is not None and sample_time - self._cpu_baseline_time >= 0.1:
                record("process_cpu_percent", lambda: self._process.cpu_percent(interval=None))
                record("system_cpu_percent", lambda: psutil.cpu_percent(interval=None))
                self._cpu_baseline_time = sample_time
            else:
                result["process_cpu_percent"] = None
                result["system_cpu_percent"] = None
            record("process_rss_bytes", lambda: self._process.memory_info().rss)
            record("process_thread_count", lambda: self._process.num_threads())
            record("system_available_ram_bytes", lambda: psutil.virtual_memory().available)
            record("cpu_physical_core_count", lambda: psutil.cpu_count(logical=False))
            record("cpu_logical_core_count", lambda: psutil.cpu_count(logical=True))

            def cpu_temperature() -> float | None:
                if platform.system() == "Windows":
                    return None
                temperatures = psutil.sensors_temperatures()
                values = [entry.current for label, entries in temperatures.items() if any(token in label.lower() for token in ("cpu", "coretemp", "k10temp")) for entry in entries if entry.current is not None]
                return max(values) if values else None

            record("cpu_temperature_celsius", cpu_temperature)

        if self._nvml is not None and self._nvml_device_index is not None:
            nvml = self._nvml
            try:
                device = nvml.nvmlDeviceGetHandleByIndex(self._nvml_device_index)
            except Exception:
                return result
            record("gpu_utilization_percent", lambda: nvml.nvmlDeviceGetUtilizationRates(device).gpu)
            record("gpu_temperature_celsius", lambda: nvml.nvmlDeviceGetTemperature(device, nvml.NVML_TEMPERATURE_GPU))
            record("gpu_power_watts", lambda: nvml.nvmlDeviceGetPowerUsage(device) / 1000)
            record("gpu_sm_clock_mhz", lambda: nvml.nvmlDeviceGetClockInfo(device, nvml.NVML_CLOCK_SM))

            try:
                memory_info = nvml.nvmlDeviceGetMemoryInfo(device)
            except Exception:
                memory_info = None
            result["gpu_memory_used_bytes"] = memory_info.used if memory_info is not None else None
            result["gpu_memory_total_bytes"] = memory_info.total if memory_info is not None else None
        return result

    def _run(self) -> None:
        try:
            self.samples.append(self._sample())
        except Exception:
            pass
        finally:
            self._started.set()
        while not self._stop.wait(self.interval_seconds):
            try:
                self.samples.append(self._sample())
            except Exception:
                pass
        try:
            self.samples.append(self._sample())
        except Exception:
            pass

    def __enter__(self) -> "ResourceSampler":
        if not self.enabled:
            return self
        try:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._started.wait()
        except Exception:
            self._stop.set()
            if self._thread is not None and self._thread.ident is None:
                self._thread = None
            self._shutdown_nvml()
            self._started.set()
        return self

    def __exit__(self, *_: object) -> None:
        try:
            if self.enabled:
                self._stop.set()
                if self._thread:
                    self._thread.join()
        finally:
            self._shutdown_nvml()

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
            self._nvml = None
        self._close_nvml_dll_directories()

    def _close_nvml_dll_directories(self) -> None:
        for handle in reversed(self._nvml_dll_handles):
            try:
                handle.close()
            except Exception:
                pass
        self._nvml_dll_handles.clear()

    @staticmethod
    def integrate_gpu_energy(samples: list[dict[str, Any]]) -> float | None:
        """Integrate whole-device NVML power with at least one second of coverage."""
        if len(samples) < 2:
            return None
        points: list[tuple[float, float]] = []
        for sample in samples:
            timestamp = sample.get("_sample_time_monotonic")
            power = sample.get("gpu_power_watts")
            if not isinstance(timestamp, (int, float)) or not isinstance(power, (int, float)):
                return None
            try:
                timestamp_value, power_value = float(timestamp), float(power)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(timestamp_value) or not math.isfinite(power_value) or power_value < 0:
                return None
            points.append((timestamp_value, power_value))
        energy = 0.0
        for (start, first_power), (end, second_power) in zip(points, points[1:]):
            if end <= start:
                return None
            energy += (first_power + second_power) * 0.5 * (end - start)
            if not math.isfinite(energy):
                return None
        if points[-1][0] - points[0][0] < 1.0:
            return None
        return energy

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"sample_count": 0, "sample_interval_seconds": 0, "sampling_available": False, "gpu_sampling_available": False, "gpu_device_index": self.gpu_device_index, "gpu_energy_joules_estimate": None}
        keys = (set().union(*(sample.keys() for sample in self.samples)) - {"_sample_time_monotonic"}) if self.samples else set()
        gpu_sampled = any(any(isinstance(sample.get(key), (int, float)) for key in keys if key.startswith("gpu_")) for sample in self.samples)
        result: dict[str, Any] = {"sample_count": len(self.samples), "sample_interval_seconds": self.interval_seconds, "sampling_available": any(isinstance(sample.get(key), (int, float)) for sample in self.samples for key in keys), "gpu_sampling_available": gpu_sampled, "gpu_device_index": self.gpu_device_index}
        for key in sorted(keys):
            values = [sample[key] for sample in self.samples if isinstance(sample.get(key), (int, float))]
            result[f"{key}_mean"] = sum(values) / len(values) if values else None
            result[f"{key}_peak"] = max(values) if values else None
        result["gpu_energy_joules_estimate"] = self.integrate_gpu_energy(self.samples) if self.gpu_device_index is not None else None
        return result


def resolve_cuda_device_index(model: Any) -> int | None:
    """Resolve one exact CUDA device; do not guess for CPU, sharded, or unknown models."""
    try:
        return _resolve_cuda_device_index(model)
    except Exception:
        return None


def resolve_nvml_device_index(
    cuda_device_index: int,
    nvml: Any,
    cuda_visible_devices: str | None,
    cuda_device_identity: tuple[str, int] | None = None,
) -> int | None:
    """Resolve NVML via explicit visibility or unique name/memory identity."""
    def normalize_uuid(value: Any) -> str:
        raw = getattr(value, "bytes", value)
        if isinstance(raw, bytes):
            return raw.hex()
        return str(raw).lower().removeprefix("gpu-").replace("-", "")

    try:
        if not isinstance(cuda_device_index, int) or cuda_device_index < 0:
            return None
        nvml_count = nvml.nvmlDeviceGetCount()
        if nvml_count <= 0:
            return None
        if cuda_visible_devices is None:
            if cuda_device_index >= nvml_count or cuda_device_identity is None:
                return None
            name, total_memory = cuda_device_identity
            if not isinstance(name, str) or not name.strip() or not isinstance(total_memory, int) or total_memory <= 0:
                return None
            name = " ".join(name.casefold().split())
            matches = []
            for index in range(nvml_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                device_name = nvml.nvmlDeviceGetName(handle)
                if isinstance(device_name, bytes):
                    device_name = device_name.decode("utf-8")
                device_memory = nvml.nvmlDeviceGetMemoryInfo(handle).total
                if " ".join(str(device_name).casefold().split()) == name and device_memory == total_memory:
                    matches.append(index)
            return matches[0] if len(matches) == 1 else None

        identifiers = cuda_visible_devices.split(",")
        if not identifiers or any(not item for item in identifiers):
            return None
        if cuda_device_index >= len(identifiers):
            return None
        if all(item.strip().isdecimal() for item in identifiers):
            identifiers = [item.strip() for item in identifiers]
            if len(set(identifiers)) != len(identifiers):
                return None
            physical_index = int(identifiers[cuda_device_index])
            return physical_index if physical_index < nvml_count else None

        if not all(item.startswith("GPU-") for item in identifiers) or len(set(identifiers)) != len(identifiers):
            return None
        device_uuids = [
            normalize_uuid(nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(index)))
            for index in range(nvml_count)
        ]
        resolved = []
        for identifier in identifiers:
            candidate = normalize_uuid(identifier)
            if not candidate or any(character not in "0123456789abcdef" for character in candidate):
                return None
            matches = [index for index, device_uuid in enumerate(device_uuids) if device_uuid.startswith(candidate)]
            if len(matches) != 1:
                return None
            resolved.append(matches[0])
        if len(set(resolved)) != len(resolved):
            return None
        return resolved[cuda_device_index]
    except Exception:
        return None


def resolve_cuda_device_identity(torch_module: Any, cuda_device_index: int) -> tuple[str, int] | None:
    """Read public CUDA properties used to identify an unmasked device safely."""
    try:
        properties = torch_module.cuda.get_device_properties(cuda_device_index)
        name = properties.name
        if not isinstance(name, str):
            return None
        name = name.strip()
        total_memory = int(properties.total_memory)
        return (name, total_memory) if name and total_memory > 0 else None
    except Exception:
        return None


def _open_windows_nvml_dll_directories() -> list[Any]:
    """Keep Windows DLL search directories open while pynvml uses NVML."""
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = (
        program_files / "NVIDIA Corporation" / "NVSMI",
        system_root / "System32",
    )
    handles = []
    for directory in candidates:
        if directory.is_dir():
            try:
                handles.append(os.add_dll_directory(str(directory)))
            except (OSError, AttributeError):
                pass
    return handles


def _resolve_cuda_device_index(model: Any) -> int | None:
    if model is None:
        return None
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        indices = set()
        for device in device_map.values():
            if isinstance(device, int):
                if device >= 0:
                    indices.add(device)
            elif isinstance(device, str) and device.startswith("cuda"):
                _, separator, index = device.partition(":")
                try:
                    resolved_index = int(index) if separator else 0
                    if resolved_index < 0:
                        return None
                    indices.add(resolved_index)
                except ValueError:
                    return None
            elif getattr(device, "type", None) == "cuda":
                device_index = getattr(device, "index", None)
                resolved_index = int(device_index) if device_index is not None else 0
                if resolved_index < 0:
                    return None
                indices.add(resolved_index)
        if len(indices) > 1:
            return None
        if indices:
            return next(iter(indices))
        return None
    device = getattr(model, "device", None)
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None) if device_type == "cuda" else None
    if device_type == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
        if device_index is not None:
            return int(device_index)
        if isinstance(device, str) and ":" in device:
            try:
                return int(device.split(":", 1)[1])
            except ValueError:
                return None
        try:
            import torch
            return int(torch.cuda.current_device())
        except Exception:
            return None
    return None
