"""Generation usage and observed wall-clock latency accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GenerationMeasurement:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    generation_latency_seconds: float
    output_tokens_per_second: float

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

    def to_dict(self) -> dict[str, int | float]:
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
