# ODD task: RAG benchmark operational telemetry

## Goal
Add reproducible operational, hardware, retrieval-identifier, and inference-completion diagnostics to the RAG benchmark while preserving existing serialized metrics and RAGAS behavior.

Windows is the primary experiment OS. This task records Windows-specific availability and safe device identity handling; physical Windows hardware verification remains pending.

## Verified current state
- `runner.py` persists resumable JSONL sample rows and summary JSON; legacy `operations` summary values are means despite their unqualified names.
- Generation counts are integer per-call values; current generation timing includes prompt preparation and generation.
- HyDE is prepared/cached before sample execution; cached work is not current-sample workload.
- RAGAS reports six metrics; reference chunk identifiers may be absent.

## Decisions
- Preserve all prior row fields, RAGAS fields, and summary keys; add explicit `operations_mean` aliases.
- Sample process/system resource data during answer inference and uncached local Hugging Face HyDE generation; cache hits and remote HyDE receive no current-run hardware attribution. NVML remains best effort and its binding is included in the CUDA requirements path.
- Use one-reference identifier-match retrieval diagnostics, not semantic judgments.
- Keep provider libraries optional and runtime guarded; do not add mandatory dependencies.
- Expose measured local prompt/model sub-stages while preserving existing generation timing semantics.
- Estimate approximate whole-device energy only from valid timestamped NVML power samples spanning at least one second, using trapezoidal integration; never attribute this estimate to the benchmark process.
- Map numeric selection by NVML ordinal and `GPU-` UUID selections by unique NVML UUID matching, following CUDA 2.5.0 visibility semantics. When visibility is unset, require a unique match on public CUDA device name plus total memory; never use the CUDA ordinal alone as physical identity. Reject malformed, missing, duplicate, or ambiguous mappings. Do not call PyTorch's private `_get_nvml_device_index` because its UUID helper loads Linux-only `libnvidia-ml.so.1`.
- On Windows, add NVIDIA `NVSMI` and `System32` directories to DLL search when present; retain DLL-directory handles until NVML shutdown. Missing DLLs or driver initialization continue to disable GPU metrics without failing the benchmark.
- Treat CPU temperature as unavailable on Windows; use psutil for CPU/RAM/process metrics. NVML metrics vary by device/driver; WDDM memory can be system-wide with a large baseline, Ampere+ power is a one-second average, and only SM clock is sampled.
- CPU utilization can be unavailable in very short runs because the initial baseline is discarded and readings require at least 0.1 seconds between samples.
- Do not call non-streaming timing TTFT.

## Scope
Benchmark telemetry implementation, focused tests, this task record, and the current technical document `docs/operational-metrics.md`. No unrelated files or commits.

## Acceptance criteria
- Existing RAGAS and operational output remains available and serialized compatibly.
- Integer sample values and explicit fractional summary means are distinguishable.
- Context sizes, retrieval count, optional resource snapshots, stage details, retrieval diagnostics, completion status, and runtime metadata have documented definitions and null behavior.
- Provider failures do not prevent benchmark execution.
- Resource sampling avoids optional provider initialization when disabled, captures start/final observations, joins before provider shutdown, and handles short operations.
- Error rows explicitly carry `completion_status: "error"`; successful rows retain completion/truncation semantics.
- Legacy HyDE cache records remain readable, and cached HyDE telemetry is not represented as current-run resource/energy data.
- Documentation matches emitted field names and limitations.

## Tasks
- [x] `RBT-1` Inspect benchmark orchestration, telemetry, generation, dependencies, and tests.
- [x] `RBT-2` Add compatible per-sample metrics and explicit mean aliases; preserve the legacy summary section.
- [x] `RBT-3` Add best-effort process/system/NVML resource sampling and sampling control.
- [x] `RBT-4` Add retrieval diagnostics, context sizes, measurable local stage/load timing, and conservative completion flags.
- [x] `RBT-5` Record runtime/provider metadata without adding required dependencies.
- [x] `RBT-6` Add focused tests for reference-ID rank diagnostics and missing-reference behavior.
- [x] `RBT-7` Document output schemas, formulas, installation, interpretation, and limitations.
- [x] `RBT-8` Run focused benchmark tests in a fresh action worker.
- [x] `RBT-9` Correct binary nDCG@k formula and test its distinction from MRR at rank 2.
- [x] `RBT-10` Harden optional resource sampling, immediate/final snapshots, provider shutdown, and deterministic sampler tests.
- [x] `RBT-11` Resolve one exact model CUDA device and add guarded timestamped whole-device energy estimates with explicit summary means.
- [x] `RBT-12` Sample uncached local Hugging Face HyDE generation and preserve cache-hit/current-run separation and old cache compatibility.
- [x] `RBT-13` Mark inference/serialization failures with `completion_status: "error"` and add a focused error-row test.
- [x] `RBT-14` Update the technical document for corrected formulas, energy, model/device limits, and HyDE sampling semantics.
- [x] `RBT-15` Make Windows the primary target, install telemetry providers via standard/CUDA requirements paths, and fail closed on CUDA-to-NVML identity ambiguity.
- [x] `RBT-16` Require at least one second of energy sampling coverage and align sampling CLI help with uncached local HyDE behavior.
- [x] `RBT-17` Require unique public CUDA name+memory identity when visibility is unset; map explicit numeric/UUID visibility through NVML, retain Windows DLL search handles through shutdown, and test fail-closed behavior.

## Verification
- `python -m unittest tests.test_rag_benchmark` — 33 tests passed in a fresh action worker.
- `git diff --check` — passed.
- `ruff format --check` was unavailable because Ruff is not installed in the environment.
- Native review status returned `next_transition.kind: stop` with `reason_code: rdd_disabled`; no review lifecycle operation was started.
- Windows physical smoke test: pending; no Windows hardware test is claimed.

## Windows smoke-test checklist
- [ ] Install `requirements-cuda.txt` on Windows with a supported NVIDIA driver.
- [ ] Run a short telemetry-enabled benchmark; verify graceful behavior when NVML/device metrics are unavailable.
- [ ] Confirm CPU temperature is unavailable, process/system metrics load, and each supported visibility form selects only its mapped physical GPU.
- [ ] Verify unset, reordered numeric (`CUDA_VISIBLE_DEVICES=2,0`), and UUID visibility mapping, plus fail-closed malformed/ambiguous cases.
- [ ] Confirm unset visibility maps through unique CUDA name+memory identity and fails closed for duplicate or unavailable identity.
- [ ] Confirm NVML DLL directory handles remain available until shutdown and close afterward.
- [ ] Check WDDM memory baseline, SM clock output, and the one-second minimum energy coverage behavior.
