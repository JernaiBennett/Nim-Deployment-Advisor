"""
Deployment sizing logic for NVIDIA NIM.

This is deliberately NOT an LLM call — it's the same kind of back-of-envelope
math a Solutions Architect does on a whiteboard with a customer: given a model
size and available hardware, does it fit, at what precision, and on how many
GPUs.

The formulas here are documented rules of thumb (see docs pulled in the
companion nim-support-agent project's 01_getting_started.md and
02_model_profiles.md), not exact benchmarked numbers — this tool answers
"will this fit and what should I request," not "here's your exact tokens/sec."
That distinction is called out explicitly in the README and in this file's
docstrings so nobody mistakes a sizing estimate for a performance guarantee.
"""
import json
import math
from pathlib import Path
from dataclasses import dataclass, field

DATA_PATH = Path(__file__).parent.parent / "data" / "gpu_specs.json"

# Bytes needed per parameter, by precision. This is the standard rule of thumb
# NVIDIA's own NIM docs use ("# model parameters * 2 GB" for fp16/bf16).
BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "awq": 0.5,
}

# Flat overhead for OS + Docker + KV-cache buffer, per the same doc's guidance
# (5-10 GB OS + 16 GB Docker, rounded to a single conservative constant here).
OVERHEAD_GB = 24.0

# NIM's own profile-selection preference order for backend and precision,
# used to justify the "recommended" choice rather than just picking one.
BACKEND_PREFERENCE = ["tensorrt_llm", "vllm", "sglang"]
PRECISION_PREFERENCE = ["fp8", "int8", "fp16", "int4"]


@dataclass
class SizingResult:
    fits: bool
    precision: str
    gpu_count: int
    total_memory_needed_gb: float
    memory_per_gpu_gb: float
    optimization_target: str
    reasoning: list[str] = field(default_factory=list)


def load_gpu_specs() -> dict[str, int]:
    raw = json.loads(DATA_PATH.read_text())
    return {g["name"]: g["vram_gb"] for g in raw["gpus"]}


def next_power_of_two(n: int) -> int:
    """Tensor parallelism is conventionally split across a power-of-2 number
    of GPUs (1, 2, 4, 8...), not an arbitrary count, so round up to that."""
    if n <= 1:
        return 1
    return 2 ** math.ceil(math.log2(n))


def memory_required_gb(params_billions: float, precision: str) -> float:
    bytes_per_param = BYTES_PER_PARAM[precision]
    return params_billions * bytes_per_param + OVERHEAD_GB


def choose_precision(accuracy_critical: bool, latency_sensitive: bool) -> str:
    """
    Mirrors NIM's own profile-selection preference order (fp8 > int8 > fp16),
    but overridden by explicit customer constraints:
    - accuracy_critical customers get fp16 (no quantization) by default
    - everyone else defaults to fp8, which NIM itself prefers when available
    """
    if accuracy_critical:
        return "fp16"
    return "fp8"


def recommend_deployment(
    params_billions: float,
    gpu_vram_gb: int,
    latency_sensitive: bool = False,
    accuracy_critical: bool = False,
    available_gpu_count: int | None = None,
) -> SizingResult:
    reasoning = []

    precision = choose_precision(accuracy_critical, latency_sensitive)
    reasoning.append(
        f"Selected {precision} precision "
        + ("because accuracy was flagged as critical (no quantization)."
           if accuracy_critical else
           "as the default — NIM itself prefers fp8 over fp16 when available for its memory/speed benefit.")
    )

    total_mem = memory_required_gb(params_billions, precision)
    reasoning.append(
        f"Estimated total memory: {params_billions}B params x "
        f"{BYTES_PER_PARAM[precision]} GB/param + {OVERHEAD_GB} GB overhead "
        f"= {total_mem:.1f} GB."
    )

    raw_gpu_count = math.ceil(total_mem / gpu_vram_gb)
    gpu_count = next_power_of_two(raw_gpu_count)
    reasoning.append(
        f"{total_mem:.1f} GB / {gpu_vram_gb} GB per GPU = {raw_gpu_count} GPU(s) minimum, "
        f"rounded up to {gpu_count} for standard tensor-parallel splitting."
    )

    mem_per_gpu = total_mem / gpu_count
    fits = mem_per_gpu <= gpu_vram_gb

    if not fits:
        reasoning.append(
            f"Even at {gpu_count} GPUs this doesn't comfortably fit "
            f"({mem_per_gpu:.1f} GB needed per GPU vs {gpu_vram_gb} GB available). "
            f"Consider a smaller model, a lower precision (int4/awq), or more GPUs."
        )

    # Check the ideal GPU count against what the customer actually has.
    # We keep `gpu_count` as the honest ideal number rather than silently
    # capping it — `fits` and the reasoning trail carry the bad news instead.
    if available_gpu_count is not None and fits and gpu_count > available_gpu_count:
        fits = False
        reasoning.append(
            f"Customer has {available_gpu_count}x GPU(s) available, but this deployment "
            f"needs {gpu_count} to fit at {precision}. "
            + (
                "Even the most aggressive quantization (int4/awq) may not close this gap — "
                "reconsider the model size or add hardware."
                if precision in ("int4", "awq")
                else "Consider a more aggressive precision (fp8 → int4/awq) or fewer available GPUs isn't workable without a smaller model."
            )
        )

    optimization_target = "latency" if latency_sensitive else "throughput"
    reasoning.append(
        f"Optimization target set to '{optimization_target}' "
        + ("to minimize time-to-first-token and inter-token latency."
           if latency_sensitive else
           "to maximize total tokens served per GPU.")
    )

    return SizingResult(
        fits=fits,
        precision=precision,
        gpu_count=gpu_count,
        total_memory_needed_gb=round(total_mem, 1),
        memory_per_gpu_gb=round(mem_per_gpu, 1),
        optimization_target=optimization_target,
        reasoning=reasoning,
    )


if __name__ == "__main__":
    specs = load_gpu_specs()

    scenarios = [
        # (label, params_billions, gpu_name, latency_sensitive, accuracy_critical, available_gpu_count)
        ("Llama 70B, latency-critical, 2x A100-80GB", 70, "A100-80GB", True, False, 2),
        ("Llama 8B on a single L4", 8, "L4", False, False, 1),
        ("Llama 70B, accuracy-critical, single L40S", 70, "L40S", False, True, 1),
    ]

    for label, params, gpu_name, latency, accuracy, available in scenarios:
        print(f"\n=== {label} ===")
        result = recommend_deployment(params, specs[gpu_name], latency, accuracy, available)
        print(f"Fits: {result.fits} | Precision: {result.precision} | "
              f"GPUs needed: {result.gpu_count} | Target: {result.optimization_target}")
        for r in result.reasoning:
            print(f"  - {r}")
