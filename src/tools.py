"""
Wraps sizing.py's deterministic calculator as an LLM-callable tool, using the
same TOOLS / DISPATCH pattern as the nim-support-agent project's tools.py.
"""
from dataclasses import asdict
from sizing import recommend_deployment


def recommend_deployment_tool(
    params_billions: float,
    gpu_vram_gb: int,
    latency_sensitive: bool = False,
    accuracy_critical: bool = False,
    available_gpu_count: int | None = None,
) -> dict:
    """JSON-serializable wrapper around recommend_deployment for tool dispatch."""
    result = recommend_deployment(
        params_billions=params_billions,
        gpu_vram_gb=gpu_vram_gb,
        latency_sensitive=latency_sensitive,
        accuracy_critical=accuracy_critical,
        available_gpu_count=available_gpu_count,
    )
    return asdict(result)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_deployment_tool",
            "description": (
                "Calculates whether an LLM deployment fits on given NVIDIA GPU hardware, "
                "and recommends precision, GPU count, and optimization target. Use this "
                "any time a customer describes a model size and available hardware."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "params_billions": {
                        "type": "number",
                        "description": "Model size in billions of parameters, e.g. 70 for a 70B model.",
                    },
                    "gpu_vram_gb": {
                        "type": "integer",
                        "description": "VRAM in GB of a single GPU of the type the customer has, e.g. 80 for A100-80GB.",
                    },
                    "latency_sensitive": {
                        "type": "boolean",
                        "description": "True if the customer cares more about fast response time than total throughput.",
                    },
                    "accuracy_critical": {
                        "type": "boolean",
                        "description": "True if the customer cannot tolerate quantization-related accuracy loss.",
                    },
                    "available_gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs the customer actually has available, if known.",
                    },
                },
                "required": ["params_billions", "gpu_vram_gb"],
            },
        },
    },
]

DISPATCH = {
    "recommend_deployment_tool": recommend_deployment_tool,
}


if __name__ == "__main__":
    import json
    result = recommend_deployment_tool(
        params_billions=70,
        gpu_vram_gb=80,
        latency_sensitive=True,
        available_gpu_count=2,
    )
    print(json.dumps(result, indent=2))
