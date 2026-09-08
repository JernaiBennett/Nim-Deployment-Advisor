"""
NIM Deployment Advisor.

Takes a customer's use case in plain language, lets the LLM extract the
relevant parameters and call recommend_deployment_tool, then writes a final
recommendation grounded in that tool's real math.

Same tool-calling round-trip pattern as nim-support-agent/src/agent.py:
  1. send the customer's message + available tools to the model
  2. if the model calls a tool, run the real Python function and feed the
     result back
  3. get a final answer that's grounded in that real calculation, not a guess
"""
import os
import sys
import json
import argparse
from pathlib import Path

from tools import TOOLS, DISPATCH
from sizing import load_gpu_specs

SYSTEM_PROMPT = """You are a Solutions Architect advisor for NVIDIA NIM deployments. \
A customer will describe their use case in plain language — the model they want to run, \
what hardware they have, and whether latency or throughput matters more.

Rules:
- Always call recommend_deployment_tool to get real sizing numbers before answering. \
Never estimate GPU memory or GPU count yourself — that's the tool's job.
- If the customer doesn't specify GPU VRAM directly but names a GPU model (e.g. "A100", \
"L40S"), use these known VRAM values: """ + json.dumps(load_gpu_specs()) + """
- If the tool reports fits=False, be direct about it and explain the real options \
(more GPUs, lower precision, smaller model) rather than softening the answer.
- Keep the final answer concise and concrete: state the recommended GPU count, precision, \
and optimization target explicitly, then the reasoning in a sentence or two.
"""


def get_client(base_url: str | None, api_key: str | None):
    from openai import OpenAI
    return OpenAI(
        base_url=base_url or "https://integrate.api.nvidia.com/v1",
        api_key=api_key or os.environ.get("NVIDIA_API_KEY", "not-used"),
    )


def run_advisor_turn(client, model: str, user_message: str, history: list) -> tuple[str, list]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1024,
        temperature=0.2,
    )
    choice = response.choices[0].message

    if choice.tool_calls:
        messages.append({"role": "assistant", "content": choice.content or "", "tool_calls": choice.tool_calls})
        for tool_call in choice.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = DISPATCH[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        followup = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        final_text = followup.choices[0].message.content
    else:
        final_text = choice.content

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_text},
    ]
    return final_text, new_history


def run_demo():
    """Runs the sizing tool directly, without a live API key, so the core
    logic is visible even without NVIDIA credentials."""
    from tools import recommend_deployment_tool

    print("=" * 70)
    print("DEMO MODE — no API key required. Showing the sizing tool directly.")
    print("=" * 70)

    use_case = "I need to deploy Llama 70B for a customer support chatbot. Latency matters. They have 2x A100-80GB."
    print(f"\nCustomer use case:\n{use_case}\n")

    print("-- What the LLM would extract and pass to the tool --")
    extracted = {"params_billions": 70, "gpu_vram_gb": 80, "latency_sensitive": True, "available_gpu_count": 2}
    print(json.dumps(extracted, indent=2))

    print("\n-- Real tool output (recommend_deployment_tool) --")
    result = recommend_deployment_tool(**extracted)
    print(json.dumps(result, indent=2))

    print("\n-- What the final answer would incorporate --")
    print("  A grounded recommendation stating the GPU count, precision, and optimization")
    print("  target from the tool's actual math above — not a guessed number.")
    print("\nTo run this end-to-end against a real model, set NVIDIA_API_KEY and run")
    print("without --demo. Get a free key at https://build.nvidia.com/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    if args.demo or not os.environ.get("NVIDIA_API_KEY"):
        run_demo()
        return

    client = get_client(args.base_url, os.environ.get("NVIDIA_API_KEY"))

    print("NIM Deployment Advisor (type 'exit' to quit)\n")
    history: list = []
    while True:
        try:
            user_message = input("customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue
        answer, history = run_advisor_turn(client, args.model, user_message, history)
        print(f"\nadvisor> {answer}\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
