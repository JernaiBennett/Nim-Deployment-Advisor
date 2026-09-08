# NIM Deployment Advisor

A capacity-planning agent for NVIDIA NIM deployments. A customer describes their use case in
plain language — model, available GPU hardware, whether latency or throughput matters more —
and the agent returns a real sizing recommendation: precision, GPU count, and optimization
target, backed by deterministic math rather than an LLM guess.

This automates the conversation a Solutions Architect has with a customer *before* any
deployment happens: will this actually fit, and what should they request.

## What it does

- **Extracts** the customer's requirements (model size, hardware, priorities) from a natural
  language description via LLM tool-calling.
- **Calculates**, with real deterministic logic, not an LLM estimate:
  - Memory required at a given precision (`params × bytes-per-param + overhead`, the same rule
    of thumb NVIDIA's own NIM docs use)
  - Minimum GPU count, rounded to the power-of-2 splits tensor parallelism actually uses
  - Whether the customer's *actual* available hardware can support it — and says so plainly
    when it can't, rather than pretending
- **Recommends** a precision (defaulting to fp8, matching NIM's own profile-selection
  preference order) and an optimization target (latency vs. throughput).

## How it works

```
Customer describes use case in plain language
   │
   ▼
advisor.py    → sends the message + available tools to the LLM
   │
   ▼
tools.py      → if the model calls recommend_deployment_tool, dispatches to:
   │
   ▼
sizing.py     → deterministic calculation:
                 1. pick precision (fp8 default, fp16 if accuracy-critical)
                 2. compute total memory needed
                 3. compute GPU count (rounded to next power of 2)
                 4. check against the customer's actual available GPUs
                 5. return fits/doesn't-fit + full reasoning trail
   │
   ▼
advisor.py    → LLM writes the final recommendation grounded in that result
```

## Example interaction

```
customer> I need to deploy Llama 70B for a customer support chatbot.
          Latency matters. They have 2x A100-80GB.

advisor> That fits. Recommendation: fp8 precision on 2x A100-80GB
         (94 GB total memory needed, 47 GB per GPU), optimized for latency.
         fp8 is NIM's own default preference when available, and 2 GPUs
         matches your hardware exactly with no need to scale up.
```

```
customer> Same model, but they only have 1 L40S.

advisor> That doesn't fit. Even at fp8, a 70B model needs ~94 GB and an
         L40S has 48 GB — you'd need 2 GPUs minimum, but they only have 1.
         Options: add a second L40S, drop to a smaller model, or move to
         int4/awq quantization (though that may not fully close the gap
         and comes with an accuracy tradeoff).
```

## Design notes

**Why this is deterministic math, not an LLM estimate:** an LLM asked "how much memory does a
70B model need" will produce a plausible-sounding but unverified number. `sizing.py` uses the
same params-times-bytes-per-parameter rule of thumb documented in NVIDIA's own NIM
documentation, so every number in the final answer traces back to a fixed, auditable formula —
the LLM's job is extracting the customer's intent and writing the explanation, not doing the math.

**Why gpu_count isn't silently capped to what the customer has:** if a customer's hardware can't
support the ideal deployment, the honest answer is "this doesn't fit" plus real alternatives —
not a quietly downgraded recommendation that looks like it works. `fits=False` and the reasoning
trail carry that message explicitly.

**Scope limitation, stated plainly:** this tool answers "will it fit and what should I request,"
not "here's your exact tokens/sec." Real throughput/latency numbers depend heavily on batch size,
sequence length, and specific hardware generation — this tool doesn't fabricate those, on purpose.

## Project structure

```
nim-deployment-advisor/
├── data/
│   └── gpu_specs.json      # real GPU VRAM specs (A100, H100, L40S, L4, H200)
├── src/
│   ├── sizing.py            # deterministic capacity-planning math
│   ├── tools.py              # wraps sizing.py as an LLM-callable tool
│   └── advisor.py            # orchestration: natural language → grounded recommendation
└── requirements.txt
```

## Try it

```bash
pip install -r requirements.txt
cd src
python advisor.py --demo
```

Runs the sizing tool directly without needing an API key. For a live conversational advisor,
set `NVIDIA_API_KEY` (free at [build.nvidia.com](https://build.nvidia.com)) and run
`python advisor.py`.
