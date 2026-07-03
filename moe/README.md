# QVLM — Mixture of Experts (MoE) Router

This directory is the **primary entry point for running QVLM**. It contains a keyword-based router that dispatches an image + text prompt to the correct fine-tuned quantum expert, then returns the expert's response.

---

## Files

| File | Purpose |
|---|---|
| `handler.py` | HuggingFace Inference Endpoint handler — also usable as a Python API |
| `app.py` | Gradio web app — the recommended way to run QVLM interactively |

---

## Experts

Four quantum-domain experts are supported, each available in a 7B and an 8B variant:

| Expert key | Task | 7B model | 8B model |
|---|---|---|---|
| `WIGNER` | Quantum optical state identification from Wigner / phase-space images | `CQILAB/model_qvlm7b-wigner-expert` | `CQILAB/model_qvlm8b-wigner-expert` |
| `CIRCUIT` | Gate sequence and circuit structure extraction from circuit diagrams | `CQILAB/model_qvlm7b-circuit-expert` | `CQILAB/model_qvlm8b-circuit-expert` |
| `ENTANGLEMENT` | Entanglement type and property determination | `CQILAB/model_qvlm7b-entanglement-expert` | `CQILAB/model_qvlm8b-entanglement-expert` |
| `CODEGEN` | Qiskit code generation from quantum-circuit visuals | `CQILAB/model_qvlm7b-codegen-expert` | `CQILAB/model_qvlm8b-codegen-expert` |

Backbones:
- **7B** — `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` (~4 GB VRAM per expert, 4-bit)
- **8B** — `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (~5 GB VRAM per expert, 4-bit)

Models are loaded **lazily** (on first use) and cached for the session.

---

## Routing

Routing is keyword-based — the router scores the user prompt against each expert's keyword list and picks the highest-scoring expert. When no keywords match, it defaults to `WIGNER`.

| Keywords (examples) | Expert routed to |
|---|---|
| wigner, fock, coherent, cat state, phase space, thermal, photon, optical, squeezed | `WIGNER` |
| circuit, gate, hadamard, cnot, cx, cz, rx/ry/rz, unitary, depth | `CIRCUIT` |
| entanglement, entangled, bell state, epr, separable, concurrence, bipartite | `ENTANGLEMENT` |
| code, qiskit, implement, generate, python, write, compile, transpile | `CODEGEN` |

---

## Gradio App — `app.py`

### Launch

```bash
python moe/app.py
```

Open **http://localhost:7860**.

### Model size

Use the **"Expert Model Size"** radio button in the UI to choose between `7b` and `8b` before submitting. Models are loaded lazily on first use.

### Two modes

**Tab 1 — Route to Best Expert**
The router selects one expert based on your prompt. The routing decision and the expert's response are displayed.

**Tab 2 — Run All Experts**
All four experts run on the same image and prompt. Each response is shown in its own panel alongside the router's recommendation — useful for comparison or when the task type is ambiguous.

---

## HuggingFace Inference Endpoint — `handler.py`

`handler.py` defines `EndpointHandler`, the class looked up by HuggingFace's serverless infrastructure. Deploy this file as a HuggingFace Inference Endpoint.

### Request format

```json
{
  "inputs": {
    "image":           "<base64-encoded image string>",
    "prompt":          "What quantum state is shown in this Wigner function?",
    "model_size":      "7b",
    "run_all_experts": false
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | string | required | Base64-encoded image (PNG / JPEG) |
| `prompt` | string | required | User question or task description |
| `model_size` | `"7b"` / `"8b"` | `"7b"` | Which expert checkpoint to use |
| `run_all_experts` | bool | `false` | `true` → run all four experts; `false` → route to one |

### Response format

```json
{
  "route_selected": "WIGNER",
  "result": "<primary expert response>"
}
```

When `run_all_experts` is `true`, the response also includes:

```json
{
  "route_selected": "WIGNER",
  "result": "<response from the routed expert>",
  "all_expert_results": {
    "WIGNER":       "<response>",
    "CIRCUIT":      "<response>",
    "ENTANGLEMENT": "<response>",
    "CODEGEN":      "<response>"
  }
}
```

### Programmatic usage

```python
import base64
from moe.handler import EndpointHandler

handler = EndpointHandler()

with open("wigner_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Single expert (7B)
response = handler({
    "inputs": {
        "image":      image_b64,
        "prompt":     "Identify the quantum state and its parameters.",
        "model_size": "7b",
    }
})
print(response["route_selected"])  # WIGNER
print(response["result"])

# All experts (8B)
response = handler({
    "inputs": {
        "image":           image_b64,
        "prompt":          "Describe what you see.",
        "model_size":      "8b",
        "run_all_experts": True,
    }
})
for expert, text in response["all_expert_results"].items():
    print(f"\n=== {expert} ===\n{text}")
```

---

## VRAM Requirements

| Mode | Model size | Peak VRAM |
|---|---|---|
| Route to Best Expert | 7B | ~4 GB |
| Route to Best Expert | 8B | ~5 GB |
| Run All Experts | 7B | ~4 GB (experts loaded sequentially and cached) |
| Run All Experts | 8B | ~5 GB (experts loaded sequentially and cached) |
