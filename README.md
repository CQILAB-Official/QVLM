# QVLM — Quantum-domain Vision Language Models

This repository contains the **first Quantum-domain Vision Language Model (QVLM) playground**, designed to be scalable, reproducible, and adaptable for quantum information tasks. The framework fine-tunes vision-language backbones (Qwen2.5-VL, Qwen3-VL) as specialised experts for quantum state classification (Wigner functions), entanglement verification, circuit analysis, and quantum code generation.

---

## Features

### 🔧 Adaptable Quantum-Domain VLM
Fine-tunes Qwen vision-language backbones as specialised experts for quantum-domain tasks.

### 🤖 Expert-Driven Fine-Tuning
Domain-specific experts for:
- **Wigner Function Analysis** — identifying quantum optical states and parameters from phase-space distributions.
- **Quantum Circuit Understanding** — extracting gate sequences and logical structure from circuit diagrams.
- **Entanglement Verification** — determining entanglement types and properties from visual representations.
- **Quantum Code Generation** — translating visual quantum concepts into executable Qiskit code.

### 🔀 Mixture of Experts (MoE) Router — the main QVLM entry point
A central routing system (`moe/`) that dispatches an image + text prompt to the most suitable fine-tuned expert. **Running the MoE is the primary way to run QVLM** — it gives you access to all four experts through a single interface, with automatic task routing.

Deployable as a Gradio web app (`moe/app.py`) or a HuggingFace Inference Endpoint (`moe/handler.py`).

```
User Input (Image + Prompt)   ← choose 7B or 8B experts
        │
        ▼
  [Keyword Router] ─── detects task type from prompt
        │
   ┌────┼────┬──────────┐
   ▼    ▼    ▼          ▼
Wigner Circuit Entangle CodeGen
Expert Expert  Expert   Expert
        │
        ▼
  Expert Response
```

**Two modes:**
- **Route to Best Expert** — the router picks the single best expert and returns its response.
- **Run All Experts** — all four experts run on the same image + prompt; responses are shown side by side for comparison.

### 📊 Baseline Comparisons
Inference scripts for evaluating QVLM experts against general-purpose models:
- **B1**: Qwen3-VL-8B
- **B2**: Llama-3.2-11B-Vision
- **B3**: ChatGPT-4.1

---

## Datasets

All datasets are hosted on HuggingFace and are **downloaded automatically** when the fine-tuning or inference scripts run — no manual download step required.

| Dataset | HuggingFace | Used by |
|---|---|---|
| QVLM-Wigner | [CQILAB/QVLM-Wigner](https://huggingface.co/datasets/CQILAB/QVLM-Wigner) | Wigner experts |
| QVLM-Circuit | [CQILAB/QVLM-Circuit](https://huggingface.co/datasets/CQILAB/QVLM-Circuit) | Circuit & CodeGen experts |
| QVLM-Circuit-Entanglement | [CQILAB/QVLM-Circuit-Entanglement](https://huggingface.co/datasets/CQILAB/QVLM-Circuit-Entanglement) | Entanglement experts |

### Dataset Columns
- **image** — raw image (Wigner function, circuit diagram, etc.)
- **ground_truth** — target text label for the quantum task
- **input\_N / output\_N** — multi-turn conversation turns (Circuit and Entanglement datasets)

---

## Models

All fine-tuned expert models are hosted on HuggingFace as LoRA adapters.

| Expert | 7B Model | 8B Model |
|---|---|---|
| Wigner | [CQILAB/model_qvlm7b-wigner-expert](https://huggingface.co/CQILAB/model_qvlm7b-wigner-expert) | [CQILAB/model_qvlm8b-wigner-expert](https://huggingface.co/CQILAB/model_qvlm8b-wigner-expert) |
| Circuit | [CQILAB/model_qvlm7b-circuit-expert](https://huggingface.co/CQILAB/model_qvlm7b-circuit-expert) | [CQILAB/model_qvlm8b-circuit-expert](https://huggingface.co/CQILAB/model_qvlm8b-circuit-expert) |
| Entanglement | [CQILAB/model_qvlm7b-entanglement-expert](https://huggingface.co/CQILAB/model_qvlm7b-entanglement-expert) | [CQILAB/model_qvlm8b-entanglement-expert](https://huggingface.co/CQILAB/model_qvlm8b-entanglement-expert) |
| CodeGen | [CQILAB/model_qvlm7b-codegen-expert](https://huggingface.co/CQILAB/model_qvlm7b-codegen-expert) | [CQILAB/model_qvlm8b-codegen-expert](https://huggingface.co/CQILAB/model_qvlm8b-codegen-expert) |

Base backbones:
- **7B experts** — `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- **8B experts** — `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`

---

## Setup

### 1. Environment
```bash
conda env create -f environment.yml
conda activate qvlm
```

### 2. HuggingFace Login
Required to download models and datasets from the CQILAB organisation:
```bash
huggingface-cli login
```

---

## Running QVLM

The recommended way to use QVLM interactively is through the **MoE Gradio app**, which gives you access to all four fine-tuned experts through a single web interface.

### Quick Start

```bash
# 1. Activate the environment
conda activate qvlm

# 2. Log in to HuggingFace (models are downloaded automatically on first use)
huggingface-cli login

# 3. Launch QVLM
python moe/app.py
```

Open your browser at **http://localhost:7860**.

---

### Choosing the Model Size

Both the Gradio app and the API endpoint support two expert sizes. Select the size that matches your VRAM budget:

| Size | Backbone | VRAM per expert (4-bit) | Best for |
|---|---|---|---|
| **7B** (default) | Qwen2.5-VL-7B | ~4 GB | Faster inference, lower VRAM |
| **8B** | Qwen3-VL-8B | ~5 GB | Higher accuracy, stronger reasoning |

In the Gradio app, use the **"Expert Model Size"** radio button (7b / 8b) before submitting.

> **VRAM guide:**
> - "Route to Best Expert" mode loads **one expert** at a time (lazy loading) — minimum ~4 GB.
> - "Run All Experts" mode runs all four experts sequentially on the same GPU — peak usage ~4–5 GB (models are loaded one at a time and cached).

---

### Gradio App — Two Modes

#### Tab 1 — Route to Best Expert
The router analyses your prompt with keyword matching and automatically selects the most suitable expert:

| Routing keywords (examples) | Expert selected |
|---|---|
| wigner, fock, coherent, cat state, phase space, thermal, photon | **WIGNER** |
| circuit, gate, hadamard, cnot, rx/ry/rz, unitary | **CIRCUIT** |
| entanglement, bell state, epr, separable, concurrence | **ENTANGLEMENT** |
| code, qiskit, implement, generate, python, write | **CODEGEN** |

The routing decision and the expert's full response are shown.

#### Tab 2 — Run All Experts
All four experts run on the same image and prompt. The router's recommendation is shown alongside all four independent responses — useful for comparison or when the task type is ambiguous.

---

### API Usage via `handler.py`

`moe/handler.py` implements the HuggingFace Inference Endpoint interface and can also be called programmatically:

```python
import base64
from moe.handler import EndpointHandler

handler = EndpointHandler()

with open("my_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Route to the best expert (7B)
response = handler({
    "inputs": {
        "image":      image_b64,
        "prompt":     "What quantum state is shown in this Wigner function?",
        "model_size": "7b",        # "7b" (default) or "8b"
    }
})
print(response["route_selected"])  # e.g. "WIGNER"
print(response["result"])          # expert response text

# Run all four experts (8B) and compare
response_all = handler({
    "inputs": {
        "image":            image_b64,
        "prompt":           "Describe the quantum circuit shown.",
        "model_size":       "8b",
        "run_all_experts":  True,
    }
})
for expert, text in response_all["all_expert_results"].items():
    print(f"\n=== {expert} ===\n{text}")
```

---

## Fine-Tuning

Datasets are downloaded from HuggingFace automatically on first run. Results are saved under `model/`.

### Wigner Expert
```bash
python src/finetune/qvlm7b-wigner-expert.py   # Qwen2.5-VL 7B
python src/finetune/qvlm8b-wigner-expert.py   # Qwen3-VL 8B
```
Dataset: `CQILAB/QVLM-Wigner`

### Circuit Expert
```bash
python src/finetune/qvlm7b-circuit-expert.py  # Qwen2.5-VL 7B
python src/finetune/qvlm8b-circuit-expert.py  # Qwen3-VL 8B
```
Dataset: `CQILAB/QVLM-Circuit`

### Entanglement Expert
```bash
python src/finetune/qvlm7b-entanglement-expert.py  # Qwen2.5-VL 7B
python src/finetune/qvlm8b-entanglement-expert.py  # Qwen3-VL 8B
```
Dataset: `CQILAB/QVLM-Circuit-Entanglement`

### CodeGen Expert
```bash
python src/finetune/qvlm7b-codegen-expert.py  # Qwen2.5-VL 7B
python src/finetune/qvlm8b-codegen-expert.py  # Qwen3-VL 8B
```
Dataset: `CQILAB/QVLM-Circuit` (turn 6 — code generation step)

### Uploading Fine-Tuned Models to HuggingFace
After training, upload all expert checkpoints to HuggingFace:
```bash
python upload_models_to_hf.py
```
This pushes the latest checkpoint of each model under `CQILAB/model_qvlm*-*-expert`.

---

## Inference

All expert inference scripts load the model and dataset directly from HuggingFace. Results are written to `ResultUpdate/`.

### QVLM Expert Models

| Task | 7B Script | 8B Script |
|---|---|---|
| Wigner | `src/inference/qvlm7b-wigner-expert.py` | `src/inference/qvlm8b-wigner-expert.py` |
| Circuit | `src/inference/qvlm7b-circuit-expert.py` | `src/inference/qvlm8b-circuit-expert.py` |
| Entanglement | `src/inference/qvlm7b-entanglement.py` | `src/inference/qvlm8b-entanglement.py` |
| CodeGen | `src/inference/qvlm7b-codegen.py` | `src/inference/qvlm8b-codegen.py` |

```bash
# Examples
python src/inference/qvlm7b-wigner-expert.py
python src/inference/qvlm7b-circuit-expert.py
python src/inference/qvlm7b-entanglement.py
python src/inference/qvlm7b-codegen.py
```

### Baseline Models

| Baseline | Wigner script | Circuit / Entanglement / CodeGen script |
|---|---|---|
| B1 — Qwen3-VL-8B | `src/inference/b1-qwen8b-wigner.py` | `src/inference/b1-qwen8b-circuit-ent-codegen.py` |
| B2 — Llama-3.2-11B | `src/inference/b2-llama-3.2-wigner.py` | `src/inference/b2-llama-3.2-circuit-ent-codegen.py` |
| B3 — ChatGPT-4.1 | `src/inference/b3-chatgpt4.1-wigner.py` | `src/inference/b3-chatgpt4.1-circuit-ent-codegen.py` |

```bash
# Examples
python src/inference/b1-qwen8b-wigner.py
python src/inference/b2-llama-3.2-circuit-ent-codegen.py
```

### MoE Router — Running QVLM (Gradio App)

The MoE Gradio app is the **primary interface for running QVLM**. It integrates all four expert models and handles routing automatically.

```bash
# Launch with default settings (experts load lazily on first use)
python moe/app.py
```

Open **http://localhost:7860**. Use the **"Expert Model Size"** radio button to switch between 7B (Qwen2.5-VL) and 8B (Qwen3-VL) experts at runtime.

See the [Running QVLM](#running-qvlm) section above for full details on modes, VRAM requirements, and the routing table.

#### Deploy as a HuggingFace Inference Endpoint

```bash
# moe/handler.py is the endpoint entry point
# Deploy to HuggingFace Spaces / Inference Endpoints using this file.
# Supports model_size ("7b"/"8b") and run_all_experts (true/false) in the request body.
```

---

## Repository Structure

```
QVLM/
├── src/
│   ├── finetune/          # Fine-tuning scripts (8 experts × 2 backbones)
│   └── inference/         # Inference scripts (experts + 3 baselines)
├── moe/                   # MoE router (handler.py + app.py)
├── model/                 # Local model checkpoints (after training)
├── ResultUpdate/          # Inference output CSVs
├── logs/                  # Training logs
├── upload_models_to_hf.py # Script to push trained models to HuggingFace
└── environment.yml        # Conda environment
```

---

## Reproducibility

### 📊 Performance Metrics
- Accuracy, F1 Score, BERT Score, BLEU Score, Word Error Rate (WER)
- Custom quantum metrics (e.g., parameter estimation error, state classification accuracy)

### 🔁 Consistent Splits
All scripts use `seed=42` for dataset shuffling and train/test splitting, ensuring reproducible evaluation across runs.

---

## Citation
If you use this framework in your research, please cite:
```bibtex
```

---

## Links
- Repository: [CQILAB/QVLM](https://github.com/CQILAB-Official/QVLM)
- Organisation: [CQILAB on HuggingFace](https://huggingface.co/CQILAB)

## License
MIT License

## Contributors
CQILAB Contributors

## Changelog
- **MoE router rewritten** — `moe/handler.py` and `moe/app.py` now load and run the real fine-tuned expert models (Wigner, Circuit, Entanglement, CodeGen) using the same `FastVisionModel` inference pipeline as `src/inference/`. Supports 7B and 8B model sizes, keyword-based quantum routing, and a "Run All Experts" mode.
- Migrated all datasets and models to HuggingFace (`CQILAB/` organisation).
- Added 8 fine-tuned expert models (7B and 8B variants for Wigner, Circuit, Entanglement, CodeGen).
- All fine-tuning and inference scripts now load datasets and models directly from HuggingFace.
- Initial release of QVLM framework and expert scripts.
