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

### 🔀 Mixture of Experts (MoE) Router
A central routing system (`moe/`) that dispatches an image + prompt to the most suitable expert. Deployable as a HuggingFace Inference Endpoint (`moe/handler.py`) or as a Gradio web app (`moe/app.py`).

```
User Input (Image + Prompt)
        │
        ▼
  [Router Model] ─── analyzes task type
        │
   ┌────┼────┬──────────┐
   ▼    ▼    ▼          ▼
Wigner Circuit Entangle CodeGen
Expert Expert  Expert   Expert
```

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

### MoE Router (Gradio Demo)
```bash
python moe/app.py
```
Or deploy as a HuggingFace Inference Endpoint using `moe/handler.py`.

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
- Migrated all datasets and models to HuggingFace (`CQILAB/` organisation).
- Added 8 fine-tuned expert models (7B and 8B variants for Wigner, Circuit, Entanglement, CodeGen).
- All fine-tuning and inference scripts now load datasets and models directly from HuggingFace.
- Initial release of QVLM framework and expert scripts.
