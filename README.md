# QVLM - Quantum-domain Vision Language Models for Quantum Optical States and Circuits

This repository contains the **first Quantum-domain Vision Language Model (QVLM) playground**, designed to be scalable, reproducible, and adaptable for quantum information tasks. The framework is tailored for quantum state classification (Wigner functions), entanglement verification, circuit analysis, and quantum code generation. It leverages state-of-the-art vision-language models fine-tuned as "experts" for diverse quantum-domain applications.

---

## Features of the QVLM Framework

### 🔧 Adaptable Quantum-Domain VLM
A prototype that fine-tunes mainly qwen vision-language backbones (e.g., Qwen2.5-VL, Qwen3-VL) as specialized experts for quantum-domain tasks.

### 🤖 Expert-Driven Fine-Tuning
The integration of domain-specific expert training for:
- **Wigner Function Analysis**: Identifying quantum optical states and parameters from phase-space distributions.
- **Quantum Circuit Understanding**: Extracting gate sequences and logical structure from circuit diagrams.
- **Entanglement Verification**: Determining entanglement types and properties from visual representations.
- **Quantum Code Generation**: Translating visual quantum concepts into executable code (e.g., Qiskit).

### 🔀 Mixture of Experts (MoE) Router
A central routing system (`moe/`) that acts as the entry-point dispatcher for the QVLM framework. Given an image and a text prompt, the **Router Model** automatically analyzes the inputs and forwards them to the most suitable expert model (Wigner, Circuit, Entanglement, or CodeGen). The MoE system is designed for deployment as a **Hugging Face Inference Endpoint** (`moe/handler.py`) and includes a **Gradio web interface** (`moe/app.py`) for interactive use.

```
User Input (Image + Prompt)
        │
        ▼
  [Router Model] ─── analyzes task type
        │
   ┌────┼────┬────────┐
   ▼    ▼    ▼        ▼
Wigner  Circuit  Entangle  CodeGen
Expert  Expert   Expert    Expert
```

### 📊 Comprehensive Baseline Comparison
Includes extensive inference scripts to evaluate QVLM performance against leading general-purpose models (Baselines):
- **B1**: Qwen-8B
- **B2**: Llama-3.2-11B
- **B3**: ChatGPT-4.1

### 📥 Dataset Download and Overview

#### Main Dataset
**[Download the QVLM dataset here](https://huggingface.co/datasets/CQILAB/QVLM)** *(Placeholder for CQILAB repository)*

### 📝 Case Study on Quantum Tasks
A detailed case study evaluating baseline models across various quantum semantic tasks, assessing performance and robustness under different state configurations to validate the QVLM framework.

<p align="center">
  <img src="visuals/figures/Fig1.pdf" width="50%" />
  <img src="visuals/figures/Fig2.pdf" width="45%" />
</p>

---

## Table of Supported Experts

<table>
  <tr>
    <th>Wigner Experts</th>
    <th>Circuit Experts</th>
    <th>Entanglement Experts</th>
    <th>CodeGen Experts</th>
  </tr>
  <tr>
    <td>QVLM-7B Wigner</td>
    <td>QVLM-7B Circuit</td>
    <td>QVLM-7B Entanglement</td>
    <td>QVLM-7B CodeGen</td>
  </tr>
  <tr>
    <td>QVLM-8B Wigner</td>
    <td>QVLM-8B Circuit</td>
    <td>QVLM-8B Entanglement</td>
    <td>QVLM-8B CodeGen</td>
  </tr>
  <tr>
    <td>Qwen2.5-VL (Baseline)</td>
    <td>Llama-3.2-Vision (Baseline)</td>
    <td>ChatGPT-4o (Baseline)</td>
    <td></td>
  </tr>
</table>

---

## Setup Instructions

### 1. Environment Setup
- Install [Anaconda](https://www.anaconda.com/products/distribution).
- Create an environment using:
  ```bash
  conda env create -f environment.yml
  conda activate qvlm
  ```

### 2. Dataset Setup
- Download the dataset from [HuggingFace🤗](https://huggingface.co/datasets/CQILAB/QVLM-6G):
  ```bash
  cd QuantumVLM
  git clone https://huggingface.co/datasets/CQILAB/QVLM_dataset/
  ```

### 3. Training/Fine-tuning Scripts
- To fine-tune an expert model (example: QVLM-7B Wigner):
  ```bash
  python src/finetune/qvlm7b-wigner-expert.py
  ```
- To fine-tune a Circuit expert on QVLM-8B:
  ```bash
  python src/finetune/qvlm8b-circuit-expert.py
  ```

### 4. Inference and Experiments
- Run inference for a specific baseline (example: Baseline 1 - Qwen8B):
  ```bash
  python src/inference/b1-qwen8b-wigner.py
  ```
- Run inference for a QVLM expert model:
  ```bash
  python src/inference/qvlm7b-wigner-expert.py
  ```

---

## Reproducibility

### 🗃️ Dataset 
Labeled dataset with ground-truth data and quantum images.
#### Dataset Columns
- **image**: Raw image data (Wigner functions, Circuits, etc.) used for training and evaluation.
- **ground_truth**: The target text/labels for the quantum task.

### 🏗️ Testbed
The framework supports rapid prototyping and evaluation of new quantum-domain experts by swapping backbones and expert-specific datasets.

### 💻 Modular Structure
- `dataset/`: Central storage for all quantum datasets.
- `src/finetune/`: Scripts for fine-tuning expert models.
- `src/inference/`: Scripts for baseline comparisons and expert inference.
- `moe/`: **Main MoE router** — `handler.py` (HuggingFace endpoint) and `app.py` (Gradio UI) for dispatching tasks to the correct QVLM expert.
- `model/`: Directory for storing model weights and checkpoints.
- `results/`: Directory for inference outputs and evaluation CSVs.
- `logs/`: Directory for training and execution logs.

### 🔀 Running the MoE Router
- Launch the Gradio demo locally:
  ```bash
  python moe/app.py
  ```
- Deploy as a Hugging Face Inference Endpoint using `moe/handler.py` as the custom handler.
  
### 📊 Performance Metrics
- Metrics include:
  - **Accuracy**
  - **F1 Score**
  - **BERT Score**
  - **BLEU Score**
  - **Word Error Rate (WER)**
  - **Custom Quantum Metrics** (e.g., parameter estimation error)

---

## Citation
If you use this framework in your research, please cite:

```bibtex
```

---

## Others

- Official Repository: [CQILAB/QVLM](https://github.com/CQILAB-Official/QVLM)
- Dataset: [HuggingFace](https://huggingface.co/datasets/CQILAB/QVLM)

## License
```
MIT License
```

## Contributor
- CQILAB Contributors

## Changelogs
- Initial release of QVLM framework and expert scripts.
