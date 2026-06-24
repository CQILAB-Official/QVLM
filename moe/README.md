# Multi-Model "Banana" System (Mixture of Experts) 🍌

This directory contains the boilerplate code to deploy a Multi-Model AI Routing system (where one small model routes inputs to larger, specialized sub-models).

## Files Included

1. `handler.py` - Use this if you want to deploy to **Hugging Face Inference Endpoints** (API only, no UI). Hugging Face's serverless infrastructure directly looks for the `EndpointHandler` class defined in this file.
2. `app.py` - Use this if you want to deploy to **Hugging Face Spaces** (Creates a Gradio Web UI).

## Next Steps

To make this functional, you need to replace the mockup code in these files with actual model loading code:
1. Pick a fast router model (e.g., `Qwen2.5-VL-3B-Instruct`).
2. Pick your specialized expert models (e.g., `Qwen2.5-VL-72B-Instruct` for heavy general tasks, or specialized Math OCR models).
3. If memory (VRAM) is a constraint, consider making HTTP requests (`requests.post()`) inside your expert functions pointing to externally hosted API endpoints rather than loading all weights into the same GPU.
