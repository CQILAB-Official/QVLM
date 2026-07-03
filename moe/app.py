import gradio as gr
from PIL import Image
from unsloth import FastVisionModel

# ──────────────────────────────────────────
# EXPERT REGISTRY
# ──────────────────────────────────────────
EXPERTS = {
    "WIGNER": {
        "7b": "CQILAB/model_qvlm7b-wigner-expert",
        "8b": "CQILAB/model_qvlm8b-wigner-expert",
        "max_new_tokens": 1000,
        "label": "Wigner / Quantum State",
    },
    "CIRCUIT": {
        "7b": "CQILAB/model_qvlm7b-circuit-expert",
        "8b": "CQILAB/model_qvlm8b-circuit-expert",
        "max_new_tokens": 1500,
        "label": "Quantum Circuit",
    },
    "ENTANGLEMENT": {
        "7b": "CQILAB/model_qvlm7b-entanglement-expert",
        "8b": "CQILAB/model_qvlm8b-entanglement-expert",
        "max_new_tokens": 1500,
        "label": "Entanglement",
    },
    "CODEGEN": {
        "7b": "CQILAB/model_qvlm7b-codegen-expert",
        "8b": "CQILAB/model_qvlm8b-codegen-expert",
        "max_new_tokens": 1500,
        "label": "Code Generation (Qiskit)",
    },
}

ROUTING_KEYWORDS = {
    "WIGNER": [
        "wigner", "phase space", "quantum state", "fock", "coherent",
        "cat state", "thermal", "photon", "optical", "superposition",
        "qubit state", "density matrix", "hilbert space", "squeezed",
    ],
    "CIRCUIT": [
        "circuit", "gate", "hadamard", "cnot", "cx", "cz", "pauli",
        "quantum circuit", "gate sequence", "depth", "unitary", "rx", "ry", "rz",
    ],
    "ENTANGLEMENT": [
        "entanglement", "entangled", "bell state", "epr", "correlation",
        "separable", "concurrence", "fidelity", "bipartite", "multipartite",
    ],
    "CODEGEN": [
        "code", "qiskit", "implement", "generate", "programming", "python",
        "write", "function", "class", "script", "compile", "transpile",
    ],
}

# ──────────────────────────────────────────
# LAZY MODEL CACHE
# ──────────────────────────────────────────
_loaded: dict[str, tuple] = {}


def _load_expert(expert: str, size: str):
    key = f"{expert}_{size}"
    if key not in _loaded:
        model_id = EXPERTS[expert][size]
        print(f"Loading {expert} ({size}) from {model_id} …")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(model)
        _loaded[key] = (model, tokenizer)
        print(f"{expert} ({size}) ready.")
    return _loaded[key]


# ──────────────────────────────────────────
# ROUTING
# ──────────────────────────────────────────
def _route(prompt: str) -> str:
    prompt_lower = prompt.lower()
    scores = {expert: 0 for expert in ROUTING_KEYWORDS}
    for expert, keywords in ROUTING_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                scores[expert] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "WIGNER"


# ──────────────────────────────────────────
# CORE INFERENCE
# ──────────────────────────────────────────
def _run_inference(
    model, tokenizer, image: Image.Image, prompt: str, max_new_tokens: int
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image, text, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")
    gen_ids = model.generate(
        **{k: v.to("cuda") for k, v in inputs.items()},
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        min_p=0.1,
        use_cache=True,
    )[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(
        gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def _call_expert(
    expert: str, size: str, image: Image.Image, prompt: str
) -> str:
    model, tokenizer = _load_expert(expert, size)
    return _run_inference(
        model, tokenizer, image, prompt, EXPERTS[expert]["max_new_tokens"]
    )


# ──────────────────────────────────────────
# GRADIO HANDLERS
# ──────────────────────────────────────────
def route_and_run(image: Image.Image, prompt: str, size: str):
    """Route to the best expert and run inference."""
    if image is None or not prompt.strip():
        return "Please provide both an image and a prompt.", "", ""

    image_rgb = image.convert("RGB").resize((1000, 600))
    expert = _route(prompt)
    label  = EXPERTS[expert]["label"]
    route_msg = f"Selected Expert: {expert} — {label}"

    result = _call_expert(expert, size, image_rgb, prompt)
    return route_msg, expert, result


def run_all_experts(image: Image.Image, prompt: str, size: str):
    """Run inference on ALL four experts and return each result."""
    if image is None or not prompt.strip():
        empty = "Please provide both an image and a prompt."
        return empty, empty, empty, empty, ""

    image_rgb = image.convert("RGB").resize((1000, 600))
    expert_route = _route(prompt)

    results = {}
    for expert in EXPERTS:
        print(f"Running expert: {expert} ({size})")
        results[expert] = _call_expert(expert, size, image_rgb, prompt)

    route_msg = (
        f"Router recommendation: {expert_route} — {EXPERTS[expert_route]['label']}"
    )
    return (
        results["WIGNER"],
        results["CIRCUIT"],
        results["ENTANGLEMENT"],
        results["CODEGEN"],
        route_msg,
    )


# ──────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────
with gr.Blocks(title="QVLM — Quantum MoE Router", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # QVLM — Quantum Vision-Language Model · Mixture of Experts Router
        Upload a quantum image and enter a prompt. The router dispatches to the most
        suitable fine-tuned expert (Wigner / Circuit / Entanglement / CodeGen).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input    = gr.Image(type="pil", label="Input Image")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="e.g. 'Identify the quantum state in this Wigner function.'",
                lines=4,
            )
            size_radio = gr.Radio(
                choices=["7b", "8b"],
                value="7b",
                label="Expert Model Size",
            )

        with gr.Column(scale=1):
            gr.Markdown("### Expert Descriptions")
            gr.Markdown(
                "- **WIGNER** — Identifies quantum optical states and parameters from "
                "Wigner / phase-space distributions.\n"
                "- **CIRCUIT** — Extracts gate sequences and logical structure from "
                "quantum circuit diagrams.\n"
                "- **ENTANGLEMENT** — Determines entanglement type and properties from "
                "visual representations.\n"
                "- **CODEGEN** — Translates quantum-circuit visuals into executable "
                "Qiskit code."
            )

    # ── TAB 1: Route & Run ──────────────────
    with gr.Tab("Route to Best Expert"):
        route_btn  = gr.Button("Run MoE Router", variant="primary")
        route_info = gr.Textbox(label="Routing Decision", interactive=False)
        expert_tag = gr.Textbox(label="Expert Selected", interactive=False)
        route_out  = gr.Textbox(label="Expert Response", lines=15, interactive=False)

        route_btn.click(
            fn=route_and_run,
            inputs=[img_input, prompt_input, size_radio],
            outputs=[route_info, expert_tag, route_out],
        )

    # ── TAB 2: Run All Experts ───────────────
    with gr.Tab("Run All Experts"):
        gr.Markdown(
            "Runs **all four experts** on the same image and prompt, "
            "then shows each response side by side."
        )
        all_btn       = gr.Button("Run All Experts", variant="primary")
        all_route_msg = gr.Textbox(label="Router Recommendation", interactive=False)

        with gr.Row():
            wigner_out = gr.Textbox(
                label="WIGNER Expert", lines=12, interactive=False
            )
            circuit_out = gr.Textbox(
                label="CIRCUIT Expert", lines=12, interactive=False
            )
        with gr.Row():
            ent_out = gr.Textbox(
                label="ENTANGLEMENT Expert", lines=12, interactive=False
            )
            codegen_out = gr.Textbox(
                label="CODEGEN Expert", lines=12, interactive=False
            )

        all_btn.click(
            fn=run_all_experts,
            inputs=[img_input, prompt_input, size_radio],
            outputs=[wigner_out, circuit_out, ent_out, codegen_out, all_route_msg],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
