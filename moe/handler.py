import torch
import base64
from io import BytesIO
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
    },
    "CIRCUIT": {
        "7b": "CQILAB/model_qvlm7b-circuit-expert",
        "8b": "CQILAB/model_qvlm8b-circuit-expert",
        "max_new_tokens": 1500,
    },
    "ENTANGLEMENT": {
        "7b": "CQILAB/model_qvlm7b-entanglement-expert",
        "8b": "CQILAB/model_qvlm8b-entanglement-expert",
        "max_new_tokens": 1500,
    },
    "CODEGEN": {
        "7b": "CQILAB/model_qvlm7b-codegen-expert",
        "8b": "CQILAB/model_qvlm8b-codegen-expert",
        "max_new_tokens": 1500,
    },
}

# Keywords used to route prompts to the correct expert
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


class EndpointHandler:
    """
    HuggingFace Inference Endpoint handler for the QVLM MoE router.

    Input format (POST body):
    {
        "inputs": {
            "image":             "<base64-encoded image>",
            "prompt":            "User question / task description",
            "model_size":        "7b" | "8b"          (optional, default "7b"),
            "run_all_experts":   true | false          (optional, default false)
        }
    }

    Output format:
    {
        "route_selected":    "WIGNER" | "CIRCUIT" | "ENTANGLEMENT" | "CODEGEN",
        "result":            "<primary expert response>",
        "all_expert_results": { ... }   # present only when run_all_experts=true
    }
    """

    def __init__(self, path=""):
        print("Initializing QVLM MoE System …")
        # Expert models are loaded lazily (on first use) to minimise startup VRAM.
        self._loaded: dict[str, tuple] = {}
        print("MoE System ready. Experts will be loaded on first call.")

    # ──────────────────────────────────────────
    # LOADING
    # ──────────────────────────────────────────
    def _load_expert(self, expert: str, size: str = "7b"):
        key = f"{expert}_{size}"
        if key not in self._loaded:
            model_id = EXPERTS[expert][size]
            print(f"Loading expert {expert} ({size}) from {model_id} …")
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_id,
                load_in_4bit=True,
            )
            FastVisionModel.for_inference(model)
            self._loaded[key] = (model, tokenizer)
            print(f"Expert {expert} ({size}) loaded.")
        return self._loaded[key]

    # ──────────────────────────────────────────
    # ROUTING
    # ──────────────────────────────────────────
    def _route(self, prompt: str) -> str:
        """Keyword-based routing; falls back to WIGNER when no match."""
        prompt_lower = prompt.lower()
        scores = {expert: 0 for expert in ROUTING_KEYWORDS}
        for expert, keywords in ROUTING_KEYWORDS.items():
            for kw in keywords:
                if kw in prompt_lower:
                    scores[expert] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "WIGNER"

    # ──────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────
    def _run_inference(
        self,
        model,
        tokenizer,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1000,
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
        self, expert: str, size: str, image: Image.Image, prompt: str
    ) -> str:
        model, tokenizer = self._load_expert(expert, size)
        max_tokens = EXPERTS[expert]["max_new_tokens"]
        return self._run_inference(model, tokenizer, image, prompt, max_tokens)

    # ──────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────
    def __call__(self, data: dict) -> dict:
        inputs = data.pop("inputs", data)
        image_b64  = inputs.get("image", None)
        prompt     = inputs.get("prompt", "")
        size       = inputs.get("model_size", "7b")
        run_all    = inputs.get("run_all_experts", False)

        if not image_b64 or not prompt:
            return {"error": "Both 'image' (base64) and 'prompt' must be provided."}

        if size not in ("7b", "8b"):
            return {"error": "model_size must be '7b' or '8b'."}

        try:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((1000, 600))
        except Exception as exc:
            return {"error": f"Failed to decode image: {exc}"}

        # ── STEP 1: ROUTING ──
        route = self._route(prompt)
        print(f"Router selected expert: {route}")

        # ── STEP 2: INFERENCE ──
        if run_all:
            all_results = {}
            for expert in EXPERTS:
                print(f"Running expert: {expert}")
                all_results[expert] = self._call_expert(expert, size, image, prompt)
            return {
                "route_selected": route,
                "result": all_results[route],
                "all_expert_results": all_results,
            }
        else:
            result = self._call_expert(route, size, image, prompt)
            return {
                "route_selected": route,
                "result": result,
            }
