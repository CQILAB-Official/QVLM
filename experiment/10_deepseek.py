import os
import threading
import torch
from unsloth import FastLanguageModel
from PIL import Image
from unsloth.chat_templates import get_chat_template
# from src.utils import find_highest_checkpoint
from transformers import TextIteratorStreamer

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


def initialize_model(model_id: str, checkpoint_root: str = "./model_cp"):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    # Check if local fine-tuned model is present and non-empty
    # try:
    #     adapter_path = find_highest_checkpoint(checkpoint_root)
    #     print(f"Highest checkpoint found: {adapter_path}")
    #     model_name = adapter_path
    # except:
    model_name = model_id

    print(f"Loading model from: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
    )
    
    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER


def format_data_inference(tokenizer, user_input, model_id: str) -> str:
    template_name = None
    model_id_lower = model_id.lower()

    if "mistral" in model_id_lower:
        template_name = "mistral"
    elif "llama" in model_id_lower:
        template_name = "llama-3"
    elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
        template_name = None

    if template_name:
        row_json = [{"role": "user", "content": user_input}]
        tokn = get_chat_template(
            tokenizer,
            chat_template=template_name,
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            map_eos_token=True,
        )
        try:
            formatted_text = tokn.apply_chat_template(
                row_json,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            formatted_text = f"### Instruction:\n{user_input}\n### Response:\n"
    elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
        formatted_text = f"### Instruction:\n{user_input}\n### Response:\n"
    else:
        formatted_text = (
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return formatted_text


def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 100000, model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:
    model, tokenizer = initialize_model(model_id)
    FastLanguageModel.for_inference(model)
    prompt = format_data_inference(tokenizer, user_input, model_id) 

    # 4. Tokenize inputs
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # 5. Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        # pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        repetition_penalty=1.2,
        use_cache=True 
    )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    if "llama" in model_id.lower():
        unwanted_prefix = "assistant\n\n"
        if generated_text.startswith(unwanted_prefix):
            generated_text = generated_text[len(unwanted_prefix):].lstrip()
    
    return generated_text


if __name__ == "__main__":
    # Example usage
    user_input = """
    DO NOT SUMMARIZE. ONLY OUTPUT RAW TAGGED BLOCKS.
    Generate  2-qubit Grover circuits, each with a marked state (|00⟩).

For the circuit, do the following:

1. Start with a reasoning block in this format:
<think>
1. Based on the imagined image, the circuit starts with Hadamard gates on both qubits. This suggests a superposition state.
2. The oracle appears to use X gates followed by CZ, then X gates again to mark a specific state like |11⟩ or |01⟩.
3. The circuit uses 2 qubits: q[0], q[1].
4. Classical registers are assumed: c[0], c[1].
5. Total circuit depth is approximately 6:
   * Layer 1: Hadamard
   * Layer 2: X (pre-oracle)
   * Layer 3: CZ gate
   * Layer 4: Undo X
   * Layer 5: Diffusion
   * Layer 6: Measurement
6. Oracle targets a specific basis state.
7. No learned parameters or embeddings used.
8. Logical gate layout is inferred visually.
</think>

2. Then output the full OpenQASM 2.0 code block in this format:
<OPENQASM code>
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Initialization
h q[0];
h q[1];

// Oracle for |11>
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];

// Diffusion
h q[0];
h q[1];
x q[0];
x q[1];
cz q[0], q[1];
x q[0];
x q[1];
h q[0];
h q[1];

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
</OPENQASM code>

Repeat this process for 10 variations. Output only <think> and <OPENQASM code> sections for each circuit. Do not explain, summarize, or add commentary outside these tags.
"""
    model_id = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
    response = run_inference_lm(user_input, model_id=model_id)
    print("Response:", response)