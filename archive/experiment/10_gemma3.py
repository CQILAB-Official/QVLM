# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-12b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

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

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_input}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

