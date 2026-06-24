from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model with 4-bit quant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# prepare the model input
prompt = """
    DO NOT SUMMARIZE. ONLY OUTPUT RAW TAGGED BLOCKS.
    Generate random QPE circuit with depth 8 and number of qubits 6 (Qiskit qasm), using logical gate, use only minimal classical register for measurement
For the circuit, do the following:

1. Start with a reasoning block in this format:
<thought>  
Based on the image, consist of pattern .. there is CCNOT and X gate as oracle then it is classified as Grover (as detail as possible) (Might be different for QPE, QML etc) 
(You may add more steps here as detail as possible)
(as detail as possible, you may add more here EXAMPLE: (not strictly following this)
- Based on the imagined image, the circuit starts with Hadamard gates on both qubits. This suggests a superposition state.
- The oracle appears to use X gates followed by CZ, then X gates again to mark a specific state like |11⟩ or |01⟩.
- The circuit uses 2 qubits: q[0], q[1].
- Classical registers are assumed: c[0], c[1].
- Total circuit depth is approximately 6:
   * Layer 1: Hadamard
   * Layer 2: X (pre-oracle)
   * Layer 3: CZ gate
   * Layer 4: Undo X
   * Layer 5: Diffusion
   * Layer 6: Measurement
- Oracle targets a specific basis state.
- No learned parameters or embeddings used.
- Logical gate layout is inferred visually.
)

2. Number of qubits, how many quantum classical registers, is there measurement, reverse engineering from image to thinking process, how to classify number of qubits, registers, and measurement

3. Composition - Logical GATE (reverse OCR): 
eg: 
8-depth and 7-qubit:
rz0 none none none none none none; sx0 none none none none none none; rz0 none none none none none none; sx0 none none none none none none; rz0 none none none none none none; rz0 none none none none none none; x0 none none none none none none; measure0 none none none none none none
4-depth and 7-qubit:
rz0 none none none none none none; rz0 none none none none none none; rz0 none none none none none none; measure0 none none none none none none
</thought>

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

Output only start with <think> .. </think> and end with <OPENQASM code> ... </OPENQASM code> sections for the circuit. Do not explain, summarize, or add commentary outside these tags.
"""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1200
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))