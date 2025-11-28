import os
import textwrap
import threading
import torch
import csv
import os
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
# from src.utils import find_highest_checkpoint
# from transformers import TextIteratorStreamer
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag
import base64
import pandas as pd
import os
import time
import openai
from PIL import Image
import io
import csv
import uuid


def count_logical_gate_layers(qc: QuantumCircuit) -> int:
    dag = circuit_to_dag(qc)
    layers = 0
    for layer in dag.layers():
        instructions = list(layer["graph"].op_nodes())
        if any(instr.name not in ("measure", "barrier", "delay") for instr in instructions):
            layers += 1
    return layers


def depth_exclude_measure(qc: QuantumCircuit) -> int:
    stripped = qc.remove_final_measurements(inplace=False)
    return stripped.depth()

def validate_qc(qc: QuantumCircuit, expected_qubits: int, expected_depth: int) -> tuple[bool, int, int]:
    actual_qubits = qc.num_qubits
    actual_depth = depth_exclude_measure(qc)

    if actual_qubits != expected_qubits or actual_depth != expected_depth:
        return False, actual_qubits, actual_depth

    return True, actual_qubits, actual_depth



# Access the key
from dotenv import load_dotenv

# Load from .env
load_dotenv()

# Access the key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup OpenAI client (new in v1+)
client = openai.OpenAI(api_key=openai_api_key)


def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 1500, model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a quantum circuit generator assistant that will carefully analyze and generate circuit. Be careful with the circuit DEPTH."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/png;base64,{base64_image}",
                    #         "detail": "high"
                    #     }
                    # }
                ]
            }
        ],
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def get_prompt(qubit, depth, additional="", variations=""):
    return textwrap.dedent(f"""\
DO NOT SUMMARIZE, DO NOT OUTPUT CHINESE, STAY IN CONTEXT. ONLY OUTPUT RAW TAGGED BLOCKS.
Generate random **Variational Quantum Eigensolver (VQE)** (For the circuit, strictly follow the VQE structure!) circuit with depth {depth} (MAKE SURE DEPTH IS CORRECT) and number of qubits {qubit} (Qiskit qasm), using logical gate, use only minimal classical register for measurement
For the circuit, do the following: 

Start with a reasoning block in this format (EXACTLY FOLLOW THIS FORMAT WITH NUMBERING):
<thought>  
1. Based on the image, consist of pattern .. there is CCNOT and X gate as oracle then it is classified as Grover (as detail as possible) (Might be different for QPE, QML etc) 
(You may add more steps here as detail as possible)
(as detail as possible, you may add more here EXAMPLE ANALYSIS: (not strictly following this)
- Based on the imagined image, the circuit starts with Hadamard gates on both qubits. This suggests a superposition state.
- The oracle appears to use X gates followed by CZ, then X gates again to mark a specific state like |11⟩ or |01⟩.
- The circuit uses 2 qubits: q[0], q[1].
- Classical registers are assumed: c[0], c[1].
- Total circuit depth is approximately 6 (explain each layers):
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
2. Given the circuit, how you identify it as Variational Quantum Eigensolver (VQE)?

3. Number of qubits, how many quantum classical registers, is there measurement, reverse engineering from image to thinking process, how to classify number of qubits, registers, and measurement

4. Composition - Logical GATE (reverse OCR)
Make text representation FULL diagram from circuit here
</thought>

Then output the full OpenQASM 2.0 code block in this format:
<OPENQASM code>
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Layers 1
h q[0];
h q[1];

// Layers n
...

// Measurement
measure q[0] -> c[0];
measure q[1] -> c[1];
</OPENQASM code>

Output only start with <thought> .. </thought> and end with <OPENQASM code> ... </OPENQASM code> sections for the circuit. 
Do not explain, summarize, or add commentary outside these tags. also must include 'include "qelib1.inc";' after OPENQASM 2.0; inside <OPENQASM code> ... </OPENQASM code> section.
Make sure the open qasm code is working and there is no wrong syntax like needed the end of argument list error (example: for(i = 0; i < 7; ++i) h q[i]; => please run this code to generate the right syntax). make sure the code is open qasm 2.0 compatible and not other code.

IMPORTANT:
- MAKE SURE THE DEPTH IS CORRECT. “Depth” here refers to the number of **logical gate layers**, **NOT** the number of individual gate lines.
    * Multiple gates on different qubits in the same time-step count as **one layer**.
    * Example of 1 depth layer:
        h q[0];
        h q[1];
    * This is NOT two layers, it is **one** layer because gates happen in parallel.
    * If the circuit contains:
        h q[0];
        cx q[0], q[1];
        h q[1];
      → This counts as **3 depth layers**, because gates happen in sequence.
- Your goal is to ensure that the circuit has exactly {depth} **sequential gate layers** (excluding measure/barrier).
- DO NOT use any for-loops. Write out every gate explicitly, one per line.
- DO NOT use any gates that are not supported by OpenQASM 2.0, such as `cu1`, `u1`, `u2`, or `u3`. Only use gates from "qelib1.inc" like h, x, cx, cz, rz, sx, and measure.
- DO NOT write pseudocode. ONLY produce real, working OpenQASM 2.0 syntax that can run in Qiskit or IBM Q runner without modification.
- All code must be fully expanded. No loops, no macros, no `for`, `while`, or similar.
- Make sure measurement on all qubits

When using rotation gates like rz or rx, always write them as: rz(angle) qubit;
Example: rz(pi/2) q[0];
Do NOT write: rz(q[0], pi/2); ← this is invalid in OpenQASM 2.0

When declaring classical registers, make sure the number of classical bits matches the number of measurements.
For example, if measuring 2 qubits:
    creg c[2];
    measure q[0] -> c[0];
    measure q[1] -> c[1];

Do NOT write creg c[0]; unless no measurements are used.


RECENT FAILED COMPILATION CODE (Ignore if empty):
{additional}

You MUST NOT copy or reuse any pattern or gate order from these previous circuits. Even if they are valid VQE circuits, your output MUST be different in structure, gate order, angles, or control-target layout.

Strictly avoid reproducing ANY of these previously generated code patterns:
{variations}

""")
    
def extract_qasm_block(response: str) -> str:
    start = response.find("<OPENQASM code>")
    end = response.find("</OPENQASM code>")
    if start == -1 or end == -1:
        raise ValueError("Missing <OPENQASM code> block in response")
    return response[start + len("<OPENQASM code>"):end].strip()


type_circuit = "VQE"
folder = f"Images_GPT/{type_circuit}"
trial = 4

# number of qubits (1-10)
# circuit_depth (2-8)

csv_path = os.path.join(folder, f"{type_circuit}_results.csv")
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "type", "qubits", "depth", "response"])

def variation(qubits, depth):
    raw = 0.2 * (
        4 + (100 - 4) *
        ((qubits - 2) / 6) ** 0.6 *
        (depth / 8) ** 2.2
    )
    return min(round(raw), 15)

for n_qubit in range(2, 11):
    for n_depth in range(2, 9):
        num_variation = variation(n_qubit, n_depth)
        recent_variations = []
        for var in range(num_variation):
            failed_qasms = []
            for trial_number in range(trial):
                additional_qasm = "\n---\n".join(failed_qasms[-3:])
                variations_text = "\n---\n".join(recent_variations[-3:])
                try:
                    prompt = get_prompt(n_qubit, n_depth, additional=additional_qasm)
                    response = run_inference_lm(prompt, max_tokens=3500, temperature=0.1)
                    response_qasm = extract_qasm_block(response)

                    try:
                        qc = QuantumCircuit.from_qasm_str(response_qasm)
                    except Exception as parse_err:
                        print(f"[QASM Parse Error] Trial {trial_number}: {type(parse_err).__name__} - {parse_err}")
                        print(f"[QASM Dump] ---\n{response_qasm[40:200]}\n---")
                        failed_qasms.append(f"{parse_err} --- {response_qasm[40:300]}")
                        continue

                    # Validation after parsing
                    is_valid, actual_qubits, actual_depth = validate_qc(qc, n_qubit, n_depth)
                    if not is_valid:
                        missmatch = f"[Validation Warning] Depth/Qubit mismatch: {actual_qubits} qubits, {actual_depth} layers"
                        print(missmatch)
                        failed_qasms.append(f"{missmatch} --- {response_qasm[40:300]}")

                    # Save image
                    fig = circuit_drawer(qc, output='mpl')
                    unique_id = uuid.uuid4().hex[:3]
                    img_path = f"{folder}/{type_circuit}_{n_qubit}_{n_depth}_{var}_{trial_number}_{unique_id}.png"

                    fig.savefig(img_path)

                    # Save metadata
                    with open(csv_path, mode="a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([img_path, type_circuit, actual_qubits, actual_depth, response])

                    status = "✅" if is_valid else "⚠️"
                    print(f"{status} Saved: {img_path}")
                    recent_variations.append(response_qasm)
                    if not is_valid:
                        continue
                    else:
                        break

                except Exception as e:
                    print(f"[Gen Error] Trial {trial_number}: {type(e).__name__} - {e}")
                    torch.cuda.empty_cache()
                    continue

