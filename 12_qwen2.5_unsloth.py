import os
import textwrap
import threading
import uuid
import torch
import csv
import os
from unsloth import FastLanguageModel
from PIL import Image
from unsloth.chat_templates import get_chat_template
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
# from src.utils import find_highest_checkpoint
# from transformers import TextIteratorStreamer
from qiskit.circuit.instruction import Instruction
from qiskit.converters import circuit_to_dag
from qiskit.converters import circuit_to_dag

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

# def validate_qc(qc: QuantumCircuit, expected_qubits: int, expected_depth: int):
#     actual_qubits = qc.num_qubits
#     actual_layers = count_logical_gate_layers(qc) #depth_exclude_measure(qc) 
#     if actual_qubits != expected_qubits:
#         raise ValueError(f"Expected {expected_qubits} qubits but got {actual_qubits}")
#     if actual_layers != expected_depth:
#         raise ValueError(f"Expected {expected_depth} gate layers but got {actual_layers}")

def validate_qc(qc: QuantumCircuit, expected_qubits: int, expected_depth: int) -> tuple[bool, int, int]:
    actual_qubits = qc.num_qubits
    actual_depth = depth_exclude_measure(qc)

    if actual_qubits != expected_qubits or actual_depth != expected_depth:
        return False, actual_qubits, actual_depth

    return True, actual_qubits, actual_depth


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
        load_in_4bit=True,
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
    elif "qwen" in model_id_lower:
        template_name = "qwen2.5"
    # elif "granite" in model_id_lower:
    #     template_name = "granite-3.2"
    # elif "deepseek" in model_id_lower and "qwen" in model_id_lower:
    #     template_name = None
    
    # row_json = [{"role": "user", "content": user_input}]
    # formatted_text = tokenizer.apply_chat_template(
    #     row_json,
    #     tokenize=False,
    #     add_generation_prompt=False
    # )

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


def run_inference_lm(user_input: str, temperature: float = 1.0, max_tokens: int = 1500, model_id: str = "unsloth/Phi-3.5-mini-instruct") -> str:
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
        do_sample=True,
        repetition_penalty=1.1,
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

def get_prompt(qubit, depth, additional="", variations=""):
    return textwrap.dedent(f"""\
DO NOT SUMMARIZE, DO NOT OUTPUT CHINESE. ONLY OUTPUT RAW TAGGED BLOCKS.
You MUST NOT copy or reuse any pattern or gate order from these previous circuits. Even if they are valid QPE circuits, your output MUST be different in structure, gate order, angles, or control-target layout.

Generate random **Quantum Phase Estimation (QPE)** (For the circuit, strictly follow the QPE structure!) circuit with depth {depth} (MAKE SURE DEPTH IS CORRECT) and number of qubits {qubit} (Qiskit qasm), using logical gate, use only minimal classical register for measurement
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
2. Given the circuit, how you identify it as Quantum Phase Estimation (QPE)?

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
- Make sure measurement on all qubits
- DO NOT use any for-loops. Write out every gate explicitly, one per line.
- DO NOT use any gates that are not supported by OpenQASM 2.0, such as `cu1`, `u1`, `u2`, or `u3`. Only use gates from "qelib1.inc" like h, x, cx, cz, rz, sx, and measure.
- DO NOT write pseudocode. ONLY produce real, working OpenQASM 2.0 syntax that can run in Qiskit or IBM Q runner without modification.
- All code must be fully expanded. No loops, no macros, no `for`, `while`, or similar.

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
""")
    
def extract_qasm_block(response: str) -> str:
    start = response.find("<OPENQASM code>")
    end = response.find("</OPENQASM code>")
    if start == -1 or end == -1:
        raise ValueError("Missing <OPENQASM code> block in response")
    return response[start + len("<OPENQASM code>"):end].strip()


type_circuit = "QPE"
model_id = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
folder = f"Images/{type_circuit}"
trial = 6

# number of qubits (1-10)
# circuit_depth (2-8)

csv_path = os.path.join(folder, f"{type_circuit}_results.csv")
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "type", "qubits", "depth", "response"])

def variation(qubits, depth):
    return round(
        4 + (100 - 4) *
        ((qubits - 2) / 6) ** 0.6 *
        (depth / 8) ** 2.2
    )

for n_qubit in range(2, 11):
    for n_depth in range(2, 9):
        num_variation = variation(n_qubit, n_depth)
        recent_variations = []
        for var in range(num_variation):
            failed_qasms = []
            for trial_number in range(trial):
                additional_qasm = "\n---\n".join(failed_qasms[-3:])
                variations_text = "\n---\n".join(recent_variations[-4:])
                try:
                    prompt = get_prompt(n_qubit, n_depth, additional=additional_qasm)
                    response = run_inference_lm(prompt, model_id=model_id, max_tokens=4500, temperature=0.5)
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
                        print(f"[Validation Warning] Depth/Qubit mismatch: {actual_qubits} qubits, {actual_depth} layers")
                        failed_qasms.append(response_qasm[40:300])

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

