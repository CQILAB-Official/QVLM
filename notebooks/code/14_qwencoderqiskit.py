# !pip install llama-cpp-python

import textwrap
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="Qiskit/Qwen2.5-Coder-14B-Qiskit-GGUF",
	filename="Qwen2.5-Coder-14B-Qiskit.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=60
)

def get_prompt(qubit, depth, additional="", variations=""):
    return textwrap.dedent(f"""\
DO NOT SUMMARIZE, DO NOT OUTPUT CHINESE, STAY IN CONTEXT. ONLY OUTPUT RAW TAGGED BLOCKS.
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

2. Number of qubits, how many quantum classical registers, is there measurement, reverse engineering from image to thinking process, how to classify number of qubits, registers, and measurement

3. Composition - Logical GATE (reverse OCR)
Make text representation FULL diagram from circuit here
</thought>

Then output the full OpenQASM 2.0 code block in this format:
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

RECENT VARIATIONS that already generated (Make sure different from these):
{variations}
""")

prompt = get_prompt(4, 5)
output = llm(
	prompt,
	max_tokens=4096,
	echo=True
)
print(output)