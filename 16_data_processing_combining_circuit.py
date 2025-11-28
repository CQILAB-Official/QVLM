

QUESTION_1 = [
    "Describe and analyze the quantum circuit.",
    "Describe and analyze the quantum circuit.",
    "Explain the structure and function of the quantum circuit in detail.",
    "Provide a thorough interpretation and breakdown of the quantum circuit.",
    "Describe the logical flow and purpose of operations within the quantum circuit.",
    "Analyze the circuit’s design and explain how it transforms quantum states.",
    "Give a structured explanation of the circuit and assess its behavior.",
    "Describe how the quantum circuit operates and what kind of state it prepares.",
    "Walk through the circuit’s steps and discuss the role of each component.",
    "Offer a full technical interpretation of the circuit’s construction and impact.",
    "Explain the overall design and effect of the quantum operations in the circuit.",
    "Break down the sequence of gates and describe the circuit’s intended outcome.",
    "Describe the transformation applied by the circuit from initialization to measurement.",
    "Provide a detailed gate-level analysis of the circuit and the resulting quantum state.",
    "Interpret the entire quantum circuit and explain what it achieves.",
    "Give a comprehensive explanation of how this circuit prepares or manipulates quantum information.",
    "Analyze the intent and structure of the quantum circuit stage by stage.",
    "Describe the full function of the circuit and its impact on qubit entanglement.",
    "Explain how the circuit structure leads to the final measured outcome.",
    "Evaluate the circuit's components and how they collectively realize a quantum task.",
    "Explain each gate's role in the circuit and how it contributes to the final state.",
    "Describe the circuit’s operation and analyze the quantum logic behind its construction."
]

QUESTION_2 = [
    "Can you identify the quantum circuit?",
    "Can you identify the quantum circuit? Explain how",
    "Explain the identity of the quantum circuit.",
    "Explain which known quantum circuit is represented here.",
    "Explain what kind of circuit this is.",
    "Explain which named quantum operation this circuit corresponds to.",
    "Explain what standard circuit this matches.",
    "Explain which common quantum state or algorithm this circuit prepares.",
    "Explain how this circuit would be classified.",
    "Explain what this circuit is typically called.",
    "Explain the type of circuit shown in the diagram.",
    "Explain which textbook circuit this structure aligns with.",
    "Can you identify this quantum circuit by name?",
    "Identify this circuit using standard terminology.",
    "What is the correct label for this quantum circuit?",
    "Explain whether this circuit matches a well-known pattern.",
    "Give the name of this circuit based on its layout.",
    "Can this circuit be mapped to a standard quantum protocol?",
    "What named circuit does this configuration reflect?",
    "Explain what quantum primitive is being implemented.",
    "Does this circuit correspond to a standard quantum task? If so, which?",
    "State the type of quantum circuit based on its structure."
]

QUESTION_3 = [
    "Identify number of qubits and parameters.",
    "Identify the number of qubits and parameters used in the circuit.",
    "State how many qubits and parameters are present.",
    "Explain how many qubits and parameterized gates are in the circuit.",
    "Determine the number of qubits and parameters involved.",
    "Give the total number of qubits and any parameter values used.",
    "Count the qubits and list any parameters used in gates.",
    "How many qubits and tunable parameters are in the circuit?",
    "Report the qubit count and whether the circuit includes parameters.",
    "List the number of qubits and any adjustable parameters.",
    "Specify the number of quantum bits and parameterized operations.",
    "State the qubit count and indicate if parameters are present.",
    "Indicate the total qubits and the presence of any variables.",
    "Explain how many quantum wires and gate parameters exist.",
    "What is the number of qubits and how many gates are parameterized?",
    "Identify how many qubits the circuit uses and whether it includes any parameters."
]

QUESTION_4 = [
    "What is the general composition of the quantum circuits?",
    "What is the general composition of the quantum circuits?",
    "Explain the overall structure and layout of the quantum circuit.",
    "Describe the main components and layers of the circuit.",
    "What is the general gate composition in this circuit?",
    "Summarize the types of gates and their arrangement in the circuit.",
    "Give an overview of how the circuit is composed.",
    "Explain how the gates are structured throughout the circuit.",
    "What is the typical composition pattern used here?",
    "Describe the sequence of gate operations in the circuit.",
    "List the main gate types and their ordering in the circuit.",
    "Explain the logic structure used to build this circuit.",
    "What kinds of operations are applied and how are they arranged?",
    "Outline the circuit’s construction in terms of gate categories.",
    "What is the layer-by-layer composition of this quantum circuit?",
    "Describe how the circuit is organized across its depth.",
    "Explain the gate flow and composition style in this circuit."
]

QUESTION_5 = [
    "Give me the QASM code from the given circuit.",
    "Provide the QASM code for the circuit shown.",
    "Generate the OpenQASM code that corresponds to this circuit.",
    "Provide the full QASM code for this quantum circuit.",
    "Write the QASM 2.0 representation of the circuit.",
    "Convert the circuit into its OpenQASM code format.",
    "Give the complete OpenQASM output for this circuit.",
    "Output the QASM version of the shown circuit.",
    "Return the QASM source code for this circuit diagram.",
    "Translate the circuit into valid QASM 2.0 code.",
    "Write out the QASM code that builds this circuit.",
    "Show the QASM implementation matching this quantum circuit."
]

QUESTION_6 = "What is the type of circuit? Only answer in one word (Eg. GHz, Grover, etc.)."
QUESTION_7 = "What is number of qubit in circuit? Only answer in single number (Eg. 2, 3, 4, etc.)."
QUESTION_8 = "Is this random circuit engtangled? Only answer in Yes or No."
QUESTION_9 = "What is the purity of the circuit? Only answer in single number (Eg. 0.5, 0.7, etc.)."

LIST_OF_DATASETS = [
    "QuantumCircuit_VQA/GHZ/GHZ_results.csv",
    "QuantumCircuit_VQA/Grover/Grover_results.csv",
    "QuantumCircuit_VQA/QAOA/QAOA_results.csv",
    "QuantumCircuit_VQA/QFT/QFT_results.csv",
    "QuantumCircuit_VQA/QPE/QPE_results.csv",
    "QuantumCircuit_VQA/Random/Random_results.csv",
    "QuantumCircuit_VQA/VQE/VQE_results.csv"
]

DATASET_ENTANGLEMENT_DETECTION = "QuantumCircuit_VQA/Random_Entanglement/Random_entanglement_results.csv"

import os, re, random, shutil, pandas as pd, itertools

backup_dir = "backup"
os.makedirs(backup_dir, exist_ok=True)

def _section(txt: str, n: int) -> str:
    pat = rf"(?:^|\n)\s*{n}\.\s*(.*?)" \
          rf"(?=\n\s*{n+1}\.\s|<OPENQASM|</thought>|$)"
    m = re.search(pat, txt, flags=re.S)
    return m.group(1).strip() if m else ""


def _qasm(txt):
    m = re.search(r"<OPENQASM code>(.*?)</OPENQASM code>", txt, flags=re.S)
    return m.group(1).strip() if m else ""

def _cycler(lst):
    while True:
        random.shuffle(lst)
        for x in lst:
            yield x

cycles = [_cycler(L) for L in (QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5)]

rows, hdr = [], ["image"] + sum([[f"input_{i}", f"output_{i}"] for i in range(1, 8)], [])

for path in LIST_OF_DATASETS:
    shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        resp = str(r["response"])
        outs = [_section(resp, i) for i in range(1, 5)] + [_qasm(resp), str(r["type"]), str(r["qubits"])]
        ins  = [next(c) for c in cycles] + [QUESTION_6, QUESTION_7]
        row  = [r["image"]] + list(itertools.chain.from_iterable(zip(ins, outs)))
        rows.append(row)

pd.DataFrame(rows, columns=hdr).to_csv("case_study_circuit.csv", index=False)

# Entangflement detection dataset
# build case_study_entanglement.csv (fixed order: Q8 → Yes/No, Q9 → purity)
shutil.copy2(
    DATASET_ENTANGLEMENT_DETECTION,
    os.path.join(backup_dir, os.path.basename(DATASET_ENTANGLEMENT_DETECTION))
)

ent_df = pd.read_csv(DATASET_ENTANGLEMENT_DETECTION)
ent_rows = [
    [
        r["image"],
        QUESTION_8,
        "Yes" if int(r["entangled"]) else "No",
        QUESTION_9,
        str(r["purity"])
    ]
    for _, r in ent_df.iterrows()
]

pd.DataFrame(
    ent_rows,
    columns=["image", "input_1", "output_1", "input_2", "output_2"]
).to_csv("case_study_entanglement.csv", index=False)
