import pandas as pd

# Input and output file paths
input_file = "case_study_circuit.csv"
output_file = "case_study_circuit_patched.csv"

# Load the CSV
df = pd.read_csv(input_file)

# Optimized instruction text
patched_value = 'Provide the full QASM code for this quantum circuit. The response must be only valid QASM code that runs directly in Qiskit, starting with include "qelib1.inc".'

# Patch all rows in column "input_5"
if "input_5" in df.columns:
    df["input_5"] = patched_value
else:
    raise KeyError("Column 'input_5' not found in CSV.")

# Save the patched CSV
df.to_csv(output_file, index=False)

print(f"Patched CSV saved as {output_file}")
