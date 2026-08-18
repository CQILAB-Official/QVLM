import re
import pandas as pd

# ================================
# Load CSV
# ================================
base_path = "../../../../"
goto_folder = "ResultGroup/1.Wigner/"
filename = "Wigner-Baseline1-Qwen3VL-v2.csv"
print("[INFO] Loading inference CSV...")
df = pd.read_csv(f"{base_path}{goto_folder}{filename}")
output_file = "../2.Output/1.Converted-Wigner-Baseline1-Qwen3VL-v2.csv"

# Universal float pattern
FLOAT_PATTERN = r"(\d+(?:\.\d+)?)"

# ================================
# Extraction helper
# ================================
def extract_fields(text):
    # Handle "random mixed state" or just "random state"
    state_match = re.search(r"(?P<state>cat|thermal|coherent|fock|random|number)(?:\s+mixed)?\s+state",
                            text, re.IGNORECASE)

    alpha_match   = re.search(r"(?:α|alpha)\s*(?:≈|=)\s*(\d+(?:\.\d+)?)",
                              text, re.IGNORECASE)
    # Handle "number density value" or "with density" or just "density"
    density_match = re.search(r"(?:number\s+)?(?:with\s+)?density(?:\s*value)?\s*(?:≈|=)\s*(\d+(?:\.\d+)?)",
                              text, re.IGNORECASE)
    # Handle "photon number n=X" or "average photons ≈ X" or just "photon = X"
    photon_match  = re.search(r"(?:average\s+)?photon(?:s|\s+number)?\s*(?:n\s*=|≈|=)\s*(\d+(?:\.\d+)?)",
                              text, re.IGNORECASE)

    # NEW: Extract dimension from "dimension d = X"
    dimension_match = re.search(r"dimension\s*d\s*=\s*(\d+)", text)
    
    # UPDATED: Extract qubits from new format "⌈log₂(X)⌉ = Y qubits"
    qubits_match = re.search(r"⌈log₂\(\d+\)⌉\s*=\s*(\d+)\s*qubits", text)
    
    linear_match  = re.search(r"linear space.*?(?:from\s*)?-?(\d+)\s*to\s*-?(\d+)",
                              text, re.IGNORECASE)

    param = None
    if alpha_match:
        param = float(alpha_match.group(1))
    elif density_match:
        param = float(density_match.group(1))
    elif photon_match:
        param = float(photon_match.group(1))

    return {
        "state": state_match.group("state").lower() if state_match else None,
        "parameter": param,
        "dimension": int(dimension_match.group(1)) if dimension_match else None,
        "qubits": int(qubits_match.group(1)) if qubits_match else None,
        "linear_space": abs(int(linear_match.group(2))) if linear_match else None
    }

# ================================
# Process each row
# ================================
rows = []

for idx, row in df.iterrows():
    gen = extract_fields(row["generated"])
    gt  = extract_fields(row["ground_truth"])

    rows.append({
        "test_case": row["test_case"],

        "state_generated": gen["state"],
        "param_generated": gen["parameter"],
        "dimension_generated": gen["dimension"],
        "qubits_generated": gen["qubits"],
        "linear_generated": gen["linear_space"],

        "state_gt": gt["state"],
        "param_gt": gt["parameter"],
        "dimension_gt": gt["dimension"],
        "qubits_gt": gt["qubits"],
        "linear_gt": gt["linear_space"],
    })

# ================================
# Save to CSV
# ================================
out_df = pd.DataFrame(rows)
out_df.to_csv(output_file, index=False)

print("Saved to {output_file}")
