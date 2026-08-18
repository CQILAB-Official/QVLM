import re
import pandas as pd

# ================================
# Load CSV
# ================================
base_path = "../../../../"
goto_folder = "ResultGroup/1.Wigner/"
filename = "WignerRefactor-QuantumVLM-3VL-8B-v2.csv"
print("[INFO] Loading inference CSV...")
df = pd.read_csv(f"{base_path}{goto_folder}{filename}")
output_file = "../2.Output/v2/4.Converted-Wigner-QuantumVLM-3VL-8B-v2.csv"

# Universal float pattern
FLOAT_PATTERN = r"(\d+(?:\.\d+)?)"

# ================================
# Extraction helper for ground truth (concise format)
# ================================
def extract_fields_gt(text):
    """Extract fields from ground truth - concise format"""
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

    # Extract dimension from "dimension d = X"
    dimension_match = re.search(r"dimension\s*d\s*=\s*(\d+)", text)
    
    # Extract qubits from format "⌈log₂(X)⌉ = Y qubits"
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
# Extraction helper for predicted (verbose step-by-step format)
# ================================
def extract_fields_predicted(text):
    """Extract fields from predicted - verbose step-by-step format with Answer/Conclusion section"""
    
    if not isinstance(text, str):
        text = str(text)
        
    # Check for </think> and take everything after it
    last_think_idx = text.rfind("</think>")
    if last_think_idx != -1:
        answer_section = text[last_think_idx + len("</think>"):].strip()
    else:
        # PRIORITIZE the "Answer" section (with or without colon) as requested by user
        # Fallback to "Conclusion" if Answer doesn't exist
        # Pattern matches both: "**Answer**" (paragraph) and "**Answer:**" (inline)
        # Use [*] instead of \* for more reliable matching of asterisks
        answer_match = re.search(r"[*][*]Answer:?[*][*](.+)", text, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            # Try alternative format where colon is inside: "**Answer:** text"
            answer_match = re.search(r"[*][*]Answer[*][*]:(.+)", text, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            answer_section = answer_match.group(1)
        else:
            # Fallback: try Conclusion section
            conclusion_match = re.search(r"[*][*]Conclusion[*][*](.+?)(?:[*][*]Answer[*][*]|$)", text, re.DOTALL | re.IGNORECASE)
            if conclusion_match:
                answer_section = conclusion_match.group(1)
            else:
                answer_section = text
    
    # Remove bold markdown formatting (**text**) for easier extraction
    answer_cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_section)
    
    # Extract state type - looking for patterns like "This is a cat state" or "Coherent state with" or just "Fock state"
    state_match = re.search(r"(?:this is a |it (?:appears to )?(?:is |be )?a )?(?P<state>cat|thermal|coherent|fock|random|number)(?:\s+mixed)?\s+state",
                            answer_cleaned, re.IGNORECASE)
    
    # Extract N (number of photons) - multiple patterns, ordered by specificity
    # Pattern 1: "N = 15" or "with N = 15" or "N ≈ 15"
    photon_n_match = re.search(r"(?:with\s+)?N\s*(?:≈|=)\s*(\d+(?:\.\d+)?)", answer_cleaned, re.IGNORECASE)
    # Pattern 2: "number of photons = 9"
    photon_of_match = re.search(r"number\s+of\s+photons\s*(?:≈|=|:)\s*(\d+(?:\.\d+)?)", answer_cleaned, re.IGNORECASE)
    # Pattern 3: "with 26 photons"
    photon_with_match = re.search(r"with\s+(\d+)\s+photons?", answer_cleaned, re.IGNORECASE)
    # Pattern 4: "15 photons"
    photon_number_match = re.search(r"(?<!with\s)(\d+)\s+photons?(?:\s|,|$|and)", answer_cleaned, re.IGNORECASE)
    
    # Extract α (alpha) - looking for "alpha = 15" or "α ≈ 4"
    alpha_match = re.search(r"(?:α|alpha)\s*(?:≈|=)\s*(\d+(?:\.\d+)?)", answer_cleaned, re.IGNORECASE)
    
    # Extract density - looking for "density = 0.1" or "number density value ≈ 0.5"
    density_match = re.search(r"(?:number\s+)?(?:with\s+)?density(?:\s*value)?\s*(?:≈|=)\s*(\d+(?:\.\d+)?)", answer_cleaned, re.IGNORECASE)
    
    # Extract d (dimension) - multiple patterns for inclusivity
    dimension_d_match = re.search(r"(?:dimension\s+)?d\s*(?:≈|=)\s*(\d+)", answer_cleaned, re.IGNORECASE)
    dimension_of_match = re.search(r"(?:truncated\s+Hilbert\s+space\s+)?dimension\s+of\s+(\d+)", answer_cleaned, re.IGNORECASE)
    
    # Extract linear space range - multiple patterns
    linear_from_to_match = re.search(r"(?:in\s+the\s+)?linear\s+space\s+(?:range\s+)?from\s+-?(\d+)\s*to\s+-?(\d+)", answer_cleaned, re.IGNORECASE)
    linear_range_of_match = re.search(r"linear\s+space\s+range\s+of\s+(?:approximately\s+)?(\d+)", answer_cleaned, re.IGNORECASE)
    
    # Extract qubits - looking for "⌈log2(9)⌉ = 4" or "Using 4 qubits"
    qubits_match = re.search(r"⌈log[₂2]\(\d+\)⌉\s*(?:≈|=)\s*(\d+)", answer_cleaned, re.IGNORECASE)
    if not qubits_match:
        qubits_match = re.search(r"(?:using|represented\s+using)\s+(\d+)\s*qubits?", answer_cleaned, re.IGNORECASE)
    
    # Determine parameter based on state type or available values
    # Priority: alpha > N explicit > number of photons > with X photons > photon number > density
    param = None
    if alpha_match:
        param = float(alpha_match.group(1))
    elif photon_n_match:
        param = float(photon_n_match.group(1))
    elif photon_of_match:
        param = float(photon_of_match.group(1))
    elif photon_with_match:
        param = float(photon_with_match.group(1))
    elif photon_number_match:
        param = float(photon_number_match.group(1))
    elif density_match:
        param = float(density_match.group(1))
    
    # Determine dimension - prefer explicit d= format, fallback to "dimension of"
    dimension = None
    if dimension_d_match:
        dimension = int(dimension_d_match.group(1))
    elif dimension_of_match:
        dimension = int(dimension_of_match.group(1))
    
    # Determine linear space - prefer "from X to Y", fallback to "range of X"
    linear_space = None
    if linear_from_to_match:
        linear_space = abs(int(linear_from_to_match.group(2)))
    elif linear_range_of_match:
        linear_space = int(linear_range_of_match.group(1))
    
    return {
        "state": state_match.group("state").lower() if state_match else None,
        "parameter": param,
        "dimension": dimension,
        "qubits": int(qubits_match.group(1)) if qubits_match else None,
        "linear_space": linear_space
    }

# ================================
# Process each row
# ================================
rows = []

for idx, row in df.iterrows():
    # Use different extractors for generated (predicted) vs ground_truth
    gen = extract_fields_predicted(row["generated"])
    gt  = extract_fields_gt(row["ground_truth"])

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

print(f"[INFO] Saved to {output_file}")
print(f"[INFO] Processed {len(rows)} rows")
