import json
import re

notebook_path = '/home/cqimain/1.Riset/1.Result/MetricsGroup/3.Circuit/3.QuantumCircuitClassification-Accuracy-Precision-Recall/3-3-4.classification-circuit-quantumvlm-3vl.ipynb'

def modify_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    
    # Target code block to replace
    target_code = [
        "    with open(file_path, mode='r', encoding='utf-8') as file:\n",
        "        reader = csv.DictReader(file)\n",
        "        for row in reader:\n",
        "            output_6.append(row['output_6'])  # Ground truth\n",
        "            prediction_6.append(row['prediction_6'])  # Predictions\n",
        "    \n",
        "    return output_6, prediction_6\n"
    ]
    
    # Replacement code block
    replacement_code = [
        "    import re\n",
        "    with open(file_path, mode='r', encoding='utf-8') as file:\n",
        "        reader = csv.DictReader(file)\n",
        "        for row in reader:\n",
        "            # Remove <think>...</think> tags and trim whitespace\n",
        "            clean_output = re.sub(r'<think>.*?</think>', '', row['output_6'], flags=re.DOTALL).strip()\n",
        "            output_6.append(clean_output)  # Ground truth\n",
        "            prediction_6.append(row['prediction_6'])  # Predictions\n",
        "    \n",
        "    return output_6, prediction_6\n"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Convert list of strings to single string for easier finding
            # But the structure is a list of lines. 
            
            # Simple approach: iterate through lines and find the block
            # Since the block is contiguous, we can look for the sequence
            
            i = 0
            while i < len(source):
                # check if source[i] matches target_code[0]
                # We need to be careful about exact matches including newlines
                if source[i] == target_code[0]:
                    # Check if the next lines match
                    match = True
                    for j in range(1, len(target_code)):
                        if i + j >= len(source) or source[i+j] != target_code[j]:
                            match = False
                            break
                    
                    if match:
                        print(f"Found match at line {i} in a cell.")
                        # Replace
                        source[i:i+len(target_code)] = replacement_code
                        modified = True
                        break # Move to next cell or continue? 
                        # There might be multiple occurrences in one cell? Unlikely here but safer to continue scanning this cell if we want ALL occurrences.
                        # But loop invalidates index. Better to just break and restart search or handle carefully.
                        # Given the structure, one per cell is expected or one global definition.
                        # The plan noted multiple cells might have it.
                        
                i += 1
                
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1) # minimize diff noise slightly with default indent, but notebook format usually check indentation
        print(f"Successfully modified {path}")
    else:
        print("No matching code block found to replace.")

if __name__ == "__main__":
    modify_notebook(notebook_path)
