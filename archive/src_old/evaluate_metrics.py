import pandas as pd
import re

def extract_parameters(text):
    if not isinstance(text, str):
        return None
        
    # If there are thinking tags, extract the text after them
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    
    # Regex designed to capture all the target variables robustly.
    # We use non-greedy matches (.*?) for text and capture groups for variables.
    pattern = re.compile(
        r"This is a\s+(.+?)\s+state with\s+(.+?)\s*(?:equal to|≈|=|is)\s*([\d\.-]+),?\s*represented using a truncated Hilbert space of dimension d = (\d+), in the linear space from -(\d+) to \d+\.? Under a compact binary encoding, this truncated space could be represented using ⌈log(?:₂|2)\(\d+\)⌉ = (\d+) qubits",
        re.IGNORECASE | re.DOTALL
    )
    
    match = pattern.search(text)
    if match:
        try:
            state_type = match.group(1).strip().lower()
            param_name = match.group(2).strip().lower()
            param_value = float(match.group(3))
            dim_d = int(match.group(4))
            linear_space = int(match.group(5))
            qubits = int(match.group(6))
            
            return {
                "state_type": state_type,
                "param_name": param_name,
                "param_value": param_value,
                "dimension": dim_d,
                "linear_space": linear_space,
                "qubits": qubits,
            }
        except Exception as e:
            return None
    return None

def evaluate(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")
    
    metrics = {
        "state_type": {"correct": 0, "total": 0},
        "param_name": {"correct": 0, "total": 0},
        "param_value": {"correct": 0, "total": 0},
        "dimension": {"correct": 0, "total": 0},
        "linear_space": {"correct": 0, "total": 0},
        "qubits": {"correct": 0, "total": 0},
        "perfect_match": {"correct": 0, "total": 0}, # all fields correct
        "parse_success_gen": 0,
        "parse_success_gt": 0,
        "failed_gen_texts": []
    }
    
    for idx, row in df.iterrows():
        gen_text = row.get("generated", "")
        gt_text = row.get("ground_truth", "")
        
        gen_data = extract_parameters(gen_text)
        gt_data = extract_parameters(gt_text)
        
        if gen_data is not None:
            metrics["parse_success_gen"] += 1
        else:
            if isinstance(gen_text, str) and len(gen_text) > 5:
                 metrics["failed_gen_texts"].append(f"Row {idx+1}:\n{gen_text}\n---")
                 
        if gt_data is not None:
            metrics["parse_success_gt"] += 1
            
        # We only evaluate accuracy if the ground truth can be successfully parsed.
        # If ground truth parses, but generation fails to parse, we count it as 0 correct.
        if gt_data:
            all_correct = True
            for key in ["state_type", "param_name", "param_value", "dimension", "linear_space", "qubits"]:
                metrics[key]["total"] += 1
                
                # If gen_data failed to parse, the answer for this key is wrong
                if not gen_data:
                    all_correct = False
                    continue
                
                # For string values (state type, param name)
                if isinstance(gen_data[key], str):
                    if gen_data[key] == gt_data[key]:
                        metrics[key]["correct"] += 1
                    else:
                        all_correct = False
                        
                # For float values (param value can have small drift)
                elif isinstance(gen_data[key], float):
                    if abs(gen_data[key] - gt_data[key]) < 1e-4:
                        metrics[key]["correct"] += 1
                    else:
                        all_correct = False
                        
                # For integer values (dimension, linear space, qubits)
                else: 
                    if gen_data[key] == gt_data[key]:
                        metrics[key]["correct"] += 1
                    else:
                        all_correct = False
            
            metrics["perfect_match"]["total"] += 1
            if all_correct:
                metrics["perfect_match"]["correct"] += 1
                
    # Print results summary
    print("=== Parsing Success Rate ===")
    print(f"Ground Truth (Target format) parse rate: {metrics['parse_success_gt']}/{len(df)} ({(metrics['parse_success_gt']/len(df))*100:.1f}%)")
    print(f"Generated (Model prediction) parse rate: {metrics['parse_success_gen']}/{len(df)} ({(metrics['parse_success_gen']/len(df))*100:.1f}%)\n")
    
    print("=== Accuracy Metrics (Compared against Parsable Ground Truths) ===")
    for key in ["state_type", "param_value", "dimension", "linear_space", "qubits", "perfect_match"]:
        data = metrics[key]
        if data["total"] > 0:
            acc = (data["correct"] / data["total"]) * 100
            print(f"{key.ljust(15)}: {data['correct']:>3}/{data['total']:<3} => {acc:.1f}%")
        else:
            print(f"{key.ljust(15)}: N/A (no matched GTs)")

    if metrics["failed_gen_texts"] and metrics["parse_success_gen"] < len(df):
        print("\n=== Examples of unparsable generated outputs ===")
        # Print up to 2 examples of failed regex parses so you can see why they failed
        for text in metrics["failed_gen_texts"][:2]:
            print(text)

if __name__ == '__main__':
    csv_file = "inference-result/qwen-wigner-3vl-8b-v2444-new-prompt-v2.csv"
    try:
        evaluate(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}. Please ensure inference calculation has finished.")
