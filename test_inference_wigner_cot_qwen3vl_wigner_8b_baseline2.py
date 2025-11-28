import os
import torch
from unsloth import FastVisionModel
from PIL import Image
import pandas as pd
import random  
from sklearn.model_selection import train_test_split
import csv

# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


# def find_highest_checkpoint(checkpoint_dir: str) -> str:
#     checkpoints = [
#         d for d in os.listdir(checkpoint_dir)
#         if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
#     ]
#     if not checkpoints:
#         raise ValueError(f"No checkpoints found in {checkpoint_dir}")

#     # Sort by the numeric portion after "checkpoint-"
#     checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
#     highest_checkpoint = checkpoints_sorted[-1]
#     return os.path.join(checkpoint_dir, highest_checkpoint)

def find_checkpoint(checkpoint_dir: str, specific_checkpoint: str = None) -> str:
    if specific_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, specific_checkpoint)
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        else:
            raise ValueError(f"Checkpoint {specific_checkpoint} not found in {checkpoint_dir}")

    # Fallback: find the highest checkpoint
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints_sorted[-1]
    return os.path.join(checkpoint_dir, highest_checkpoint)


def initialize_model(model_id: str, checkpoint_root: str = "./model_cot_qwen25vl_32b_v2", specific_checkpoint: str = None):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    # adapter_path = find_checkpoint(checkpoint_root, specific_checkpoint)
    # print(f"Highest checkpoint found: {adapter_path}")
    
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name =  model_id,  # Trained model either locally or from huggingface
        # load_in_4bit = True,
        
        # NEW: shard across available GPUs automatically
        # device_map = "auto",
    )
    print("Base model loaded.")

    # 2. Find highest checkpoint

    MODEL = model
    TOKENIZER = tokenizer

    return MODEL, TOKENIZER


def run_inference_qwenvl(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "unsloth/Qwen2-VL-7B-Instruct", checkpoint_root: str = "model_cot_qwen3vl_32b_4bit_thinking_v1", specific_checkpoint: str = "checkpoint-2814") -> str:

    model, tokenizer = initialize_model(model_id, checkpoint_root, checkpoint)
    FastVisionModel.for_inference(model) 
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": user_input
                },
            ]
        }
    ]
    # Tokenize prompt using the built-in chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=False,
        temperature=temperature,
        min_p=0.1
    )
    generate_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return generated_text

def csv_to_ft_lists(csv_path, test_frac=0.1, seed=42):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    return train_df.to_dict("records"), test_df.to_dict("records")

if __name__ == "__main__":
    CSV_FILENAME = 'Dataset/wigner_analysis_results_combined.csv' 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['image', 'ground_truth']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Filter out data you don't want (e.g., remove type "Number state")
    # data = data[data['type'] != 'Number state']
    
    # Split data into train and test sets
    _, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Drop rows with NaN and reset indices

    test_data  = test_data.reset_index(drop=True)
    
    
    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state etc.) "
        "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
        "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
        "YOU MUST PROVIDE <think></think> BRACKET FOR THE THINKING PROCESS, "
        "YOU SHOULD ANSWER IN THIS EXACT FORMAT IN THE AFTER THE THINKING PROCESS:"
        "*This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].*"
    )

    # Prepare your train_data
    x_test   = test_data['image'][:]
    images    = test_data['image'][:]
    y_test   = test_data['ground_truth'][:]
    prompts   = BEST_PROMPT

    results = []
    
    model_id = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
        
    checkpoint_root = "./model_cot_qwen3vl_32b_4bit_thinking_v1"
    checkpoint = "checkpoint-2814"
    output_file = "inference_results-wigner-baseline-2.csv"

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["test_case", "image", "generated", "ground_truth"])

        for i in range(len(x_test)):
            image = Image.open(images[i]).resize((1000, 600))
            user_input = BEST_PROMPT + x_test[i]
            generated = run_inference_qwenvl(
                image,
                user_input,
                temperature=0.1,
                max_tokens=1500,
                model_id=model_id,
                checkpoint_root=checkpoint_root,
                specific_checkpoint=checkpoint,
            )

            # Write row directly
            writer.writerow([
                i + 1,
                images[i],
                generated,
                y_test[i],
            ])

    print(f"Saved {len(x_test)} rows to inference_results.csv")