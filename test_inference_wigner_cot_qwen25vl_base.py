import os
import torch
from unsloth import FastVisionModel
from PIL import Image
import pandas as pd
import random  
from sklearn.model_selection import train_test_split


# Globals for holding the loaded model and processor
MODEL = None
TOKENIZER = None


def find_checkpoint(checkpoint_root: str = "model_cot_qwen3vl_32b_4bit_thinking_v1", specific_checkpoint: str = "checkpoint-2814") -> str:
    if specific_checkpoint:
        checkpoint_path = os.path.join(checkpoint_root, specific_checkpoint)
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        else:
            raise ValueError(f"Checkpoint {specific_checkpoint} not found in {checkpoint_root}")
    
    checkpoints = [
        d for d in os.listdir(checkpoint_root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_root}")

    # Sort by the numeric portion after "checkpoint-"
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints_sorted[-1]
    return os.path.join(checkpoint_root, highest_checkpoint)


def initialize_model(model_id: str, checkpoint_root: str = "model_cot_qwen3vl_32b_4bit_thinking_v1", specific_checkpoint: str = "checkpoint-2814"):
    global MODEL, TOKENIZER

    # If already loaded, just return
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER
    
    adapter_path = find_checkpoint(checkpoint_root, specific_checkpoint)
    
    print("Loading base model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name =  model_id,  # Trained model either locally or from huggingface
        load_in_4bit = True,
    )
    print("Base model loaded.")

    # 2. Find highest checkpoint

    MODEL = model.to("cuda")
    TOKENIZER = tokenizer

    return MODEL, TOKENIZER


def run_inference_qwenvl(image: Image.Image, user_input: str, temperature: float = 0.0, 
                        max_tokens: int = 500, model_id: str = "unsloth/Qwen2-VL-7B-Instruct") -> str:
    
    model, tokenizer = initialize_model(model_id)
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
        use_cache=True,
        temperature=temperature,
        min_p=0.1
    )
    generate_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return generated_text

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
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Drop rows with NaN and reset indices

    test_data  = test_data.reset_index(drop=True)
    
    
    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state etc.) "
        "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
        "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
    )

    # Prepare your train_data
    x_test   = test_data['image'][:]
    images    = test_data['image'][:]
    y_test   = test_data['ground_truth'][:]
    prompts   = BEST_PROMPT

    for i in range(len(x_test)):
        image = Image.open(images[i])
        image = image.resize((1000, 600))
        user_input = BEST_PROMPT + x_test[i]
        temperature = 1.5
        max_tokens = 1024
        model_id = "unsloth/Qwen2.5-VL-32B-Instruct"

        generated_text = run_inference_qwenvl(image, user_input, temperature, max_tokens, model_id)
        # generate print for the number of the test case and the generated textn result of generated_text and ground_truth
        print(f"Test case {i+1}")
        print(f"Generated text: {generated_text}")
        print(f"Ground truth: {y_test[i]}")
        print("\n")