import os
import csv
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from unsloth import FastVisionModel

# ──────────────────────────────────────────
# GLOBALS – loaded once
# ──────────────────────────────────────────
MODEL = None
TOKENIZER = None
model_location = "./model_cot_qwen25vl_32b_vfocused"
ckpt_location = "checkpoint-1060"
model_name = "qwen2.5vl"
model_version = "32b-vfocused"
model_id = "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit"

# ──────────────────────────────────────────
# MODEL HELPERS
# ──────────────────────────────────────────
def find_checkpoint(root: str, ckpt: str | None = None) -> str:
    if ckpt:
        path = os.path.join(root, ckpt)
        if os.path.isdir(path):
            return path
        raise ValueError(f"checkpoint {ckpt} not found in {root}")

    cand = [
        d for d in os.listdir(root)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(root, d))
    ]
    if not cand:
        raise ValueError(f"no checkpoints in {root}")
    return os.path.join(root, sorted(cand, key=lambda x: int(x.split("-")[1]))[-1])

def init_model(
    model_root=model_location,
    ckpt=ckpt_location,
    base_id=model_id,
):
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    adapter_path = find_checkpoint(model_root, ckpt)
    print("using adapter:", adapter_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=True,
        device_map = "auto",
    )
    MODEL = model
    TOKENIZER = tokenizer
    FastVisionModel.for_inference(MODEL)
    return MODEL, TOKENIZER

def run_inference(image: Image.Image, prompts: list[str]) -> list[str]:
    model, tok = init_model(model_location, ckpt_location, model_id)
    
    # Create batch messages
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": p},
                ],
            }
        ] for p in prompts
    ]
    
    # Apply template to each message in batch
    texts = [tok.apply_chat_template(msg, add_generation_prompt=True) for msg in messages_batch]
    
    # Repeat image for the batch
    images = [image] * len(prompts)
    
    # Tokenize as a batch
    inputs = tok(images, texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=0.1,
        min_p=0.1,
        use_cache=True,
    )
    
    # Slice the output properly. Since it's a batch, input lengths might differ if padding was used.
    # We use the length of input_ids for each sample to slice.
    input_ids = inputs.input_ids
    generated_ids = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(input_ids, gen_ids)
    ]
    
    return tok.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ──────────────────────────────────────────
# DATA / I/O
# ──────────────────────────────────────────
def process_csv(csv_path: str, max_io: int, seed=42, test_frac=0.20):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    _, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    test_df = test_df.reset_index(drop=True)

    out_file = "ResultUpdate/" + os.path.splitext(csv_path)[0] + f"_{model_name}_inference_{ckpt_location}_{model_version}.csv"
    header = ["image"]
    for i in range(1, max_io + 1):
        header.extend([f"input_{i}", f"prediction_{i}", f"output_{i}"])

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, row in test_df.iterrows():
            print(f"[{csv_path}] Processing row {idx+1}/{len(test_df)}", flush=True)
            img_path = row["image"]
            img = Image.open(img_path).resize((500, 300))
            row_out = [img_path]

            # Collect all prompts first to run in a batch
            prompts = []
            prompt_indices = []

            for n in range(1, max_io + 1):
                input_col = f"input_{n}"
                if input_col in row and not pd.isna(row[input_col]):
                    prompts.append(row[input_col])
                    prompt_indices.append(n)
            
            # Run batched inference if there are any prompts
            preds = []
            if prompts:
                preds = run_inference(img, prompts)

            # Map predictions back to their indices
            pred_map = {idx: pred for idx, pred in zip(prompt_indices, preds)}

            for n in range(1, max_io + 1):
                input_col, out_col = f"input_{n}", f"output_{n}"
                if n in pred_map:
                    row_out.extend([row[input_col], pred_map[n], row[out_col]])
                else:
                    # If this column didn't have an input or was skipped
                    row_out.extend(["", "", ""])

            writer.writerow(row_out)

    print(f"Wrote {len(test_df)} rows to {out_file}")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    # ==== OLD SINGLE‑CSV FLOW (commented out) ====
    """
    CSV_FILENAME = 'Dataset/wigner_analysis_results_combined.csv'
    # previous data-loading code here…
    """

    # ==== NEW TWO‑CSV FLOW ====
    tasks = [
        # ("case_study_entanglement.csv", 2),
        ("case_study_circuit_patched.csv", 7),
    ]
    for path, max_cols in tasks:
        process_csv(path, max_cols)
