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
    model_root="./model_cot_qwen25vl_32b_v3",
    ckpt="checkpoint-1500",
    base_id="unsloth/Qwen2.5-VL-7B-Instruct",
):
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    adapter_path = find_checkpoint(model_root, ckpt)
    print("using adapter:", adapter_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=True,
        # device_map = "auto",
    )
    MODEL = model.to("cuda")
    TOKENIZER = tokenizer
    FastVisionModel.for_inference(MODEL)
    return MODEL, TOKENIZER

def run_inference(image: Image.Image, prompt: str) -> str:
    model, tok = init_model("./model_cot_qwen25vl_32b_v3", "checkpoint-1500","unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = tok.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tok(image, text, add_special_tokens=False, return_tensors="pt").to("cuda")
    gen_ids = model.generate(
        **{k: v.to("cuda") for k, v in inputs.items()},
        max_new_tokens=1500,
        temperature=0.1,
        min_p=0.1,
        use_cache=False,
    )[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ──────────────────────────────────────────
# DATA / I/O
# ──────────────────────────────────────────
def process_csv(csv_path: str, max_io: int, seed=42, test_frac=0.20):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    _, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    test_df = test_df.reset_index(drop=True)

    out_file = "Result/" + os.path.splitext(csv_path)[0] + "_inference_chk1500_32b-v3.csv"
    header = ["image"]
    for i in range(1, max_io + 1):
        header.extend([f"input_{i}", f"prediction_{i}", f"output_{i}"])

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for _, row in test_df.iterrows():
            img_path = row["image"]
            img = Image.open(img_path).resize((1000, 600))
            row_out = [img_path]

            for n in range(1, max_io + 1):
                input_col, out_col = f"input_{n}", f"output_{n}"
                if input_col not in row or pd.isna(row[input_col]):
                    row_out.extend(["", "", ""])
                    continue

                # Use the input directly as the prompt
                user_prompt = row[input_col]

                pred = run_inference(img, user_prompt)
                row_out.extend([row[input_col], pred, row[out_col]])

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
        ("case_study_entanglement.csv", 2),
        ("case_study_circuit_patched.csv", 7),
    ]
    for path, max_cols in tasks:
        process_csv(path, max_cols)
