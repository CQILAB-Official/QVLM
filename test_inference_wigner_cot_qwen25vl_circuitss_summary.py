import os
import csv
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from unsloth import FastVisionModel
from torchsummary import summary
import torch.nn as nn

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
    model_root="./models/model_cot_qwen25vl_continued",
    ckpt="checkpoint-7496",
    base_id="unsloth/Qwen2.5-VL-7B-Instruct",
):
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    adapter_path = find_checkpoint(model_root, ckpt)
    print("using adapter:", adapter_path)

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_path,
        load_in_4bit=False,
    )
    MODEL = model.to("cuda")
    TOKENIZER = tokenizer
    FastVisionModel.for_inference(MODEL)
    return MODEL, TOKENIZER

def run_inference() -> str:
    model, tok = init_model()
    output_path = "output.log"
    output_module = "outputmodule.log"

    def identify_layer_type(name, param):
        name_lower = name.lower()

        if "window" in name_lower:
            return "[Window Attention]"
        if "rotary" in name_lower or "rope" in name_lower:
            return "[Rotary Positional Embedding]"
        if "full_attention" in name_lower or ("attn" in name_lower and "proj" not in name_lower and "qkv" not in name_lower):
            return "[Full Attention]"
        if "qkv" in name_lower:
            return "[QKV Projection]"
        if "proj" in name_lower:
            return "[Attention Output Projection]"
        if "mlp" in name_lower:
            return "[MLP]"
        if "norm" in name_lower:
            return "[LayerNorm]"
        return ""

    with open(output_path, "w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            tag = identify_layer_type(name, param)
            f.write(f"{name} {tuple(param.shape)} {tag}\n")
    
    with open(output_module, "w", encoding="utf-8") as f:
        for name, module in model.named_modules():
            f.write(f"{name} {module}\n")


    return f"Saved model parameter shapes to {output_path}"



# ──────────────────────────────────────────
# DATA / I/O
# ──────────────────────────────────────────
def process_csv(csv_path: str, max_io: int, seed=42, test_frac=0.20):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    _, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    test_df = test_df.reset_index(drop=True)

    out_file = os.path.splitext(csv_path)[0] + "_inference.csv"
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
                print(pred)
                row_out.extend([row[input_col], pred, row[out_col]])

            writer.writerow(row_out)

    print(f"Wrote {len(test_df)} rows to {out_file}")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    # ==== OLD SINGLE‑CSV FLOW (commented out) ====
    run_inference()