import os
import csv
from PIL import Image
from datasets import load_dataset
from unsloth import FastVisionModel

# ──────────────────────────────────────────
# GLOBALS
# ──────────────────────────────────────────
MODEL = None
TOKENIZER = None

HF_MODEL_ID = "CQILAB/model_qvlm8b-codegen-expert"
HF_DATASET  = "CQILAB/QVLM-Circuit"
OUTPUT_DIR  = "ResultUpdate"

# ──────────────────────────────────────────
# MODEL HELPERS
# ──────────────────────────────────────────
def init_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return MODEL, TOKENIZER

    print(f"Loading model: {HF_MODEL_ID}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=HF_MODEL_ID,
        load_in_4bit=True,
    )
    MODEL = model
    TOKENIZER = tokenizer
    FastVisionModel.for_inference(MODEL)
    return MODEL, TOKENIZER

def run_inference(image: Image.Image, prompt: str) -> str:
    model, tok = init_model()
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
        temperature=0.3,
        min_p=0.1,
        use_cache=True,
    )[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ──────────────────────────────────────────
# DATA / I/O
# ──────────────────────────────────────────
def process_dataset(cols: list, seed=42, test_frac=0.20):
    print(f"Loading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train")
    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=test_frac, seed=seed)
    test_ds = split["test"]
    print(f"Test set: {len(test_ds)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "qvlm8b-codegen-expert-inference.csv")

    header = ["index"]
    for i in cols:
        header.extend([f"input_{i}", f"prediction_{i}", f"output_{i}"])

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, row in enumerate(test_ds):
            print(f"Processing row {idx+1}/{len(test_ds)}", flush=True)
            img = row["image"].convert("RGB").resize((1000, 600))
            row_out = [idx]

            for n in cols:
                input_col = f"input_{n}"
                out_col   = f"output_{n}"
                if input_col not in row or row[input_col] is None or str(row[input_col]).strip() == "":
                    row_out.extend(["", "", ""])
                    continue

                pred = run_inference(img, row[input_col])
                row_out.extend([row[input_col], pred, row[out_col]])
                print(f"  done col {n}", flush=True)

            writer.writerow(row_out)

    print(f"Wrote {len(test_ds)} rows to {out_file}")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    process_dataset(cols=[6])
