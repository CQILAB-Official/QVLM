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

HF_MODEL_ID = "CQILAB/model_qvlm8b-wigner-expert"
HF_DATASET  = "CQILAB/QVLM-Wigner"
OUTPUT_DIR  = "ResultUpdate"

BEST_PROMPT = (
    "You are given a grayscale image representing a quantum optical state. "
    "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state, etc.) "
    "as well as its key parameters (alpha/number of photons/density, truncated Hilbert space dimension, and the linear space range). "
    "Please provide your answer in the following format: "
    "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] state with [KEY PARAMETER NAME] ≈ [VALUE], represented using a truncated Hilbert space of dimension d = [DIM_VALUE], "
    "in the linear space from -[LINVALUE] to [LINVALUE]. "
    "Under a compact binary encoding, this truncated space could be represented using ⌈log2([DIM_VALUE])⌉ = [TOTAL_QUBIT] qubits.\"\n\n"
    "Important Instructions:\n"
    "1. Extract your reasoning on how you determined the state, parameters, and number of qubits from the image.\n"
    "2. YOU MUST PROVIDE <think></think> BRACKET FOR THE THINKING PROCESS AND ALWAYS START WITH <think> FOR EACH ANSWER.\n"
    "3. DON'T TAKE THE THINKING OR CONCLUSION FROM THE IMAGE FORMAT OR IMAGE NAME OR IMAGE METADATA.\n"
    "4. DO NOT write literal strings like '[STATE TYPE]' or '[KEY PARAMETER NAME]'. You MUST replace these placeholders with the actual values you determined (e.g. 'cat', 'α', 'average photon', '19', '5' etc).\n"
    "5. THE VALUE OF [STATE TYPE], [KEY PARAMETER NAME], [VALUE], [DIM_VALUE], [LINVALUE], [TOTAL_QUBIT] IS THE MOST IMPORTANT VALUE TO BE EVALUATED.\n"
    "6. THE [TOTAL_QUBIT] IS CALCULATED BY log2([DIM_VALUE]) ROUNDED UP TO THE NEAREST INTEGER.\n"
    "7. YOU SHOULD ANSWER IN THIS EXACT FORMAT AFTER THE THINKING PROCESS:\n"
    "This is a [STATE TYPE] state with [KEY PARAMETER NAME] ≈ [VALUE], represented using a truncated Hilbert space of dimension d = [DIM_VALUE], "
    "in the linear space from -[LINVALUE] to [LINVALUE]. "
    "Under a compact binary encoding, this truncated space could be represented using ⌈log2([DIM_VALUE])⌉ = [TOTAL_QUBIT] qubits."
)

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
        max_new_tokens=1000,
        temperature=0.3,
        min_p=0.1,
        use_cache=True,
    )[:, inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# ──────────────────────────────────────────
# DATA / I/O
# ──────────────────────────────────────────
def process_dataset(seed=42, test_frac=0.10):
    print(f"Loading dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train")
    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=test_frac, seed=seed)
    test_ds = split["test"]
    print(f"Test set: {len(test_ds)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "qvlm8b-wigner-expert-inference.csv")

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "prediction", "ground_truth"])

        for idx, row in enumerate(test_ds):
            print(f"Processing row {idx+1}/{len(test_ds)}", flush=True)
            img = row["image"].convert("RGB").resize((1000, 600))

            pred = run_inference(img, BEST_PROMPT)
            writer.writerow([idx, pred, row["ground_truth"]])
            print(f"  done", flush=True)

    print(f"Wrote {len(test_ds)} rows to {out_file}")

# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == "__main__":
    process_dataset()
