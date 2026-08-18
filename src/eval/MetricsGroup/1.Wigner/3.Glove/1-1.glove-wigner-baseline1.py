import pandas as pd
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# ───────────────────────────────────────────
# 1. Load your CSV
# ───────────────────────────────────────────
# file_name = "case_study_circuit_patched_Inference_Circuit_New_Baseline1_Qwen3_BATCH8.csv"
file_name = "case_study_circuit_patched_Inference_Circuit_New_Baseline2_LlamaV_BATCH8.csv"
csv_path = f"glovecircuit/{file_name}"
df = pd.read_csv(csv_path)

# ───────────────────────────────────────────
# 2. Load or convert GloVe
# ───────────────────────────────────────────
glove_txt = "glove.6B.300d.txt"
glove_w2v = "glove.6B.300d.w2v"

if not os.path.exists(glove_w2v):
    print("[INFO] Converting GloVe format to word2vec format...")
    glove2word2vec(glove_txt, glove_w2v)
    print("[INFO] Conversion complete.")
else:
    print("[INFO] word2vec file already exists. Skipping conversion.")

print("[INFO] Loading embeddings...")
model = KeyedVectors.load_word2vec_format(glove_w2v, binary=False)
print("[INFO] Embeddings loaded.")

# ───────────────────────────────────────────
# 3. Helper: Convert a sentence → vector
# ───────────────────────────────────────────
def sentence_vector(text):
    words = str(text).lower().split()
    vecs = [model[w] for w in words if w in model]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# ───────────────────────────────────────────
# 4. Seven prediction/output pairs
# ───────────────────────────────────────────
pairs = [
    ("prediction_1", "output_1"),
    ("prediction_2", "output_2"),
    ("prediction_3", "output_3"),
    ("prediction_4", "output_4"),
    ("prediction_5", "output_5"),
    ("prediction_6", "output_6"),
    ("prediction_7", "output_7"),
]

total = len(df)

# ───────────────────────────────────────────
# 5. Compute similarity for each pair
# ───────────────────────────────────────────
for pred_col, gt_col in pairs:
    print(f"\n[INFO] Scoring {pred_col} vs {gt_col} ...")

    scores = []
    last_print = time.time()

    for i, row in df.iterrows():
        v1 = sentence_vector(row[pred_col])
        v2 = sentence_vector(row[gt_col])
        sim = cosine_similarity([v1], [v2])[0][0]
        scores.append(sim)

        if time.time() - last_print > 1:
            pct = (i + 1) / total * 100
            print(f"[INFO] {pred_col}: {i+1}/{total} ({pct:.2f}%)")
            last_print = time.time()

    df[f"glove_score_{pred_col}"] = scores

print("\n[INFO] All scoring finished.")

# ───────────────────────────────────────────
# 6. Save
# ───────────────────────────────────────────
out_path = f"glove_{file_name}"
df.to_csv(out_path, index=False)

print(f"[INFO] Saved to {out_path}")
