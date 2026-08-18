import pandas as pd
from gensim.models import Word2Vec

# Load the dataset
dataset_path = "/home/cqimain/1.Riset/1.Result/ResultGroup/4.Dataset/wigner_refactor.csv"
print(f"[INFO] Loading dataset from {dataset_path}...")
df = pd.read_csv(dataset_path)

# Extract ground_truth sentences
print("[INFO] Extracting 'ground_truth' column...")
sentences = df['ground_truth'].astype(str).tolist()

# Tokenize sentences (simple split by whitespace, lowercase)
print("[INFO] Tokenizing sentences...")
tokenized_sentences = [sentence.lower().split() for sentence in sentences]

# Train the Word2Vec model
print("[INFO] Training Word2Vec model...")
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model_path = "/home/cqimain/1.Riset/1.Result/MetricsGroup/1.Wigner/2.Word2Vec/custom_w2v_groundtruth.model"
print(f"[INFO] Saving model to {model_path}...")
model.save(model_path)
print("[INFO] Done!")
