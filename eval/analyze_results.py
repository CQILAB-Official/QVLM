import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

# Set file path relative to this script
csv_file_path = 'ResultGroup/2.Entanglement/Entanglement-QuantumVLM-3VL-8B-4bit-v2814.csv'

# Load data
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded {csv_file_path}")
except FileNotFoundError:
    print(f"Error: File {csv_file_path} not found.")
    print(f"Current working directory: {os.getcwd()}")
    exit()

# --- Task 1: Entanglement Classification (prediction_1 vs output_1) ---
print("\n--- Task 1: Entanglement Classification ---")
if 'prediction_1' in df.columns and 'output_1' in df.columns:
    y_pred_cls = df['prediction_1'].fillna('Unknown').astype(str).str.strip()
    y_true_cls = df['output_1'].fillna('Unknown').astype(str).str.strip()
    labels = sorted(list(set(y_true_cls.unique()) | set(y_pred_cls.unique())))
    
    # Metrics
    acc = accuracy_score(y_true_cls, y_pred_cls)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_cls, y_pred_cls))
    
    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Entanglement Normalized Confusion Matrix\nAccuracy: {acc:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig('result_entanglement_normalized_confusion_matrix.png')
    plt.close()
    print("Saved result_entanglement_normalized_confusion_matrix.png")
else:
    print("Columns prediction_1 or output_1 not found.")

# --- Task 2: Purity Regression (prediction_2 vs output_2) ---
print("\n--- Task 2: Purity Estimation ---")
if 'prediction_2' in df.columns and 'output_2' in df.columns:
    y_pred_reg = pd.to_numeric(df['prediction_2'], errors='coerce')
    y_true_reg = pd.to_numeric(df['output_2'], errors='coerce')
    mask = ~np.isnan(y_pred_reg) & ~np.isnan(y_true_reg)
    y_pred_clean = y_pred_reg[mask]
    y_true_clean = y_true_reg[mask]
    
    if len(y_true_clean) > 0:
        # Metrics
        r2 = r2_score(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        pearson_corr, _ = pearsonr(y_true_clean, y_pred_clean)
        spearman_corr, _ = spearmanr(y_true_clean, y_pred_clean)
        
        print(f"R2 Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        
        # Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_true_clean, shade=True, label='Ground Truth', color='blue')
        sns.kdeplot(y_pred_clean, shade=True, label='Prediction', color='orange')
        plt.title(f'Purity Distribution Check\nPearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}')
        plt.xlabel('Purity Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('result_purity_distribution.png')
        plt.close()
        print("Saved result_purity_distribution.png")
    else:
        print("No valid numeric data found for regression task.")
else:
    print("Columns prediction_2 or output_2 not found.")
