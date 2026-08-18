import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import re

# ── TPAMI Typography ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':        12,
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'figure.dpi':       300,
    'savefig.dpi':      300,
})

# ── Config — 3 models only ─────────────────────────────────────────────────
filenames = [
    'Circuit-Baseline2-Llama.csv',
    'v3/Circuit-ChatGPT-4.1.csv',
    'v4/Circuit-QuantumVLM-3VL-8B-v4-focused-compile.1590.csv',
]

MODEL_NAMES = {
    'Circuit-Baseline2-Llama.csv':                                       'Llama3.2-11B',
    'v3/Circuit-ChatGPT-4.1.csv':                                        'ChatGPT-4.1',
    'v4/Circuit-QuantumVLM-3VL-8B-v4-focused-compile.1590.csv':         'QuantumVLM-8B',
}

DATA_DIR   = '/home/cqimain/1.Riset/1.Result/ResultGroup/3.Circuit'
OUTPUT_DIR = '/home/cqimain/1.Riset/1.Result/MetricsGroup/3.Circuit/' \
             '3.QuantumCircuitClassification-Accuracy-Precision-Recall/2.Confusion2'

N_MODELS = len(filenames)   # 3
PANEL_W  = 4.5              # in — comfortable at 3-up
PANEL_H  = 4.5              # in
FIG_W    = PANEL_W * N_MODELS   # 13.5 in — LaTeX scales to \textwidth


# ── Helpers ────────────────────────────────────────────────────────────────
def load_data(filename):
    input_file = os.path.join(DATA_DIR, filename)
    print(f"Reading: {input_file}")
    if not os.path.exists(input_file):
        print(f"  ✗ File not found: {input_file}")
        return None
    try:
        df = pd.read_csv(input_file)
        for col in ('prediction_6', 'output_6'):
            if col not in df.columns:
                print(f"  ✗ Missing column '{col}' in {filename}")
                return None
        return df[['prediction_6', 'output_6']].copy()
    except Exception as e:
        print(f"  ✗ Error reading {filename}: {e}")
        return None


def prepare_clean(df):
    """Strip <think> tags, drop NaNs, drop QFT class."""
    df_clean = df.dropna(subset=['prediction_6', 'output_6']).copy()
    df_clean['prediction_6'] = df_clean['prediction_6'].apply(
        lambda x: re.sub(r'<think>.*?</think>', '', str(x), flags=re.DOTALL).strip()
    )
    df_clean = df_clean[~df_clean['output_6'].isin(['QFT', 'GHZ'])]
    return df_clean


# ── Individual plots ───────────────────────────────────────────────────────
def plot_confusion_matrix(df, model_name):
    df_clean = prepare_clean(df)
    if df_clean.empty:
        print("  ✗ No valid data after cleaning.")
        return None, None

    y_true = df_clean['output_6'].astype(str)
    y_pred = df_clean['prediction_6'].astype(str)
    labels = sorted(y_true.unique())
    n      = len(labels)
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    # Row-normalize → recall per class
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig_w = max(10, n * 0.6)
    fig_h = max(8,  n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Recall'},
                annot_kws={'size': 11, 'family': 'serif'},
                ax=ax)

    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=13, fontweight='bold', pad=16)
    ax.set_xlabel('Predicted Class Type',    fontsize=12)
    ax.set_ylabel('Ground Truth Class Type', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', rotation=0,  labelsize=11)

    plt.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR, f'Confusion-Matrix-{model_name}.{ext}')
        plt.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()

    correct  = (y_true == y_pred).sum()
    accuracy = correct / len(df_clean) * 100
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(df_clean)})")
    return df_clean, labels


# ── Combined plot ──────────────────────────────────────────────────────────
def plot_combined_confusion_matrices(model_data_list):
    valid = [(name, df, lbls)
             for name, df, lbls in model_data_list if df is not None]
    if not valid:
        print("No valid data for combined plot.")
        return

    # Use global label set for consistent ordering across panels
    all_labels = set()
    for _, df_clean, _ in valid:
        # Exclude GHZ from global label set too
        mask = ~df_clean['output_6'].astype(str).isin(['QFT', 'GHZ'])
        all_labels.update(df_clean.loc[mask, 'output_6'].astype(str).unique())
    labels = sorted(all_labels)

    fig, axes = plt.subplots(1, N_MODELS,
                             figsize=(FIG_W, PANEL_H),
                             constrained_layout=True)
    if N_MODELS == 1:
        axes = [axes]

    for i, (model_name, df_clean, _) in enumerate(valid):
        ax     = axes[i]
        y_true = df_clean['output_6'].astype(str)
        y_pred = df_clean['prediction_6'].astype(str)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)

        # Row-normalize → recall per class
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        is_last = (i == N_MODELS - 1)
        sns.heatmap(cm_norm,
                    annot=True, fmt='.2f', cmap='Blues',
                    vmin=0, vmax=1,
                    xticklabels=labels, yticklabels=labels,
                    ax=ax,
                    cbar=is_last,
                    cbar_kws={'label': 'Recall', 'shrink': 0.85, 'pad': 0.02}
                             if is_last else {},
                    annot_kws={'size': 11, 'family': 'serif'})

        ax.set_title(model_name, fontsize=12, fontweight='bold', pad=6)
        ax.set_xlabel('Predicted',    fontsize=12)
        ax.set_ylabel('Ground Truth' if i == 0 else '', fontsize=12)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=11)
        ax.set_yticklabels(labels, rotation=0,              fontsize=11)

    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR, f'Combined-Confusion-Matrices-All-Models.{ext}')
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_data_list = []

    for filename in filenames:
        print(f"\n{'='*60}\nProcessing: {filename}\n{'='*60}")
        df = load_data(filename)
        if df is None:
            model_data_list.append((None, None, None))
            continue

        model_name = MODEL_NAMES.get(filename,
                        filename.replace('Circuit-', '').replace('.csv', ''))
        df_clean, labels = plot_confusion_matrix(df, model_name)
        model_data_list.append((model_name, df_clean, labels))

    print(f"\n{'='*60}\nGenerating combined confusion matrix …\n{'='*60}")
    plot_combined_confusion_matrices(model_data_list)
    print("\nDone!")


if __name__ == "__main__":
    main()