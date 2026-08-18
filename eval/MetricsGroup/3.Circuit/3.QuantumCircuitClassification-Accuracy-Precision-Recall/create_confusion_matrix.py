import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import re

# ── TPAMI Typography Settings ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':        12,          # global minimum
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'figure.dpi':       300,
    'savefig.dpi':      300,
})

# ── File / Model Config ─────────────────────────────────────────────────────
filenames = [
    'Circuit-Baseline1-Qwen3VL.csv',
    'Circuit-Baseline2-Llama.csv',
    'v3/Circuit-ChatGPT-4.1.csv',
    'v4/Circuit-QuantumVLM-2.5VL-7B-v4-focused-compile.1590.csv',
    'v4/Circuit-QuantumVLM-3VL-8B-v4-focused-compile.1590.csv',
]

MODEL_NAMES = {
    'Circuit-Baseline1-Qwen3VL.csv':                                       'Qwen3-VL-8B',
    'Circuit-Baseline2-Llama.csv':                                         'Llama3.2-V-11B',
    'v3/Circuit-ChatGPT-4.1.csv':                                          'ChatGPT-4.1',
    'v4/Circuit-QuantumVLM-2.5VL-7B-v4-focused-compile.1590.csv':         'QuantumVLM-7B',
    'v4/Circuit-QuantumVLM-3VL-8B-v4-focused-compile.1590.csv':           'QuantumVLM-8B',
}

DATA_DIR   = '/home/cqimain/1.Riset/1.Result/ResultGroup/3.Circuit'
OUTPUT_DIR = '/home/cqimain/1.Riset/1.Result/MetricsGroup/3.Circuit/' \
             '3.QuantumCircuitClassification-Accuracy-Precision-Recall/1.Confusion'


# ── Helpers ─────────────────────────────────────────────────────────────────
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
                print(f"    Available: {df.columns.tolist()}")
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
    df_clean = df_clean[df_clean['output_6'] != 'QFT']
    return df_clean


# ── Individual plot (unchanged logic, updated font sizes) ───────────────────
def plot_confusion_matrix(df, model_name):
    df_clean = prepare_clean(df)
    if df_clean.empty:
        print("  ✗ No valid data after cleaning.")
        return None, None

    y_true  = df_clean['output_6'].astype(str)
    y_pred  = df_clean['prediction_6'].astype(str)
    labels  = sorted(y_true.unique())
    n       = len(labels)
    cm      = confusion_matrix(y_true, y_pred, labels=labels)

    fig_w = max(10, n * 0.6)
    fig_h = max(8,  n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annot_size = 10 if n <= 10 else (9 if n <= 20 else 8)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': annot_size, 'family': 'serif'},
                ax=ax)

    ax.set_title(f'Confusion Matrix — {model_name}\n(prediction\\_6 vs output\\_6)',
                 fontsize=13, fontweight='bold', pad=16)
    ax.set_xlabel('Predicted Class Type', fontsize=11)
    ax.set_ylabel('Ground Truth Class Type', fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0,  labelsize=10)

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


# ── Combined plot — TPAMI single-row layout ─────────────────────────────────
def plot_combined_confusion_matrices(model_data_list):
    valid = [(name, df, lbls)
             for name, df, lbls in model_data_list if df is not None]
    n_models = len(valid)
    if n_models == 0:
        print("No valid data for combined plot.")
        return

    # Each panel is 4.5 in wide × 4.5 in tall — enough room for 10pt labels
    panel_w, panel_h = 4.5, 4.5
    fig, axes = plt.subplots(1, n_models,
                             figsize=(panel_w * n_models, panel_h))
    if n_models == 1:
        axes = [axes]

    for i, (model_name, df_clean, labels) in enumerate(valid):
        y_true = df_clean['output_6'].astype(str)
        y_pred = df_clean['prediction_6'].astype(str)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        ax     = axes[i]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False,
                    annot_kws={'size': 10, 'family': 'serif'})

        ax.set_title(model_name, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Ground Truth' if i == 0 else '', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0,  labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 1])

    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR,
                            f'Combined-Confusion-Matrices-All-Models.{ext}')
        plt.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
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