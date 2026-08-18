import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ── Typography ─────────────────────────────────────────────────────────────
TITLE_PT = 12
ANNOT_PT = 11
TICK_PT  = 10
LABEL_PT = 12
VAL_PT   = 9
SAVE_DPI = 300

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':         TICK_PT,
    'axes.titlesize':    TITLE_PT,
    'axes.labelsize':    LABEL_PT,
    'xtick.labelsize':   TICK_PT,
    'ytick.labelsize':   TICK_PT,
    'legend.fontsize':   TICK_PT,
    'figure.dpi':        SAVE_DPI,
    'savefig.dpi':       SAVE_DPI,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3.0,
    'ytick.major.size':  3.0,
})

# ── Config ─────────────────────────────────────────────────────────────────
filenames = [
    'v2/1.Converted-Wigner-Baseline1-Qwen3VL-v2.csv',
    'v2/2.Converted-Wigner-Baseline2-Llama-v2.csv',
    'v2/5.Converted-Wigner-Baseline3-ChatGPT-4.1.csv',
    'v2/3.Converted-Wigner-QuantumVLM-2.5VL-7B-v2444-v2.csv',
    'v2/4.Converted-Wigner-QuantumVLM-3VL-8B-v2.csv',
]
filenames_rename = [
    "Qwen3-VL-8B", "Llama3.2-V-11B", "ChatGPT-4.1", "QuantumVLM-7B", "QuantumVLM-8B",
]

BASE_DIR   = '/home/cqimain/1.Riset/1.Result/MetricsGroup/1.Wigner/4.Accuracy'
OUTPUT_DIR = os.path.join(BASE_DIR, '4.Confusion')

N_MODELS  = len(filenames)
PANEL_W   = 4.0   # in — each panel, unconstrained, comfortable size
PANEL_H   = 4.0   # in — square panels
FIG_W     = PANEL_W * N_MODELS   # total figure width, let LaTeX scale it


# ── Data Loading ───────────────────────────────────────────────────────────
def load_data(filename):
    input_file = os.path.join(BASE_DIR, '2.Output', filename)
    print(f"Reading: {input_file}")
    if not os.path.exists(input_file):
        print(f"  ✗ Not found: {input_file}")
        return None
    try:
        df = pd.read_csv(input_file)
        for col in ('param', 'dimension', 'qubits', 'linear'):
            df[f'{col}_gt']        = pd.to_numeric(df[f'{col}_gt'],        errors='coerce')
            df[f'{col}_generated'] = pd.to_numeric(df[f'{col}_generated'], errors='coerce')
        return df
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


# ── Plot 1: Confusion Matrices ─────────────────────────────────────────────
def plot_combined_confusion_matrices(model_data_list):
    excluded = {'nan', 'number'}

    all_labels = set()
    for _, df, _ in model_data_list:
        if df is not None:
            mask = (~df['state_gt'].astype(str).isin(excluded) &
                    ~df['state_generated'].astype(str).isin(excluded))
            all_labels.update(df.loc[mask, 'state_gt'].astype(str).unique())
            all_labels.update(df.loc[mask, 'state_generated'].astype(str).unique())
    labels = sorted(all_labels)

    fig, axes = plt.subplots(1, N_MODELS,
                             figsize=(FIG_W, PANEL_H),
                             constrained_layout=True)

    for i, (model_name, df, _) in enumerate(model_data_list):
        ax = axes[i]
        if df is None:
            ax.axis('off')
            continue

        mask   = (~df['state_gt'].astype(str).isin(excluded) &
                  ~df['state_generated'].astype(str).isin(excluded))
        y_true = df.loc[mask, 'state_gt'].astype(str)
        y_pred = df.loc[mask, 'state_generated'].astype(str)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False,
                    annot_kws={'size': ANNOT_PT, 'family': 'serif'})

        ax.set_title(model_name, fontsize=TITLE_PT, fontweight='bold', pad=6)
        ax.set_xlabel('Predicted',  fontsize=LABEL_PT, labelpad=4)
        ax.set_ylabel('Ground Truth' if i == 0 else '', fontsize=LABEL_PT, labelpad=4)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=TICK_PT)
        ax.set_yticklabels(labels, rotation=0,              fontsize=TICK_PT)

    _save(fig, 'Combined-Confusion-Matrices')


# ── Plot 2: Scatter Plots ──────────────────────────────────────────────────
def plot_combined_parameter_scatters(model_data_list):
    fig, axes = plt.subplots(1, N_MODELS,
                             figsize=(FIG_W, PANEL_H),
                             constrained_layout=True)

    for i, (model_name, df, _) in enumerate(model_data_list):
        ax = axes[i]
        if df is None:
            ax.axis('off')
            continue

        param_df = df.dropna(subset=['param_gt', 'param_generated'])
        if param_df.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=TICK_PT)
            ax.set_title(model_name, fontsize=TITLE_PT, fontweight='bold')
            continue

        sns.scatterplot(data=param_df, x='param_gt', y='param_generated',
                        hue='state_gt', alpha=0.6, ax=ax, legend=(i == 0), s=20)

        lo = min(param_df['param_gt'].min(), param_df['param_generated'].min())
        hi = max(param_df['param_gt'].max(), param_df['param_generated'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.6, linewidth=0.8)

        ax.set_title(model_name, fontsize=TITLE_PT, fontweight='bold', pad=6)
        ax.set_xlabel('Ground Truth Parameter', fontsize=LABEL_PT, labelpad=4)
        ax.set_ylabel('Generated Parameter' if i == 0 else '', fontsize=LABEL_PT, labelpad=4)
        ax.tick_params(axis='both', labelsize=TICK_PT)

        if i == 0:
            ax.legend(fontsize=TICK_PT - 1, frameon=True, edgecolor='0.8',
                      loc='upper left', markerscale=0.8,
                      handlelength=1.0, borderpad=0.4, labelspacing=0.3)

    _save(fig, 'Combined-Parameter-Scatter')


# ── Plot 3: Bar Charts ─────────────────────────────────────────────────────
def plot_combined_accuracy_metrics(model_data_list):
    tol = 1e-5

    fig, axes = plt.subplots(1, N_MODELS,
                             figsize=(FIG_W, PANEL_H),
                             constrained_layout=True)

    metric_keys = ['State', 'Param', 'Dim', 'Qubits', 'Linear', 'All']
    bar_colors  = sns.color_palette('viridis', len(metric_keys))

    for i, (model_name, df, _) in enumerate(model_data_list):
        ax = axes[i]
        if df is None:
            ax.axis('off')
            continue

        def col_acc(col):
            valid = df.dropna(subset=[f'{col}_gt'])
            if valid.empty:
                return 0.0
            return (abs(valid[f'{col}_generated'] - valid[f'{col}_gt']) < tol).mean() * 100

        state_acc = (df['state_generated'] == df['state_gt']).mean() * 100
        all_ok    = (
            (df['state_generated'] == df['state_gt']) &
            (abs(df['param_generated']     - df['param_gt'])     < tol) &
            (abs(df['dimension_generated'] - df['dimension_gt']) < tol) &
            (abs(df['qubits_generated']    - df['qubits_gt'])    < tol) &
            (abs(df['linear_generated']    - df['linear_gt'])    < tol)
        )
        values = [
            state_acc, col_acc('param'), col_acc('dimension'),
            col_acc('qubits'), col_acc('linear'), all_ok.mean() * 100,
        ]

        bars = ax.bar(range(len(metric_keys)), values,
                      color=bar_colors, width=0.65,
                      edgecolor='white', linewidth=0.4)

        ax.set_ylim(0, 120)
        ax.set_title(model_name, fontsize=TITLE_PT, fontweight='bold', pad=6)
        ax.set_ylabel('Accuracy (%)' if i == 0 else '', fontsize=LABEL_PT, labelpad=4)
        ax.set_xticks(range(len(metric_keys)))
        ax.set_xticklabels(metric_keys, rotation=40, ha='right', fontsize=TICK_PT)
        ax.tick_params(axis='y', labelsize=TICK_PT)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f'{v:.1f}%',
                    ha='center', va='bottom',
                    fontsize=VAL_PT, fontfamily='serif')

    _save(fig, 'Combined-Accuracy-Metrics')


# ── Save helper ────────────────────────────────────────────────────────────
def _save(fig, stem):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUTPUT_DIR, f'{stem}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=SAVE_DPI)
        print(f"  Saved: {path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_data = []
    for fname, rename in zip(filenames, filenames_rename):
        model_data.append((rename, load_data(fname), fname))

    print("\nGenerating confusion matrices …")
    plot_combined_confusion_matrices(model_data)

    print("Generating scatter plots …")
    plot_combined_parameter_scatters(model_data)

    print("Generating bar charts …")
    plot_combined_accuracy_metrics(model_data)

    print("\nDone!")

if __name__ == "__main__":
    main()