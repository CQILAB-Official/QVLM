"""
Classifier Performance Evaluation on Wigner Dataset
Evaluates pre-trained classifier on 5 quantum state classes from wigner_refactor.csv
"""

import pandas as pd
import numpy as np
from PIL import Image
import os
import sys

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add qst-nn to path
sys.path.append('qst-nn')

from qst_nn.models.classifier import Classifier
from qst_nn.data.preprocess import normalize
from qst_nn.training.train_classifier import loss, optimizer

import matplotlib
matplotlib.use('Agg')  # For saving figures without display
import matplotlib.pyplot as plt
import itertools

# Custom confusion matrix plotting function (to avoid qutip dependencies)
def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title='Confusion Matrix', 
                         cmap=plt.cm.Blues, fig=None, ax=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig, ax

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.keras.backend.set_floatx('float32')

# Define the 5 classes we want to evaluate
SELECTED_CLASSES = ['fock', 'coherent', 'thermal', 'cat', 'random']

# Paths
CSV_FILE = 'Dataset/converted_wigner_refactor.csv'
CHECKPOINT_PATH = 'qst-nn/paper_figures/classifier/'
OUTPUT_DIR = 'classifier_results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("Classifier Performance Evaluation on Wigner Dataset")
print("="*80)

# =============================================================================
# Load and Filter Data
# =============================================================================
print("\n1. Loading data from CSV...")
df = pd.read_csv(CSV_FILE)
print(f"   Total samples: {len(df)}")

# Check if 'state_gt' column exists
if 'state_gt' not in df.columns:
    print(f"   ERROR: 'state_gt' column not found. Available columns: {df.columns.tolist()}")
    sys.exit(1)

print(f"   Unique types: {df['state_gt'].unique().tolist()}")

# Filter for selected classes
df_filtered = df[df['state_gt'].isin(SELECTED_CLASSES)].copy()
print(f"\n2. Filtered to {len(df_filtered)} samples from {len(SELECTED_CLASSES)} classes")
print("   Class distribution:")
for class_name in SELECTED_CLASSES:
    count = (df_filtered['state_gt'] == class_name).sum()
    print(f"     - {class_name:10s}: {count:5d} samples")

# Create label mapping
label_map = {name: i for i, name in enumerate(SELECTED_CLASSES)}
df_filtered['label'] = df_filtered['state_gt'].map(label_map)

# Split into train/test
train_df, test_df = train_test_split(
    df_filtered, 
    test_size=0.1, 
    random_state=42, 
    stratify=df_filtered['label']
)

print(f"\n3. Dataset split:")
print(f"   Train: {len(train_df)} samples")
print(f"   Test:  {len(test_df)} samples")

# =============================================================================
# Load Images
# =============================================================================
def load_images(df, base_path='Dataset'):
    """Load and preprocess images from dataframe."""
    images = []
    labels = []
    skipped = 0
    
    for idx, row in df.iterrows():
        img_path = os.path.join(base_path, row['image'])
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((32, 32)).convert('L')
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
            labels.append(row['label'])
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"   Warning: {skipped} images not found")
    
    return np.array(images), np.array(labels)

print("\n4. Loading images...")
print("   Loading test images...")
x_test, y_test = load_images(test_df)

# Reshape for CNN input (add channel dimension)
x_test = x_test.reshape(-1, 32, 32, 1)
y_test = y_test.reshape(-1, 1)

print(f"   Test data shape: {x_test.shape}")
print(f"   Test labels shape: {y_test.shape}")

# =============================================================================
# Load Pre-trained Classifier
# =============================================================================
print("\n5. Loading pre-trained classifier...")

test_data_generator = ImageDataGenerator(
    preprocessing_function=normalize
)

# Use legacy optimizer to load old checkpoint
legacy_optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.9, 0.9)

classifier = Classifier()
classifier.compile(
    optimizer=legacy_optimizer,
    loss=loss,
    metrics=['accuracy']
)

# Load pre-trained weights
try:
    classifier.load_weights(CHECKPOINT_PATH)
    print(f"   ✓ Loaded weights from {CHECKPOINT_PATH}")
except Exception as e:
    print(f"   ✗ Error loading weights: {e}")
    sys.exit(1)

# =============================================================================
# Make Predictions
# =============================================================================
print("\n6. Making predictions...")
batch_size = 128
test_data_gen = test_data_generator.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
y_pred_probs = classifier.predict(test_data_gen, verbose=0)

# Get predicted class (model has 7 outputs, we use first 5)
y_pred_7class = np.argmax(y_pred_probs, axis=1)

print(f"   Prediction distribution across all 7 model classes:")
unique, counts = np.unique(y_pred_7class, return_counts=True)
for u, c in zip(unique, counts):
    class_name = SELECTED_CLASSES[u] if u < 5 else f'class_{u}'
    print(f"     Class {u} ({class_name:10s}): {c:4d} samples")

# Filter to only our 5 classes
valid_mask = y_pred_7class < 5
y_test_filtered = y_test[valid_mask].flatten()
y_pred_filtered = y_pred_7class[valid_mask]

print(f"\n   Kept {valid_mask.sum()} / {len(y_test)} predictions (filtered to 5 classes)")

# =============================================================================
# Calculate Metrics
# =============================================================================
print("\n7. Performance Metrics")
print("="*80)

# Overall accuracy
accuracy = accuracy_score(y_test_filtered, y_pred_filtered)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Per-class accuracy
print("\nPer-Class Accuracy:")
print("-"*50)
for i, class_name in enumerate(SELECTED_CLASSES):
    mask = y_test_filtered == i
    if mask.sum() > 0:
        class_acc = (y_pred_filtered[mask] == i).sum() / mask.sum()
        n_samples = mask.sum()
        n_correct = (y_pred_filtered[mask] == i).sum()
        print(f"  {class_name:10s}: {class_acc:.4f} ({n_correct:3d}/{n_samples:3d} correct)")
print("-"*50)

# =============================================================================
# Save Detailed Predictions to CSV
# =============================================================================
print("\n8. Saving detailed predictions to CSV...")
detailed_results = []

for idx, row in test_df.iterrows():
    # Get the index in the test arrays
    test_idx = test_df.index.get_loc(idx)
    
    true_label = int(row['label'])
    pred_label = int(y_pred_7class[test_idx])
    
    true_class = SELECTED_CLASSES[true_label] if true_label < 5 else 'unknown'
    pred_class = SELECTED_CLASSES[pred_label] if pred_label < 5 else f'class_{pred_label}'
    
    is_correct = true_label == pred_label
    
    # Get ground truth text if available
    ground_truth_text = row.get('ground_truth', '')
    
    detailed_results.append({
        'image_path': row['image'],
        'true_label': true_label,
        'true_class': true_class,
        'predicted_label': pred_label,
        'predicted_class': pred_class,
        'correct': is_correct,
        'ground_truth_text': ground_truth_text
    })

# Save to CSV
detailed_csv = os.path.join(OUTPUT_DIR, 'detailed_predictions.csv')
detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(detailed_csv, index=False)
print(f"   ✓ Saved detailed predictions to: {detailed_csv}")
print(f"   Total predictions: {len(detailed_results)}")
print(f"   Correct: {detailed_df['correct'].sum()}")
print(f"   Incorrect: {(~detailed_df['correct']).sum()}")

# Show sample of predictions
print("\n   Sample predictions (first 10):")
print("-"*100)
for i, row in detailed_df.head(10).iterrows():
    status = "✓" if row['correct'] else "✗"
    print(f"   {status} GT: {row['true_class']:10s} | Pred: {row['predicted_class']:10s} | {row['image_path']}")
print("-"*100)

# =============================================================================
# Confusion Matrix
# =============================================================================
print("\n9. Generating confusion matrix...")
cm = confusion_matrix(y_test_filtered, y_pred_filtered)
print("\nConfusion Matrix (raw counts):")
print(cm)

# Create and save visualization
fig_width_pt = 246.0
inches_per_pt = 1.0/72.27
fig_width = fig_width_pt * inches_per_pt
fig_size = (fig_width, fig_width)

fig, ax = plt.subplots(1, 1, figsize=fig_size)

try:
    plot_confusion_matrix(
        y_test_filtered, 
        y_pred_filtered, 
        classes=SELECTED_CLASSES, 
        normalize=True,
        fig=fig, 
        ax=ax,
        title="Classifier Performance on Wigner Dataset"
    )
    
    output_file = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved confusion matrix to: {output_file}")
except Exception as e:
    print(f"   ✗ Error creating confusion matrix plot: {e}")
finally:
    plt.close()

# =============================================================================
# Ground Truth Examples
# =============================================================================
print("\n10. Example Predictions vs Ground Truth")
print("="*80)

n_examples = min(20, len(test_df))
indices = np.random.choice(len(test_df), n_examples, replace=False)

n_correct = 0
for i, idx in enumerate(indices):
    test_idx = test_df.index[idx]
    row = test_df.loc[test_idx]
    
    true_label = int(row['label'])
    pred_label = int(y_pred_7class[idx])
    
    true_class = SELECTED_CLASSES[true_label] if true_label < 5 else 'unknown'
    pred_class = SELECTED_CLASSES[pred_label] if pred_label < 5 else f'class_{pred_label}'
    
    match = "✓" if true_label == pred_label else "✗"
    if true_label == pred_label:
        n_correct += 1
    
    print(f"{i+1:2d}. {match} GT: {true_class:10s} | Pred: {pred_class:10s} | {row['image']}")

print("="*80)
print(f"Sample accuracy: {n_correct}/{n_examples} = {n_correct/n_examples:.2%}")

# =============================================================================
# Save Results Summary
# =============================================================================
print("\n11. Saving results summary...")
summary_file = os.path.join(OUTPUT_DIR, 'results_summary.txt')

with open(summary_file, 'w') as f:
    f.write("Classifier Performance on Wigner Dataset\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {CSV_FILE}\n")
    f.write(f"Classes: {', '.join(SELECTED_CLASSES)}\n")
    f.write(f"Test samples: {len(test_df)}\n")
    f.write(f"Valid predictions: {valid_mask.sum()}\n\n")
    
    f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
    
    f.write("Per-Class Accuracy:\n")
    f.write("-"*50 + "\n")
    for i, class_name in enumerate(SELECTED_CLASSES):
        mask = y_test_filtered == i
        if mask.sum() > 0:
            class_acc = (y_pred_filtered[mask] == i).sum() / mask.sum()
            n_samples = mask.sum()
            n_correct = (y_pred_filtered[mask] == i).sum()
            f.write(f"{class_name:10s}: {class_acc:.4f} ({n_correct}/{n_samples})\n")
    f.write("-"*50 + "\n\n")
    
    f.write("Confusion Matrix (raw counts):\n")
    f.write(str(cm) + "\n")

print(f"   ✓ Saved summary to: {summary_file}")

print("\n" + "="*80)
print("✓ Evaluation complete!")
print(f"   Results saved to: {OUTPUT_DIR}/")
print("="*80)