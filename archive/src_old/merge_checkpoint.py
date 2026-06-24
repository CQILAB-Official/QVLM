#!/usr/bin/env python3
"""
Merge LoRA checkpoint with base model using Unsloth
"""
from unsloth import FastVisionModel
import torch

# Configuration
CHECKPOINT_PATH = "./model_cot_qwen25vl_32b_v3/checkpoint-3000"
OUTPUT_PATH = "./model_cot_qwen25vl_32b_v3/checkpoint-3000-merged-16-bit"

print(f"Loading checkpoint: {CHECKPOINT_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# Load the model with adapters
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=CHECKPOINT_PATH,
    load_in_4bit=True,  # Load in full precision for merging
    device_map="auto",
)

print("Merging LoRA adapters with base model...")

# Merge and save
model.save_pretrained_merged(
    OUTPUT_PATH,
    tokenizer,
    save_method="merged_16bit",  # Options: "merged_16bit", "merged_4bit", "lora"
)

print(f"✓ Merged model saved to: {OUTPUT_PATH}")
print(f"\nYou can now use this merged checkpoint for inference!")
