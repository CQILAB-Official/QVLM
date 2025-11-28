import os, re, shutil
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModel
)
from peft import get_peft_model, LoraConfig, TaskType


class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, formatter):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.formatter = formatter
        self.placeholders = re.findall(r"{([^}]+)}", formatter)
        self.image_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_path = row['image']
        input_text = row['input']
        output_text = row['output']

        # Load and transform the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Unable to load image at path: {image_path}. Error: {e}")

        image = self.image_transform(image)

        data_dict = {}
        for placeholder in self.placeholders:
            if placeholder == 'prompt':
                data_dict[placeholder] = input_text
            elif placeholder == 'answer':
                data_dict[placeholder] = output_text
            else:
                raise ValueError(f"Unexpected placeholder '{placeholder}' in formatter.")

        # Format the text using the formatter
        try:
            formatted_text = self.formatter.format(**data_dict)
        except KeyError as e:
            raise KeyError(f"Missing key for formatter: {e}")

        # Tokenize the formatted text
        encodings = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )

        # Prepare labels by copying input_ids
        encodings['labels'] = encodings['input_ids'].clone()

        # Squeeze to remove the batch dimension
        encodings = {key: val.squeeze(0) for key, val in encodings.items()}

        # Add pixel_values to encodings
        encodings['pixel_values'] = image
        encodings['image_flags'] = torch.ones(1, dtype=torch.int) 

        return encodings


class FinetuneLM:
    def __init__(self, 
                 data,  # New parameter to receive data directly
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="microsoft/Phi-3-vision-128k-instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                ):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_8bit=True,  # Enable 8-bit loading
        #     bnb_8bit_compute_dtype=torch.float16,  # Use float16 for computation
        #     bnb_8bit_use_double_quant=True,  # Use double quantization for memory efficiency
        # )

        # self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # self.tokenizer = self.processor.tokenizer
        self.base_model = AutoModel.from_pretrained(
            model_id,
            _attn_implementation='eager',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            # quantization_config=self.bnb_config,
            device_map="auto"
        )
        # Inspect module names
        # for name, module in self.base_model.named_modules():
        #     if isinstance(module, torch.nn.Linear):
        #         print(name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        
        self.peft_config = LoraConfig(
            r=peft_r, 
            lora_alpha=peft_alpha, 
            lora_dropout=peft_dropout, 
            target_modules = ["attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"],
            inference_mode = False
        )
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.formatter = "<|user|>\n<|image_1|>{prompt}<|end|><|assistant|>{answer}<|end|>"
        self.data = data  # Store the data

    def run(self):
        dataset = ImageTextDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            formatter=self.formatter
        )

        # model = get_peft_model(self.base_model, self.peft_config)
        model = get_peft_model(self.base_model, self.peft_config)
        training_args = TrainingArguments(
            learning_rate=self.learning_rate,
            output_dir='./model_cp_phi3_qutip',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            bf16=True,
            dataloader_num_workers=0,
            report_to="none",
            optim=self.optim,
            logging_steps=1, 
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        

if __name__ == "__main__":
    moved_folder = 'Dataset/'
    CSV_FILENAME = moved_folder+ 'metadata-qutip-color_2d3d_blues.csv' 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['type', 'image', 'ground_truth', 'prompt']
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into train and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    x_train = train_data['type'][:]
    images = train_data['image'][:]
    y_train = train_data['ground_truth'][:]
    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, etc.) "
        "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
        "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
    )
    
    print(f"Number of training samples: {len(x_train)}")
    print(f"Number of test samples: {len(test_data)}")

    fine_tune_data = []
    img_folder = moved_folder
    for i in range(len(x_train)):
        # print the index for debugging
        fine_tune_data.append({
            "image": img_folder + images[i],
            "input": BEST_PROMPT,
            "output": y_train[i],
        })

    finetuner = FinetuneLM(
        data=fine_tune_data,
        epochs=10,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=8,
        model_id="OpenGVLab/InternVL3-9B",
        peft_r=32,
        peft_alpha=32,
        peft_dropout=0.0,
    )

    finetuner.run()
