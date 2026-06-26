import math
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from transformers import TrainerCallback
from unsloth import FastVisionModel 
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

output_dir = './model_3vl_wigner_refactor_imporvedprompt'
model_id = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"

# tmux new -s impprompt 'python -u restart_training_wigner_v2.py 2>&1 | tee -a  output-log/imprompt-finetune.txt'


class FinetuneQwenVL:
    def __init__(self, 
                 data,
                 eval_data,
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="unsloth/Qwen3-VL-8B-Instruct", 
                 peft_r=8,
                 peft_alpha=16,
                 peft_dropout=0.05,
                ):
        """
        Args:
            data: a list of dicts for training
            eval_data: a list of dicts for evaluating (2-3 samples for quick tests every epoch)
        """
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        # 1) Load base model and tokenizer
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = model_id,
            load_in_4bit = False,
            use_gradient_checkpointing = "unsloth",
            device_map = "auto",
        )
        
        # 2) Wrap with PEFT / LoRA
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = True, # set True if you want vision layers updated
            finetune_language_layers   = True, # set True if you want language layers updated
            finetune_attention_modules = True,
            finetune_mlp_modules       = True,
            r = peft_r,
            lora_alpha = peft_alpha,
            lora_dropout = peft_dropout,
            bias = "none",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None
        )
        
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.data = data
        self.eval_data = eval_data

    def format_data(self, row):
        if isinstance(row["image"], Image.Image):
            img = row["image"].convert("RGB")
        else:
            img = Image.open("Dataset/" + row["image"]).convert("RGB")
        img.info = {}  # Strip metadata to prevent leakage
        img = img.resize((1000, 600))

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text": row["input"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text",  "text": row['output']},
                    ],
                },
            ]
        }


    def format_data_multiturn(self, row):
        # add 'Dataset/' to image path
        img = Image.open("Dataset/" + row["image"]).convert("RGB")
        img.info = {}  # Strip metadata to prevent leakage
        img = img.resize((1000, 600))

        messages = []
        image_sent = False
        turn = 1

        while True:
            in_key = f"input_{turn}"
            out_key = f"output_{turn}"
            if in_key not in row or out_key not in row:
                break

            # USER message
            content = []
            if not image_sent:
                content.append({"type": "image", "image": img})
                image_sent = True

            content.append({"type": "text", "text": row[in_key]})
            messages.append({"role": "user", "content": content})

            # ASSISTANT message (no "assistant", no duplication, force think)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": row[out_key]}],
            })

            turn += 1

        return {"messages": messages}

    def run(self, extra_train1=None, extra_test1=None, extra_train2=None, extra_test2=None):
        """
        Executes the fine-tuning process, including evaluation
        on 2-3 test samples at the end of each epoch.
        """
        # Convert your training and evaluation datasets
        converted_train_dataset = [self.format_data(row) for row in self.data]
        converted_eval_dataset  = [self.format_data(row) for row in self.eval_data]
        
        # --- optional add-ons -------------------------------------------
        if extra_train1 is not None:
            converted_train_dataset += [self.format_data_multiturn(r) for r in extra_train1]

        if extra_train2 is not None:
            converted_train_dataset += [self.format_data_multiturn(r) for r in extra_train2]

        if extra_test1 is not None:
            converted_eval_dataset += [self.format_data_multiturn(r) for r in extra_test1]

        if extra_test2 is not None:
            converted_eval_dataset += [self.format_data_multiturn(r) for r in extra_test2]
            
        
        # 3) TrainingArguments / SFTConfig
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir=output_dir,
            optim=self.optim,
            logging_steps=1,
            report_to="none",
            
            # Use bf16 if available, else fallback to fp16
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
            # max_steps=1000,             # limit continued run to 1000 more steps
            weight_decay = 0.01,            
            lr_scheduler_type = "linear",   
            seed = 3407,
            logging_strategy = "steps",
            
            # Evaluate at the end of every epoch
            # evaluation_strategy="epoch",
            
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = None,
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 16384,
        )
        
        # Model in training mode
        FastVisionModel.for_training(self.model)
        
        # 4) Create SFTTrainer with both train & eval sets
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            data_collator = UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset = converted_train_dataset,
            eval_dataset  = converted_eval_dataset,  # Evaluate on 2-3 items each epoch
            args = training_args,
            formatting_func = lambda x: x["messages"],
        )
        
        # 5) Start training. The trainer will evaluate at the end of each epoch
        trainer.train(resume_from_checkpoint=False)
        # trainer.train(resume_from_checkpoint="./model/model_3vl_wigner_refactor/checkpoint-2444")


# ------------------------------ helpers ---------------------------------
def csv_to_ft_lists(csv_path, test_frac=0.1, seed=42):
    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    return train_df.to_dict("records"), test_df.to_dict("records")

# --------------------------- main script --------------------------------
if __name__ == "__main__":
    # ------------ download dataset from HuggingFace ----
    HF_DATASET = "CQILAB/QVLM-Wigner"
    print(f"Loading dataset from HuggingFace: {HF_DATASET}")
    hf_dataset = load_dataset(HF_DATASET, split="train")
    print(f"Dataset loaded: {len(hf_dataset)} samples")

    # Verify required columns
    required_columns = ['image', 'ground_truth']
    for column in required_columns:
        if column not in hf_dataset.column_names:
            raise ValueError(f"Column '{column}' not found in the dataset.")

    # Shuffle and split into train/test (90/10)
    hf_dataset = hf_dataset.shuffle(seed=42)
    split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    test_dataset  = split["test"]

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

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

    # Prepare training data (image is already a PIL Image from HF dataset)
    fine_tune_data = []
    for row in train_dataset:
        fine_tune_data.append({
            "image": row["image"],       # PIL Image directly from HF dataset
            "input": BEST_PROMPT,
            "output": row["ground_truth"],
        })

    # For evaluation: pick just 3 rows from test set
    eval_data = []
    for row in test_dataset.select(range(min(3, len(test_dataset)))):
        eval_data.append({
            "image": row["image"],       # PIL Image directly from HF dataset
            "input": BEST_PROMPT,
            "output": row["ground_truth"],
        })

    # ------------ set up and run finetuning --------
    finetuner = FinetuneQwenVL(
        data=fine_tune_data,
        eval_data=eval_data,
        epochs=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        optim="adamw_torch_fused",
        model_id=model_id,
        peft_r=128,
        peft_alpha=128,
        peft_dropout=0.0,
    )

    finetuner.run()