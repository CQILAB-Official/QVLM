import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import torch

from unsloth import FastVisionModel 
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# output_dir = './model_cot_qwen25vl_32b_vfocused'
output_dir = './model/model_25vl_wigner_circuit_refactor_imporvedprompt_focused'
# model_id = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
model_id = "./model_25vl_wigner_circuit_refactor_imporvedprompt/checkpoint-3752"
base_image_path = "./Dataset/"

# tmux new -s retrain 'python -u x_continued_finetuning_source2_wigner_qwen25vl_retrain.py 2>&1 | tee -a  output-log/retrain-circuits.txt'
# tmux new -s retrain 'CUDA_VISIBLE_DEVICES=1 python -u x_continued_finetuning_source2_wigner_qwen25vl_retrain.py 2>&1 | tee -a  output-log/retrain-circuits-focus.txt'

class FinetuneQwenVL:
    def __init__(self, 
                 data,
                 eval_data,
                 epochs=1, 
                 learning_rate=1e-4,
                 warmup_ratio=0.1,
                 gradient_accumulation_steps=64,
                 optim="adamw_torch",
                 model_id="unsloth/Qwen2-VL-7B-Instruct", 
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
            # Aggressively limit GPU 0 to leave room for training overhead (gradients, optimizer, activations)
            # max_memory = {0: "5GiB", 1: "16GiB"}, 
        )

        try:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers     = True, 
                finetune_language_layers   = True,
                finetune_attention_modules = True,
                finetune_mlp_modules       = True, # DISABLE to save massive memory
                r = peft_r,
                lora_alpha = peft_alpha,
                lora_dropout = peft_dropout,
                bias = "none",
                random_state = 3407,
                use_rslora = False,
                loftq_config = None,
            )
        except RuntimeError as e:
            if "already added LoRA adapters" in str(e):
                print("LoRA adapters already loaded from checkpoint. Skipping get_peft_model.")
            else:
                raise e
        
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.data = data
        self.eval_data = eval_data

    def format_data(self, row):
        image_path = row["image"]
        input_text = row['input']
        output_text = row['output']
        
        try:
            image = Image.open(base_image_path+image_path).convert("RGB")
            # If needed, you can also resize or transform:
            image = image.resize((500, 300))
        except Exception as e:
            raise FileNotFoundError(
                f"Unable to load image at path: {image_path}. Error: {e}"
            )

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        },
                        {
                            "type": "image",
                            "image": image,  
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": output_text,
                        }
                    ],
                },
            ],
        }
        
    def format_data_multiturn(self, row):
        img_path = row["image"]
        try:
            img = Image.open(img_path).convert("RGB").resize((1000, 600))
        except Exception as e:
            raise FileNotFoundError(f"Cannot open {img_path}: {e}")

        messages   = []
        image_sent = False
        # turnnya di jadiin turn 5
        turn       = 1

        while True:
            in_key  = f"input_{turn}"
            out_key = f"output_{turn}"
            if in_key not in row or out_key not in row:
                break

            user_text      = row[in_key]
            assistant_text = row[out_key]

            if user_text is None or assistant_text is None:
                break
            if isinstance(user_text, float) and math.isnan(user_text):
                break
            if isinstance(assistant_text, float) and math.isnan(assistant_text):
                break
            if str(user_text).strip() == "" and str(assistant_text).strip() == "":
                break

            # ----- user message -----
            user_content = []
            if not image_sent:
                user_content.append({"type": "image", "image": img})
                image_sent = True
            user_content.append({"type": "text", "text": str(user_text)})

            messages.append({"role": "user", "content": user_content})

            # ----- assistant reply -----
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": str(assistant_text)}],
            })

            turn += 1

        return {"messages": messages}

    def run(self, extra_train1=None, extra_test1=None, extra_train2=None, extra_test2=None):
        """
        Executes the fine-tuning process, including evaluation
        on 2-3 test samples at the end of each epoch.
        """
        # Convert your training and evaluation datasets
        print("Formatting main training data...")
        converted_train_dataset = [self.format_data(row) for row in tqdm(self.data, desc="Formatting Train Data")]
        print("Formatting main evaluation data...")
        converted_eval_dataset  = [self.format_data(row) for row in tqdm(self.eval_data, desc="Formatting Eval Data")]
        
        # --- optional add-ons -------------------------------------------
        if extra_train1 is not None:
            print("Formatting extra_train1...")
            converted_train_dataset += [self.format_data_multiturn(r) for r in tqdm(extra_train1, desc="Formatting Extra Train 1")]

        if extra_train2 is not None:
            converted_train_dataset += [self.format_data_multiturn(r) for r in extra_train2]

        if extra_test1 is not None:
            print("Formatting extra_test1...")
            converted_eval_dataset += [self.format_data_multiturn(r) for r in tqdm(extra_test1, desc="Formatting Extra Test 1")]

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
            gradient_checkpointing = True, # Critical for memory saving!
            
            logging_first_step=True,
            warmup_ratio=self.warmup_ratio,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            logging_dir='./logs',
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.epochs,
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
        trainer.train()


# ------------------------------ helpers ---------------------------------
def csv_to_ft_lists(csv_path, test_frac=0.1, seed=42, cols=None):
    df = pd.read_csv(csv_path, usecols=cols).sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    return train_df.to_dict("records"), test_df.to_dict("records")

# --------------------------- main script --------------------------------
if __name__ == "__main__":
    # ------------ format baseline ------------------
    CSV_FILENAME = 'Dataset/wigner_refactor.csv'
    CIR_CSV  = "case_study_circuit_patched.csv"
    ENT_CSV  = "case_study_entanglement.csv" 
    data = pd.read_csv(CSV_FILENAME)
    
    required_columns = ['image', 'ground_truth']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # Shuffle the dataset with a controlled random state
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Filter out data you don't want (e.g., remove type "Number state")
    # data = data[data['type'] != 'Number state']
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Drop rows with NaN and reset indices
    train_data = train_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)
    
    print(len(train_data), len(test_data))
    
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

    # Prepare your train_data
    x_train   = train_data['image'][:]
    images    = train_data['image'][:]
    y_train   = train_data['ground_truth'][:]
    prompts   = BEST_PROMPT

    fine_tune_data = []
    # for i in range(len(x_train)):
    #     fine_tune_data.append({
    #         "image": images[i],
    #         "input": prompts,
    #         "output": y_train[i],
    #     })

    # For evaluation: pick just 2 or 3 rows from test_data
    eval_subset = test_data.iloc[:3].copy()
    
    x_eval    = eval_subset['image'][:]
    img_eval  = eval_subset['image'][:]
    y_eval    = eval_subset['ground_truth'][:]
    p_eval    = BEST_PROMPT

    eval_data = []
    # for i in range(len(x_eval)):
    #     eval_data.append({
    #         "image": img_eval[i],
    #         "input": p_eval,
    #         "output": y_eval[i],
    #     })

    # ------------ load extra case-study datasets ---
    circuit_cols = ['image', 'input_5', 'output_5', 'input_6', 'output_6', 'input_7', 'output_7']
    circuit_train, circuit_eval = csv_to_ft_lists(CIR_CSV, cols=circuit_cols)

    # Rename turns 5,6,7 -> 1,2,3 so format_data_multiturn reads them sequentially
    rename_map = {
        'input_5': 'input_1', 'output_5': 'output_1',
        'input_6': 'input_2', 'output_6': 'output_2',
        'input_7': 'input_3', 'output_7': 'output_3',
    }
    circuit_train = [{rename_map.get(k, k): v for k, v in row.items()} for row in circuit_train]
    circuit_eval  = [{rename_map.get(k, k): v for k, v in row.items()} for row in circuit_eval]

    entangle_train, entangle_eval = csv_to_ft_lists(ENT_CSV)
    circuit_eval   = circuit_eval[:3]
    entangle_eval  = entangle_eval[:3]

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

    finetuner.run(
        # extra_train1=circuit_train,
        # extra_test1=circuit_eval,
        extra_train2=entangle_train,
        extra_test2=entangle_eval,
    )