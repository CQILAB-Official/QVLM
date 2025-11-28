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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id

        # 1) Load base model and tokenizer
        self.base_model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name = self.model_id,
            load_in_4bit = False,
            use_gradient_checkpointing = "unsloth",
        )
        
        # 2) Wrap with PEFT / LoRA
        self.model = FastVisionModel.get_peft_model(
            self.base_model,
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
        """
        Takes a dictionary with 'image', 'input', 'output'
        and converts it into the Qwen-VL style of instruction data.
        """
        image_path = row["image"]
        input_text = row['input']
        output_text = row['output']
        
        try:
            image = Image.open(image_path).convert("RGB")
            # If needed, you can also resize or transform:
            image = image.resize((1000, 600))
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

    def run(self):
        """
        Executes the fine-tuning process, including evaluation
        on 2-3 test samples at the end of each epoch.
        """
        # Convert your training and evaluation datasets
        converted_train_dataset = [self.format_data(row) for row in self.data]
        converted_eval_dataset  = [self.format_data(row) for row in self.eval_data]
        
        # 3) TrainingArguments / SFTConfig
        training_args = SFTConfig(
            learning_rate=self.learning_rate,
            output_dir='./model_cot_qwen25vl_rerun_v1',
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
            max_seq_length = 2048,
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


if __name__ == "__main__":
    CSV_FILENAME = 'Dataset/wigner_analysis_results_combined.csv' 
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
        "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state etc.) "
        "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
        "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
    )

    # Prepare your train_data
    x_train   = train_data['image'][:]
    images    = train_data['image'][:]
    y_train   = train_data['ground_truth'][:]
    prompts   = BEST_PROMPT

    fine_tune_data = []
    for i in range(len(x_train)):
        fine_tune_data.append({
            "image": images[i],
            "input": prompts,
            "output": y_train[i],
        })

    # For evaluation: pick just 2 or 3 rows from test_data
    eval_subset = test_data.iloc[:3].copy()
    
    x_eval    = eval_subset['image'][:]
    img_eval  = eval_subset['image'][:]
    y_eval    = eval_subset['ground_truth'][:]
    p_eval    = BEST_PROMPT

    eval_data = []
    for i in range(len(x_eval)):
        eval_data.append({
            "image": img_eval[i],
            "input": p_eval,
            "output": y_eval[i],
        })
        
    print(train_data)

    # Instantiate FinetuneQwenVL with both train_data and eval_data
    finetuner = FinetuneQwenVL(
        data=fine_tune_data,
        eval_data=eval_data,  # pass your 2-3 eval samples here
        epochs=2,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        optim="adamw_torch_fused",
        # model_id="unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit",
        # model_id="unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit",
        model_id="unsloth/Qwen2.5-VL-7B-Instruct",
        peft_r=128,
        peft_alpha=128,
        peft_dropout=0.0,
    )

    # Start the fine-tuning (with evaluation each epoch)
    finetuner.run()
