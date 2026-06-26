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
from datasets import load_dataset

output_dir = './model_qvlm7b-circuit-expert'
model_id = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

# tmux new -s circuit25vl 'python -u restart_circuit_v2.py 2>&1 | tee -a  output-log/circuit25vl-finetune_6focused.txt'


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
            model_name = self.model_id,
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
        
    def format_data_multiturn(self, row):
        if isinstance(row["image"], Image.Image):
            img = row["image"].convert("RGB").resize((1000, 600))
        else:
            try:
                img = Image.open(row["image"]).convert("RGB").resize((1000, 600))
            except Exception as e:
                raise FileNotFoundError(f"Cannot open {row['image']}: {e}")

        messages   = []
        image_sent = False

        for turn in range(1, 7):
            in_key  = f"input_{turn}"
            out_key = f"output_{turn}"
            if in_key not in row or out_key not in row:
                continue

            user_text      = row[in_key]
            assistant_text = row[out_key]

            if user_text is None or assistant_text is None:
                continue
            if isinstance(user_text, float) and math.isnan(user_text):
                continue
            if isinstance(assistant_text, float) and math.isnan(assistant_text):
                continue
            if str(user_text).strip() == "" and str(assistant_text).strip() == "":
                continue

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
            weight_decay = 0.01,            
            lr_scheduler_type = "cosine",   
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


# --------------------------- main script --------------------------------
if __name__ == "__main__":
    HF_DATASET = "CQILAB/QVLM-Circuit"
    print(f"Loading dataset from HuggingFace: {HF_DATASET}")
    hf_dataset = load_dataset(HF_DATASET, split="train")
    print(f"Dataset loaded: {len(hf_dataset)} samples")

    # Shuffle and split into train/test (90/10)
    hf_dataset = hf_dataset.shuffle(seed=42)
    split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    test_dataset  = split["test"]

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    circuit_train = list(train_dataset)
    circuit_eval  = list(test_dataset.select(range(min(3, len(test_dataset)))))

    # ------------ set up and run finetuning --------
    finetuner = FinetuneQwenVL(
        data=[],
        eval_data=[],
        epochs=6,
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
        extra_train1=circuit_train,
        extra_test1=circuit_eval,
    )