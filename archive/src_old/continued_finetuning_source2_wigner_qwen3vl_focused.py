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

output_dir = './model_cot_qwen3vl_8b_ent'
model_id = "./model_cot_qwen3vl_8b_4bit_thinking_epoch2_continued/checkpoint-1325"

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
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
            device_map = "auto",
        )
        
        # 2) Wrap with PEFT / LoRA
        # self.model = FastVisionModel.get_peft_model(
        #     self.base_model,
        #     finetune_vision_layers     = True, # set True if you want vision layers updated
        #     finetune_language_layers   = True, # set True if you want language layers updated
        #     finetune_attention_modules = True,
        #     finetune_mlp_modules       = True,
        #     r = peft_r,
        #     lora_alpha = peft_alpha,
        #     lora_dropout = peft_dropout,
        #     bias = "none",
        #     random_state = 3407,
        #     use_rslora = False,
        #     loftq_config = None
        # )
        
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim
        self.data = data
        self.eval_data = eval_data

    def format_data(self, row):
        img = Image.open(row["image"]).convert("RGB").resize((1000, 600))

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
                        {"type": "text",  "text": f"{row['output']}"},
                    ],
                },
            ]
        }


    def generate_purity_reasoning(self, purity_value, entangled_status="Yes"):
        """
        Generate synthetic chain-of-thought reasoning for purity predictions.
        This helps the model learn to associate visual features with decimal values.
        
        Args:
            purity_value: The ground truth purity (0.0 to 1.0)
            entangled_status: Whether the circuit is entangled ("Yes" or "No")
        
        Returns:
            A reasoning string that explains the purity assessment
        """
        try:
            purity = float(purity_value)
        except (ValueError, TypeError):
            # If we can't parse, return minimal reasoning
            return "Unable to determine purity from the circuit visualization."
        
        # Generate reasoning based on purity ranges
        # IMPORTANT: Do NOT include the actual decimal value in reasoning to prevent data leakage
        if purity >= 0.99:
            reasoning = (
                "The quantum circuit diagram shows a pure quantum state. "
                "The gate structure appears to preserve coherence throughout, "
                "with minimal decoherence or mixing effects visible. "
                "The state evolution maintains near-perfect purity."
            )
        elif purity >= 0.95:
            reasoning = (
                "The circuit exhibits very high purity. "
                "The gate operations show excellent coherence preservation, "
                "with only minimal environmental coupling or decoherence. "
                "The state is very close to pure but shows slight mixing."
            )
        elif purity >= 0.85:
            reasoning = (
                "Analyzing the circuit structure, I observe high purity. "
                "While the quantum operations maintain good coherence, "
                "there are some signs of decoherence or partial tracing effects. "
                "The state is predominantly pure with moderate mixing."
            )
        elif purity >= 0.70:
            reasoning = (
                "The circuit demonstrates moderate purity. "
                "The gate sequence shows noticeable decoherence effects, "
                "suggesting interaction with the environment or partial measurements. "
                "The state exhibits significant but not dominant mixing."
            )
        elif purity >= 0.50:
            reasoning = (
                "The purity appears to be in the medium range. "
                "The circuit shows substantial mixing, possibly from strong environmental coupling "
                "or measurement-induced decoherence. The pure and mixed components are comparable."
            )
        else:
            reasoning = (
                "The circuit indicates low purity. "
                "Strong decoherence and mixing effects are evident, "
                "suggesting significant environmental interaction or thermal effects. "
                "The state is heavily mixed."
            )
        
        # Add entanglement context if applicable
        if entangled_status == "Yes" and purity < 1.0:
            reasoning += " The presence of entanglement combined with sub-unity purity suggests a mixed entangled state."
        elif entangled_status == "No" and purity == 1.0:
            reasoning += " The separable structure with perfect purity indicates a pure product state."
        
        return reasoning

    def format_data_multiturn(self, row):
        """
        Format multi-turn conversation data.
        Turn 1: Entanglement question (Yes/No)
        Turn 2: Purity question (decimal value with reasoning)
        """
        img = Image.open(row["image"]).convert("RGB").resize((1000, 600))

        messages = []
        
        # ===== TURN 1: Entanglement Question =====
        if 'input_1' in row and 'output_1' in row:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": row['input_1']}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": row['output_1']}]
            })
        
        # ===== TURN 2: Purity Question with Chain-of-Thought =====
        if 'input_2' in row and 'output_2' in row:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": row['input_2']}]
            })
            
            # Generate reasoning for purity prediction
            entangled_status = row.get('output_1', 'Yes')
            reasoning = self.generate_purity_reasoning(row['output_2'], entangled_status)
            
            # CRITICAL FIX: Answer must be OUTSIDE think tags
            # Format: <think>reasoning</think>\n{value}
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"<think>{reasoning}</think>\n{row['output_2']}"}]
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
        trainer.train(resume_from_checkpoint=False)


# ------------------------------ helpers ---------------------------------
def csv_to_ft_lists(csv_path, test_frac=0.1, seed=42):
    df = pd.read_csv(csv_path)
    
    # --- Balancing Logic ---
    # We want to balance "Pure" (purity=1.0) vs "Mixed" (purity < 1.0)
    # The relevant column is 'output_2' for this specific CSV use case (entanglement study).
    # If using for other CSVs, this might need adjustment, but for now we target ENT_CSV.
    
    if 'output_2' in df.columns:
        # Helper to check if value is effectively 1.0
        def is_pure(val):
            try:
                # Some might be strings, handle conversion
                v = float(str(val).strip())
                return abs(v - 1.0) < 1e-6
            except:
                return False

        df['is_pure'] = df['output_2'].apply(is_pure)
        
        df_pure = df[df['is_pure'] == True]
        df_mixed = df[df['is_pure'] == False]
        
        # Undersample pure to match mixed (or at least reduce it)
        # If pure is much larger than mixed, we sample len(mixed) * 1.0 (1:1 ratio)
        n_mixed = len(df_mixed)
        if len(df_pure) > n_mixed:
            print(f"Balancing dataset: Found {len(df_pure)} Pure (1.0) and {n_mixed} Mixed.")
            df_pure_sampled = df_pure.sample(n=n_mixed, random_state=seed)
            df = pd.concat([df_pure_sampled, df_mixed])
            print(f"New dataset size: {len(df)} (Balanced 1:1)")
        else:
            print(f"Dataset already balanced or Mixed > Pure (Pure: {len(df_pure)}, Mixed: {n_mixed})")
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    return train_df.to_dict("records"), test_df.to_dict("records")

# --------------------------- main script --------------------------------
if __name__ == "__main__":
    # BASE_CSV = "Dataset/wigner_analysis_results_combined.csv"
    # CIR_CSV  = "case_study_circuit_patched.csv"
    ENT_CSV  = "case_study_entanglement.csv"

    BEST_PROMPT = (
        "You are given a grayscale image representing a quantum optical state. "
        "Your task is to determine the type of the state (e.g., cat state, Fock state, "
        "coherent state, thermal state, random state etc.) as well as its key parameters "
        "(alpha/number of photons/density, number of qubits, and the linear space range). "
        "Please provide your answer in the format: "
        "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] "
        "equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\" "
        "then extract your opinion on how you determine state, parameters, number of qubit "
        "from the image."
    )

    # ------------ format baseline ------------------
    # CSV_FILENAME = 'Dataset/wigner_analysis_results_combined.csv' 
    # data = pd.read_csv(CSV_FILENAME)
    
    # required_columns = ['image', 'ground_truth']
    # for column in required_columns:
    #     if column not in data.columns:
    #         raise ValueError(f"Column '{column}' not found in the CSV file.")
    
    # # Shuffle the dataset with a controlled random state
    # data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # # Filter out data you don't want (e.g., remove type "Number state")
    # # data = data[data['type'] != 'Number state']
    
    # # Split data into train and test sets
    # train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # # Drop rows with NaN and reset indices
    # train_data = train_data.reset_index(drop=True)
    # test_data  = test_data.reset_index(drop=True)
    
    # print(len(train_data), len(test_data))
    
    # BEST_PROMPT = (
    #     "You are given a grayscale image representing a quantum optical state. "
    #     "Your task is to determine the type of the state (e.g., cat state, Fock state, coherent state, thermal state, random state etc.) "
    #     "as well as its key parameters (alpha/number of photons/density, number of qubits, and the linear space range). "
    #     "Please provide your answer in the format: "
    #     "\"<think>[THINKING PROCESS]</think> This is a [STATE TYPE] with [KEY parameters] equal to [VALUE], number of qubits equal to [N] in the linear space [LOW] to [HIGH].\""
    #     "then extract your opinion on how you determine state, parameters, number of qubit from the image, "
    # )

    # Prepare your train_data
    # x_train   = train_data['image'][:]
    # images    = train_data['image'][:]
    # y_train   = train_data['ground_truth'][:]
    # prompts   = BEST_PROMPT

    fine_tune_data = []
    # for i in range(len(x_train)):
    #     fine_tune_data.append({
    #         "image": images[i],
    #         "input": prompts,
    #         "output": y_train[i],
    #     })

    # For evaluation: pick just 2 or 3 rows from test_data
    # eval_subset = test_data.iloc[:3].copy()
    
    # x_eval    = eval_subset['image'][:]
    # img_eval  = eval_subset['image'][:]
    # y_eval    = eval_subset['ground_truth'][:]
    # p_eval    = BEST_PROMPT

    eval_data = []
    # for i in range(len(x_eval)):
    #     eval_data.append({
    #         "image": img_eval[i],
    #         "input": p_eval,
    #         "output": y_eval[i],
    #     })

    # ------------ load extra case-study datasets ---
    # circuit_train, circuit_eval = csv_to_ft_lists(CIR_CSV)
    entangle_train, entangle_eval = csv_to_ft_lists(ENT_CSV)
    # circuit_eval   = circuit_eval[:3]
    # entangle_eval  = entangle_eval[:3]

    # ------------ set up and run finetuning --------
    finetuner = FinetuneQwenVL(
        data=fine_tune_data,
        eval_data=eval_data,
        epochs=2,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        gradient_accumulation_steps=16,
        optim="adamw_8bit",
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