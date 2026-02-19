import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---------------- CONFIG ----------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "training/instruction_data.jsonl"
OUTPUT_DIR = "training/lora-model"

# ---------------- LOAD DATA ----------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ---------------- FORMAT FUNCTION (NEW TRL STYLE) ----------------
def formatting_func(example):
    return f"""### Instruction:
{example['instruction']}

### Context:
{example['input']}

### Response:
{example['output']}"""

# ---------------- MODEL ----------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="auto"
)

# ---------------- LORA ----------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training complete! Model saved to:", OUTPUT_DIR)
