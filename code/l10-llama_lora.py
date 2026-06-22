# torchrun --nproc_per_node=2 llama_lora.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

import torch

# ============================================================
# 1. model
# ============================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# ============================================================
# 2. LoRA
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ============================================================
# 3. dataset (Dolly 15k)
# ============================================================

dataset = load_dataset("databricks/databricks-dolly-15k")

MAX_LENGTH = 256  # ⚠️ 比 Alpaca 更稳（很重要）

# ============================================================
# 4. format
# ============================================================

def format_example(example):

    instruction = example["instruction"]
    context = example.get("context", "")
    response = example["response"]

    if context and context.strip() != "":
        text = f"""### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{response}"""

    return text

# ============================================================
# 5. tokenize
# ============================================================

def tokenize_function(example):

    text = format_example(example)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names,
)

# ============================================================
# 6. collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ============================================================
# 7. training args
# ============================================================

training_args = TrainingArguments(
    output_dir="./llama3-dolly-lora",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=1,

    learning_rate=2e-4,

    bf16=True,

    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    optim="paged_adamw_32bit",

    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    gradient_checkpointing=True,

    report_to="none",
)

# ============================================================
# 8. trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# ============================================================
# 9. save
# ============================================================

model.save_pretrained("./llama3-dolly-lora-adapter")
tokenizer.save_pretrained("./llama3-dolly-lora-adapter")

print("Training finished!")
