# ==============================================================================
# VeriMath: Google Colab Fine-Tuning Script (Unsloth + QLoRA)
# ==============================================================================
# Run this entire script on a FREE Google Colab T4 GPU (Runtime -> Change runtime type -> T4 GPU)
# Training time: ~15 to 20 minutes for 1,000 instruction pairs.
# ==============================================================================

# Step 1: Install high-performance Unsloth and training dependencies
# In Colab, run this in a code cell:
"""
!pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ------------------------------------------------------------------------------
# 1. Configuration & Model Selection (Optimized for Raspberry Pi 4, 4GB RAM)
# ------------------------------------------------------------------------------
max_seq_length = 2048
dtype = None # Auto detection (Float16 for T4, Bfloat16 for Ampere+)
load_in_4bit = True # 4bit quantization to fit comfortably in Colab's 15 GB VRAM

# MODEL SELECTION GUIDE FOR RASPBERRY PI 4 (4 GB RAM):
# 1. "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"  [RECOMMENDED FOR PI 4]
#    - Model file size: ~980 MB (Q4_K_M)
#    - RAM usage on Pi 4: ~1.2 GB (leaves 2.5 GB free for OS & Lean 4)
#    - Pi 4 CPU speed: 8-12 tokens/sec (~4s per extraction)
#    - Unbeatable math reasoning for its compact size.
#
# 2. "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"  [FASTEST / LIGHTEST]
#    - Model file size: ~750 MB (Q4_K_M)
#    - RAM usage on Pi 4: ~0.9 GB
#    - Pi 4 CPU speed: 12-15 tokens/sec (~3s per extraction)
#
# 3. "unsloth/Phi-3-mini-4k-instruct-bnb-4bit" [HEAVY / SLOW ON PI 4]
#    - Model file size: 2.39 GB (Q4_K_M)
#    - RAM usage on Pi 4: ~2.8 GB (hits 90%+ RAM on a 4GB Pi 4)
#    - Pi 4 CPU speed: 1.5-2 tokens/sec (~35s per extraction)

model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" # Default: Best for Pi 4 4GB

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ------------------------------------------------------------------------------
# 2. Add LoRA Adapters (Parameter-Efficient Fine-Tuning)
# ------------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16, # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0, # Optimized 0 dropout for Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ------------------------------------------------------------------------------
# 3. Format the Dataset for ChatML / Instruction Tuning
# ------------------------------------------------------------------------------
# Upload 'verimath_finetune_dataset.jsonl' to Colab's file browser
dataset = load_dataset("json", data_files="verimath_finetune_dataset.jsonl", split="train")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# ------------------------------------------------------------------------------
# 4. Supervised Fine-Tuning Trainer (SFTTrainer)
# ------------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs",
    ),
)

print("Starting Fine-Tuning on Colab GPU...")
trainer.train()

# ------------------------------------------------------------------------------
# 5. Interactive Testing (Verifying the Model Handles Incomplete Problems!)
# ------------------------------------------------------------------------------
FastLanguageModel.for_inference(model)

test_problem_incomplete = (
    "A water tank has the shape of an inverted right circular cone with a height of 12 m "
    "and a base radius of 4 m. Water is being pumped into the tank at a constant rate of 5 m^3/min."
)

test_problem_complete = (
    "A water tank has the shape of an inverted right circular cone with a height of 12 m "
    "and a base radius of 4 m. Water is being pumped into the tank at a constant rate of 5 m^3/min. "
    "How fast is the water level rising when the water is 8 m deep?"
)

for label, prob in [("INCOMPLETE TEST", test_problem_incomplete), ("COMPLETE TEST", test_problem_complete)]:
    messages = [
        {"role": "system", "content": "You are VeriMath's Precision Autoformalization Extractor. Extract parameters into JSON and check completeness."},
        {"role": "user", "content": prob}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True)
    decoded = tokenizer.batch_decode(outputs)
    print(f"\n--- {label} ---")
    print(decoded[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip())

# ------------------------------------------------------------------------------
# 6. Save and Export to GGUF Format for VeriMath
# ------------------------------------------------------------------------------
# This generates a quantized GGUF file that you can directly drop into verimath/models/
model.save_pretrained_gguf("qwen2_5_1_5b_verimath_pi4", tokenizer, quantization_method="q4_k_m")
print("\nSuccess! Download 'qwen2_5_1_5b_verimath_pi4-q4_k_m.gguf' and place it in your 'models/' directory.")
