import torch
from tqdm import tqdm
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    get_scheduler,
    AdamW,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler

# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load tokenizer and model
TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained(
    "t5-base",
    device_map="auto",
    load_in_8bit=True,  # Use 8-bit precision for base model weights
)

# Prepare model for QLoRA
MODEL = prepare_model_for_kbit_training(MODEL)

# Add LoRA configurations
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],  # Apply LoRA to specific layers (query and value in attention)
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
MODEL = get_peft_model(MODEL, lora_config)
MODEL.print_trainable_parameters()

# Load data
with open('./train.json') as f:
    train_data = json.load(f)
with open('./eval.json') as f:
    eval_data = json.load(f)

# Data preparation (custom function and class assumed already defined)
train_data = pd.DataFrame(prepare_data(train_data))
eval_data = pd.DataFrame(prepare_data(eval_data))

train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(eval_data.index)

qa_dataset = QA_Dataset(TOKENIZER, train_data, 256, 32)
train_loader = DataLoader(qa_dataset, batch_size=4, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=4, sampler=val_sampler)

# Optimizer and Scheduler
OPTIMIZER = AdamW(MODEL.parameters(), lr=0.0001)
lr_scheduler = get_scheduler(
    "linear", optimizer=OPTIMIZER, num_warmup_steps=0, num_training_steps=len(train_loader) * 5
)

# Training loop with QLoRA
best_val_loss = float("inf")
EPOCHS = 5
MODEL.to(DEVICE)

for epoch in range(EPOCHS):
    MODEL.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        lr_scheduler.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    MODEL.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        MODEL.save_pretrained("qlora_model")
        TOKENIZER.save_pretrained("qlora_tokenizer")
        print("Saved best model.")

# Post-training quantization (INT8)
print("Starting quantization...")

MODEL.to("cpu")
quantized_model = torch.quantization.quantize_dynamic(
    MODEL,
    {torch.nn.Linear},  # Specify layers for quantization
    dtype=torch.qint8,  # INT8 quantization
)

# Save quantized model
quantized_model.save_pretrained("qlora_quantized_model")
TOKENIZER.save_pretrained("qlora_quantized_tokenizer")

print("Quantized model saved successfully!")

# Check quantized model size
import os
quantized_model_size = sum(
    os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk("qlora_quantized_model") for f in filenames
)
print(f"Quantized Model Size: {quantized_model_size / (1024 * 1024):.2f} MB")
