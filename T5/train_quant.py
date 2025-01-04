import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import random
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Set random seeds for reproducibility
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load tokenizer and model
TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

# Training settings
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
EPOCHS = 5
MODEL.to(DEVICE)

best_val_loss = float('inf')

# Training loop
for epoch in range(EPOCHS):
    MODEL.train()
    train_loss = 0
    train_batch_count = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1

    # Validation loop
    MODEL.eval()
    val_loss = 0
    val_batch_count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            outputs = MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
            val_loss += outputs.loss.item()
            val_batch_count += 1

    avg_train_loss = train_loss / train_batch_count
    avg_val_loss = val_loss / val_batch_count

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        MODEL.save_pretrained("qa_model")
        TOKENIZER.save_pretrained("qa_tokenizer")
        print("Saved best model.")

# Post-Training Quantization
print("Starting post-training quantization...")

# Move model to CPU for quantization
MODEL.to("cpu")
MODEL.eval()

# Prepare for quantization
MODEL.qconfig = torch.quantization.get_default_qconfig("fbgemm")
MODEL = torch.quantization.prepare(MODEL)

# Calibrate with validation data (a few batches)
for batch in tqdm(val_loader, desc="Calibration"):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    MODEL(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    break  # Only a subset is needed for calibration

# Convert to quantized model
MODEL = torch.quantization.convert(MODEL)
print("Quantization complete.")

# Save quantized model
MODEL.save_pretrained("qa_model_quantized")
TOKENIZER.save_pretrained("qa_tokenizer_quantized")

print("Quantized model saved.")

# Check quantized model size
import os
quantized_model_size = sum(
    os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk("qa_model_quantized") for f in filenames
)
print(f"Quantized Model Size: {quantized_model_size / (1024 * 1024):.2f} MB")
