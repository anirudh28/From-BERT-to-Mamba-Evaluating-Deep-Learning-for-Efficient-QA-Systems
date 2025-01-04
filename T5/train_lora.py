import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
import json
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Random seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Load tokenizer and model
TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 5
RANK = 8
LORA_ALPHA = 16
MAX_SEQ_LEN = 256

# LoRA parameterization
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.original_layer = original_layer
        self.lora_A = nn.Parameter(torch.randn((rank, original_layer.weight.shape[1])))
        self.lora_B = nn.Parameter(torch.randn((original_layer.weight.shape[0], rank)))
        self.scale = alpha / rank

    def forward(self, *args, **kwargs):
        W = self.original_layer.weight + (self.lora_B @ self.lora_A) * self.scale
        self.original_layer.weight = nn.Parameter(W)
        return self.original_layer(*args, **kwargs)

# Replace linear layers in the model with LoRA layers
def apply_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Apply LoRA to linear layers
            setattr(model, name, LoRALayer(module))
    return model

MODEL = apply_lora(MODEL)

# Dataset preparation
class QA_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question, answer = row["question"], row["answer"]
        inputs = self.tokenizer(
            question,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            answer,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "decoder_attention_mask": labels["attention_mask"].squeeze(),
        }

# Load QA data
with open("./train.json") as f:
    train_data = json.load(f)
with open("./eval.json") as f:
    eval_data = json.load(f)

train_data = pd.DataFrame(train_data)
eval_data = pd.DataFrame(eval_data)

# Create DataLoaders
train_dataset = QA_Dataset(train_data, TOKENIZER, MAX_SEQ_LEN)
eval_dataset = QA_Dataset(eval_data, TOKENIZER, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(train_dataset))
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, sampler=RandomSampler(eval_dataset))

# Optimizer and loss function
OPTIMIZER = torch.optim.Adam([p for p in MODEL.parameters() if p.requires_grad], lr=LR)

# Training loop
best_val_loss = float("inf")
MODEL.to(DEVICE)

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        loss = outputs.loss
        train_loss += loss.item()
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    # Evaluation
    MODEL.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            outputs = MODEL(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(eval_loader)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        MODEL.save_pretrained("qa_lora_model")
        TOKENIZER.save_pretrained("qa_lora_tokenizer")
        print(f"Best model saved at epoch {epoch+1}.")

