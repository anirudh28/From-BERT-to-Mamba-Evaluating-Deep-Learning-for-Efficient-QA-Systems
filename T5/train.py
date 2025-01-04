import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
import string
import evaluate
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import random
import transformers
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
from main import *

import warnings
warnings.filterwarnings("ignore")

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 256
T_LEN = 32
BATCH_SIZE = 4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('./train.json') as f:
    train_data = json.load(f)

with open('./eval.json') as f:
    eval_data = json.load(f)


train_data = prepare_data(train_data)
eval_data = prepare_data(eval_data)

# Create a Dataframe
train_data = pd.DataFrame(train_data)
eval_data = pd.DataFrame(eval_data)


train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(eval_data.index)

qa_dataset = QA_Dataset(TOKENIZER, train_data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

best_val_loss = float('inf')

for epoch in range(5):
    MODEL.to(DEVICE)
    MODEL.train()
    train_loss = 0
    train_batch_count = 0
    for batch in tqdm(train_loader, desc="Training batches"):
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
    
    # Evaluation
    MODEL.eval()
    val_loss = 0
    val_batch_count = 0


    with torch.no_grad():  # No gradient tracking for evaluation
        for batch in tqdm(val_loader, desc="Validation batches"):
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

    avg_train_loss =  train_loss / train_batch_count
    avg_val_loss = val_loss/val_batch_count

    print(f"{epoch+1}/{5} -> Train loss: {avg_train_loss / train_batch_count}\tValidation loss: {avg_val_loss/val_batch_count}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        MODEL.save_pretrained("qa_model")
        TOKENIZER.save_pretrained("qa_tokenizer")
        print("Best model saved based on validation loss.")