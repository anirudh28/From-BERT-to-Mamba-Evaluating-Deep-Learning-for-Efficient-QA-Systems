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
from datasets import load_metric


import warnings
warnings.filterwarnings("ignore")

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


TOKENIZER = T5TokenizerFast.from_pretrained("./qa_tokenizer")
MODEL = T5ForConditionalGeneration.from_pretrained("./qa_model")
Q_LEN = 256
T_LEN = 32
BATCH_SIZE = 4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('./eval.json') as f:
    eval_data = json.load(f)

eval_data = prepare_data(eval_data)

# Create a Dataframe
eval_data = pd.DataFrame(eval_data)

val_sampler = RandomSampler(eval_data.index)

qa_dataset = QA_Dataset(TOKENIZER, eval_data, Q_LEN, T_LEN)

val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

def normalize_text(s):
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

# Evaluation
MODEL.to(DEVICE)
MODEL.eval()
total_exact_matches = 0
total_questions = 0

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
        
        outputs['logits'] = torch.argmax(outputs['logits'], dim=-1)
        
        predictions = TOKENIZER.batch_decode(outputs['logits'], skip_special_tokens=True)
        answers = TOKENIZER.batch_decode(labels*decoder_attention_mask, skip_special_tokens=True)

        # Calculate exact match for each QA pair
        for pred, answer in zip(predictions, answers):
            total_exact_matches += compute_exact_match(pred, answer)
            total_questions += 1


EM = (total_exact_matches / total_questions) * 100

print(f"Exact Match: {EM}")