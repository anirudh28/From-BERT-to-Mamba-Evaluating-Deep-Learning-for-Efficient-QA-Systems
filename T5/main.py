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

import warnings
warnings.filterwarnings("ignore")


def prepare_data(data):
    articles = []
    
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                question = qa["question"]

                if not qa["is_impossible"]:
                  answer = qa["answers"][0]["text"]
                
                inputs = {"context": paragraph["context"], "question": question, "answer": answer}

            
                articles.append(inputs)

    return articles


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["context"]
        self.answer = self.data['answer']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]
        
        question_tokenized = self.tokenizer(question, context, max_length=self.q_len, padding="max_length",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length", 
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)
        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }
    
# Dataloader


def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
  
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    
    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], 
                            references=[ref_answer])
    
        print("Context: \n", context)
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer
    
