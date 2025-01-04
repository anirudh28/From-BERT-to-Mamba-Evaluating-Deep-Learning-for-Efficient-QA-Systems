# Fine-Tuning T5 with QLoRA and Quantization for QA Tasks

This repository demonstrates a complete pipeline for **fine-tuning the T5-base model** using **QLoRA (Low-Rank Adaptation)** and **quantization** for efficient and memory-optimized Question-Answering tasks. The model is prepared for 8-bit training, enhanced with LoRA layers, and finally quantized to INT8 for deployment.

---

## Features
- Fine-tuning T5-base for QA tasks using LoRA for efficient adaptation.
- Memory-efficient 8-bit training with Hugging Face's `transformers` library.
- Post-training quantization to INT8 for reduced model size and faster inference.
- Fully reproducible with deterministic seeds and step-by-step logging.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Results](#results)

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Hugging Face Transformers 4.33+
- `peft` library for LoRA integration
- Additional Python packages:
  ```bash
  pip install torch transformers datasets peft tqdm scikit-learn

## Setup

### Clone the repository:

``` git clone https://github.com/your-username/qlora-t5-qa.git ```
``` cd qlora-t5-qa ```

### Install required dependencies:

``` pip install -r requirements.txt ```

### Prepare the training and evaluation datasets:

Place your train.json and eval.json files in the root directory.
Format: Each JSON entry must include question, context, and answers.

### Pretrained model and tokenizer:

The code uses t5-base from Hugging Face as the base model.

## Usage

### Train the Model

Run the following script to start fine-tuning with LoRA and post-training quantization:

  ```bash
  python train.py # For normal T5 base training 
  python train_lora.py # For normal T5 base training with LoRA
  python train_quant.py # For normal T5 base training with Post Training Quantization 
  python train_qlora.py # For normal T5 base training with QLoRA
  ```
## Results

| Configuration        | Validation Loss    | Model Size | Performance            |
|-----------------------|--------------------|------------|------------------------|
| T5-base (FP32)       | Baseline           | ~850 MB    | Accurate but large     |
| T5-base + LoRA       | Slightly Worse     | ~900 MB    | Efficient fine-tuning  |
| T5-base + Quantization (INT8)| Slightly Worse     | ~212 MB    | Efficient fine-tuning  |
| T5-base + QLoRA      | Slightly Worse             | ~300 MB    | Memory-efficient       |

