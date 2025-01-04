# **Question Answering Experiments with LSTM-based Models**

This repository contains experiments with different LSTM-based architectures for span-based Question Answering (QA) tasks using the **SQuAD 2.0 dataset**. Various advanced techniques, such as pre-trained embeddings, attention mechanisms, and regularization, have been explored to improve performance.

---

## **Models Overview**

| Feature                | Model\_1                 | Model\_2                 | Model\_3                 | Model\_4                 |
|------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| **Embeddings**         | Random                  | GloVe (pre-trained)      | FastText (pre-trained)   | FastText (pre-trained)   |
| **Attention**          | Attention               | Attention               | Multi-Head Attention     | Self-Attention          |
| **LSTM Configuration** | Bidirectional           | Bidirectional + Regularized | Bidirectional + Residual | Bidirectional + Residual |
| **Pooling**            | Flatten                | Flatten                | GlobalMaxPooling1D       | GlobalMaxPooling1D       |
| **Regularization**     | Dropout                | Dropout, L2            | Dropout, L2             | Dropout, L2             |
| **Dataset Size**       | Training: 10,000 <br> Validation: 2,000 | Training: 20,000 <br> Validation: 5,000 | Training: 60,000 <br> Validation: 11,873 | Training: 60,000 <br> Validation: 11,873 |
| **Model Parameters**   | ~499M                   | ~51M                    | ~36M                    | ~36M                    |
| **Model Size**         | ~1.86GB                | ~195MB                 | ~138MB                 | ~138MB                 |
| **Accuracy (EM)**      | Training: 0.094 <br> Validation: 0.008 | Training: 0.102 <br> Validation: 0.029 | Training: 0.071 <br> Validation: 0.116 | Training: 0.972 <br> Validation: 0.22 |

---

## **Key Features**
1. **Pre-Trained Embeddings:** Experiments with **GloVe** and **FastText** embeddings for semantic representation.
2. **Attention Mechanisms:** Comparison of **Attention**, **Multi-Head Attention**, and **Self-Attention** for focusing on relevant parts of the input.
3. **LSTM Enhancements:**
   - **Bidirectional LSTMs:** Capturing past and future context.
   - **Residual Connections:** Preserving information and improving gradient flow.
4. **Pooling Techniques:** Shift from Flatten to **GlobalMaxPooling1D** for efficient feature extraction.
5. **Regularization:** Use of **Dropout** and **L2 Regularization** to reduce overfitting.

---

## **Dataset**
- **SQuAD 2.0**: Stanford Question Answering Dataset.
  - **Training Set:** 60,000 samples.
  - **Validation Set:** 11,873 samples.

---

## **Model Architectures**
Each model was implemented using TensorFlow and includes:
- Pre-trained embeddings.
- Bidirectional LSTMs.
- Attention mechanisms (varying by model).
- Dense layers for span prediction (start and end positions).

---

## **How to Run**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/question-answering-lstm.git
   cd question-answering-lstm

