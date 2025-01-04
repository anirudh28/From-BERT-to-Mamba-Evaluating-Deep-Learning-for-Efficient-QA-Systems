# BERT Fine-Tuning and Static Model Pruning

This repository contains experiments on fine-tuning and pruning the BERT model for developing a Question Answering System. The work demonstrates the application of Static Model Pruning (SMP) with and without Knowledge Distillation (KD) to optimize the model's performance while reducing its size.

## Structure of the Repository

- **`BERT_FINETUNE_QA.ipynb`**
  - This notebook fine-tunes the BERT-Base-Uncased model on the SQuAD dataset.
  - The fine-tuned model is evaluated on Exact Match (EM) score.

- **`Static_Model_Pruning/`**
  - Contains experiments with Static Model Pruning (SMP) applied to a pre-trained BERT model.
  - Two main notebooks:
    - **`BERT_SMP.ipynb`**: Static Model Pruning without Knowledge Distillation.
    - **`BERT_SMP_KD.ipynb`**: Static Model Pruning with Knowledge Distillation. Link to the teacher model [Teacher Model](https://drive.google.com/file/d/15zLGpdTrBlChjc_lYBJu3tln-YS1dzTW/view)

  - The results from these experiments are summarized in a table (see below).

## Summary of Results

The experiments compare the performance of the models in terms of remaining weights, model size, and accuracy (Exact Match - EM) across different configurations. The results demonstrate the trade-offs between model efficiency and accuracy.

| **Model**                             | **Remaining Weights** | **Model Size**    | **Accuracy (EM)** |
|---------------------------------------|-----------------------|-------------------|--------------------|
| BERT-Base-Uncased + Fine-Tuning       | 100%                 | ~110 million      | 0.65              |
| Static Model Pruning                  | 10%                  | ~85 million       | 0.75              |
| Static Model Pruning + Knowledge Distillation | 50%          | ~85 million       | 0.82              |

### Key Points
- **Fine-Tuning**: The `BERT_FINETUNE_QA.ipynb` notebook achieves a baseline performance by fine-tuning the BERT-Base-Uncased model on the SQuAD dataset.
- **Static Model Pruning**: The experiments in the `Static_Model_Pruning/` directory apply pruning to reduce the number of active weights, making the model smaller and faster.
- **Knowledge Distillation**: Leveraging a teacher-student training paradigm improves the performance of the pruned model, achieving higher accuracy with 50% of the weights retained.


## References
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- Hugging Face's [Transformers Library](https://github.com/huggingface/transformers)
- [Static Model Pruning](https://github.com/kongds/SMP)


