# 🏥 Predicting 30-Day Hospital Readmission from Discharge Notes (MIMIC-IV)

This repository explores clinical language modeling to predict whether a patient will be readmitted within 30 days of hospital discharge. Leveraging discharge summaries from the MIMIC-IV dataset, the model fine-tunes a domain-specific BERT architecture to capture subtle linguistic cues associated with readmission risks.

## 📌 Project Overview

- **Objective:** Predict unplanned hospital readmissions using discharge notes from ICU stays in MIMIC-IV.
- **Approach:** Fine-tune `Clinical_ModernBERT` on chunked discharge summaries using a custom focal loss function to address long input sequences and class imbalance.
- **Key Features:**
  - ✅ Chunk-based input handling for long notes
  - ✅ Focal loss implementation
  - ✅ Mixed precision training with AMP
  - ✅ Evaluation with AUC, F1 score, and accuracy

## 🧠 Model Architecture

| Component              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 🧠 Base Model          | `Simonlee711/Clinical_ModernBERT` (Transformer-based language model)        |
| ✂️ Chunked Input        | Split notes into 512-token chunks; apply mean pooling over BERT outputs     |
| 🎯 Loss Function        | Custom focal loss to prioritize hard-to-classify cases                      |
| ⚡️ Training Acceleration| Enabled via `torch.cuda.amp` (Automatic Mixed Precision)                    |

## 🗂 Dataset: MIMIC-IV

- **Source:** MIMIC-IV v2.2
- **Cohort:** Adult ICU patients with discharge summaries
- **Inputs:** Combined sections from `noteevents` and structured EHR columns:
  - Chief Complaint
  - History of Present Illness
  - Major Procedure
  - Brief Hospital Course
  - Discharge Diagnosis
  - Discharge Instructions
- **Target:** Binary label for 30-day readmission

## 📊 Performance (Validation/Test)

| Metric   | Score                   |
|----------|------------------------|
| AUC      | 0.7111                  |
| F1 Score | 0.67 (Readmitted)       |
|          | 0.63 (Not Readmitted)   |
| Accuracy | ~65%                    |

## 📁 Project Structure

```bash
├── run_readmission.py        # Main training and evaluation script
├── outputs/                  # Saved model checkpoints, metrics, plots
├── requirements.txt          # Required dependencies


Python ≥ 3.8  
PyTorch ≥ 2.0  
Huggingface transformers  
scikit-learn  
pandas  
tqdm  
matplotlib


python run_readmission.py \
  --task_name readmission \
  --do_train \
  --do_eval \
  --data_dir /path/to/data \
  --bert_model Simonlee711/Clinical_ModernBERT \
  --output_dir /path/to/save \
  --num_train_epochs 5 \
  --train_batch_size 8 \
  --max_seq_length 512 \
  --learning_rate 5e-6
```
## Acknowlegment 
Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58

