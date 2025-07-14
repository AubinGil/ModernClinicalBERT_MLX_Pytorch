🔍 Predicting 30-Day Hospital Readmission from Discharge Notes

This project develops a deep learning model to predict whether a patient will be readmitted to the hospital within 30 days, using their structured discharge notes.

🧠 Project Overview

Objective: Predict unplanned hospital readmissions using long clinical discharge notes and structured fields.
Approach: Leverage a fine-tuned Clinical_ModernBERT transformer model with chunked input and focal loss to handle:
Long text sequences exceeding the 512-token limit.
Class imbalance in readmission outcomes.
🚀 Model Features

🔗 Clinical BERT Backbone: Simonlee711/Clinical_ModernBERT
🧩 Chunked Input Handling: Splits long notes into 512-token chunks and aggregates BERT outputs via mean pooling.
⚖️ Focal Loss: Tackles class imbalance by focusing on hard examples.
⚡️ Mixed Precision (AMP): Boosts training speed and efficiency on GPUs.
📉 Metrics Tracked: Accuracy, F1 score, ROC-AUC, and Confusion Matrix.
📊 Dataset

Source: De-identified dataset of hospital discharge notes with binary labels (Readmitted vs Not Readmitted).
Input: Combination of clinical note sections such as:
Chief Complaint
History of Present Illness
Major Procedure
Discharge Diagnosis & Instructions
Target: Binary label indicating 30-day unplanned readmission.
📈 Results

Final AUC: 0.7111
F1 Score: Balanced across classes (Not Readmitted: 0.63 | Readmitted: 0.67)
Model: Efficient at identifying patients at risk of readmission using only discharge documentation.
📁 Project Structure

├── run_readmission.py        # Main training/evaluation script
├── models/                   # Model checkpoints
├── data/                     # Input data (CSV format)
├── utils/                    # Custom loss functions, chunking utilities
├── outputs/                  # Logs, plots, and final metrics
✅ Future Work

Add threshold tuning for better F1 optimization
Integrate structured features (labs, vitals) alongside text
Support multi-task prediction (e.g., mortality + readmission)
📌 Requirements

Python ≥ 3.8
PyTorch ≥ 2.0
Transformers ≥ 4.36
tqdm, scikit-learn, pandas, matplotlib
Let me know if you'd like to include example predictions, a link to the dataset (if public), or a demo notebook!

You said:
I actually used mimic data
ChatGPT said:
Great — here's an updated version of the GitHub README summary reflecting that your project uses MIMIC-IV data:

🔍 Predicting 30-Day Hospital Readmission from Discharge Notes (MIMIC-IV)

This project leverages discharge summaries from the MIMIC-IV clinical dataset to predict whether a patient will be readmitted within 30 days of hospital discharge using deep learning and transformer-based language models.

🧠 Project Overview

Objective: Predict unplanned hospital readmissions using discharge notes from ICU stays in MIMIC-IV.
Approach: Fine-tune a domain-specific BERT model on chunked clinical text with focal loss to address class imbalance and long input sequences.
🚀 Model Architecture

🧠 Base Model: Simonlee711/Clinical_ModernBERT
✂️ Chunked Input Processing: Handles long notes (up to 10,000 tokens) by splitting into 512-token chunks and aggregating BERT outputs with mean pooling.
🎯 Focal Loss: Custom loss function to focus on harder-to-classify cases and deal with class imbalance.
⚡️ AMP (Mixed Precision): Accelerates training on GPU using PyTorch’s torch.cuda.amp.
📊 Dataset: MIMIC-IV

Source: MIMIC-IV v2.2
Cohort: Adult ICU patients with documented discharge summaries.
Inputs: Combined clinical note sections from the noteevents and structured columns, including:
Chief Complaint
History of Present Illness
Major Procedure
Brief Hospital Course
Discharge Diagnosis
Discharge Instructions
Target: Binary label for 30-day unplanned hospital readmission.
📈 Performance (Validation/Test)

Metric	Score
AUC	0.7111
F1 Score	0.67 (Readmitted), 0.63 (Not Readmitted)
Accuracy	~65%
📁 Project Structure

├── run_readmission.py        # Main training and evaluation script
├── chunking.py               # Chunked dataset logic for long text
├── focal_loss.py             # Custom focal loss implementation
├── data/                     # Preprocessed MIMIC-IV CSVs (train/val/test)
├── outputs/                  # Model checkpoints, metrics, and plots
✅ Features

Training logs and performance curves (loss, AUC, F1)
Confusion matrix visualization at evaluation
Resumable training with model checkpointing
Supports evaluation-only mode for inference
🔧 Setup

Requirements:

Python ≥ 3.8
PyTorch ≥ 2.0
Huggingface transformers
scikit-learn, pandas, tqdm, matplotlib
pip install -r requirements.txt
🧪 Future Enhancements

Multi-modal model with labs, vitals, and demographics
Time-aware models (e.g., ClinicalBERT + LSTM)
Threshold tuning for optimal F1 or recall
Integration with hospital triage dashboards
