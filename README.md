🔍 Predicting 30-Day Hospital Readmission from Discharge Notes (MIMIC-IV)
This project leverages discharge summaries from the MIMIC-IV clinical dataset to predict whether a patient will be readmitted within 30 days of hospital discharge using deep learning and transformer-based language models.


🧠 Project Overview
•	Objective: Predict unplanned hospital readmissions using discharge notes from ICU stays in MIMIC-IV.
•	Approach: Fine-tune a domain-specific BERT model on chunked clinical text with focal loss to address class imbalance and long input sequences.
🚀 Model Architecture
•	🧠 Base Model: Simonlee711/Clinical_ModernBERT
•	✂️ Chunked Input Processing: Handles long notes (up to 10,000 tokens) by splitting into 512-token chunks and aggregating BERT outputs with mean pooling.
•	🎯 Focal Loss: Custom loss function to focus on harder-to-classify cases and deal with class imbalance.
•	⚡️ AMP (Mixed Precision): Accelerates training on GPU using PyTorch’s torch.cuda.amp.


📊 Dataset: MIMIC-IV
•	Source: MIMIC-IV v2.2
•	Cohort: Adult ICU patients with documented discharge summaries.
•	Inputs: Combined clinical note sections from the noteevents and structured columns, including:
o	Chief Complaint
o	History of Present Illness
o	Major Procedure
o	Brief Hospital Course
o	Discharge Diagnosis
o	Discharge Instructions
•	Target: Binary label for 30-day unplanned hospital readmission.

📈 Performance (Validation/Test)
Metric	Score
AUC	0.7111
F1 Score	0.67 (Readmitted), 0.63 (Not Readmitted)
Accuracy	~65%

📁 Project Structure
├── run_readmission.py        # Main training and evaluation script
├── outputs/                  # Model checkpoints, metrics, and plots


🔧 Setup
Requirements:
•	Python ≥ 3.8
•	PyTorch ≥ 2.0
•	Huggingface transformers
•	scikit-learn, pandas, tqdm, matplotlib
pip install -r requirements.txt
🧪 Future Enhancements
•	Multi-modal model with labs, vitals, and demographics
•	Time-aware models (e.g., ClinicalBERT + LSTM)
•	Threshold tuning for optimal F1 or recall
•	Integration with hospital triage dashboards

Call training:
!python run_readmission.py \
  --task_name readmission \
  --do_train \
  --do_eval \
  --data_dir /______ \
  --bert_model Simonlee711/Clinical_ModernBERT \
  --output_dir _____ \
  --num_train_epochs 5  \
  --train_batch_size 8 \
  --max_seq_length 512 \
  --learning_rate 5e-6

