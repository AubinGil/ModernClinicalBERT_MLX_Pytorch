import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt

# --- Config ---
MODEL_PATH = "/content/drive/MyDrive/Fair/checkpoint_epoch_1.pt"  # 🔁 change to your checkpoint if needed
TOKENIZER_PATH = "Simonlee711/Clinical_ModernBERT"
VAL_PATH = "/content/val_structured.csv"
MAX_LEN = 512
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# --- Load Validation Data ---
df = pd.read_csv(VAL_PATH)
df["TEXT"] = df[[
    "chief_complaint", "major_procedure", "hpi",
    "brief_course", "discharge_diagnosis", "discharge_instructions"
]].fillna("").agg(" ".join, axis=1)

# --- Dataset + Collate ---
class ChunkedDataset(Dataset):
    def __init__(self, df, max_len=512):
        self.texts = df["TEXT"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer(self.texts[idx], truncation=False)["input_ids"][:8192]
        chunks = [tokens[i:i + self.max_len] for i in range(0, len(tokens), self.max_len)]
        input_ids, attention_mask = [], []
        for chunk in chunks:
            pad_len = self.max_len - len(chunk)
            input_ids.append(chunk + [0] * pad_len)
            attention_mask.append([1] * len(chunk) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

def chunk_collate_fn(batch):
    input_ids = torch.cat([x["input_ids"] for x in batch])
    attention_mask = torch.cat([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch])
    doc_chunk_lens = [len(x["input_ids"]) for x in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "doc_chunk_lens": doc_chunk_lens
    }

# --- Model ---
class ChunkedClassifier(torch.nn.Module):
    def __init__(self, base_model_name, dropout=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, doc_chunk_lens):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        chunks = torch.split(out, doc_chunk_lens)
        doc_embs = torch.stack([chunk.mean(dim=0) for chunk in chunks])
        return self.classifier(doc_embs).squeeze(-1)

# --- Load Model ---
model = ChunkedClassifier(TOKENIZER_PATH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- DataLoader ---
val_loader = DataLoader(
    ChunkedDataset(df), batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=chunk_collate_fn
)

from tqdm.auto import tqdm  # make sure tqdm is imported

# --- Inference with tqdm ---
all_probs, all_preds, all_labels = [], [], []

model.eval()
with torch.no_grad():
    loop = tqdm(val_loader, desc="🔍 Running Inference")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()
        doc_chunk_lens = batch["doc_chunk_lens"]

        logits = model(input_ids, attention_mask, doc_chunk_lens)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.47).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)


# --- Metrics ---
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Not Readmitted", "Readmitted"]))
print(f"\n✅ Final AUC:  {roc_auc_score(all_labels, all_probs):.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
