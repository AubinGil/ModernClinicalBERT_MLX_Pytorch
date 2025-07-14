#!/usr/bin/env python
import os, re, argparse
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--bert_model", type=str, required=True)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Data loading ---
def load_dataset(split):
    df = pd.read_csv(os.path.join(args.data_dir, f"{split}_structured.csv"))
    df = df.dropna(subset=["Label"]).copy()
    df["TEXT"] = df[[
        "chief_complaint", "major_procedure", "hpi",
        "brief_course", "discharge_diagnosis", "discharge_instructions"
    ]].fillna("").agg(" ".join, axis=1)
    return df

df_train = load_dataset("train") if args.do_train else None
df_val = load_dataset("val") if args.do_train else None
df_test = load_dataset("test") if args.do_eval else None

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

# --- Chunked Dataset ---
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
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "doc_chunk_lens": doc_chunk_lens}

# --- Model ---
class ChunkedClassifier(torch.nn.Module):
    def __init__(self, base_model_name, dropout=0, freeze_up_to=None, freeze_embeddings=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)

        # --- Optionally freeze embeddings ---
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        # --- Optionally freeze encoder layers up to `freeze_up_to` ---
        if freeze_up_to is not None:
            for i in range(freeze_up_to + 1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # --- Classifier Head ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, doc_chunk_lens):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        chunks = torch.split(outputs.pooler_output, doc_chunk_lens)
        doc_embs = torch.stack([chunk.mean(dim=0) for chunk in chunks])
        return self.classifier(doc_embs).squeeze(-1)


model = ChunkedClassifier(args.bert_model).to(device)
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
total_steps = len(df_train)//args.train_batch_size * args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
# Compute from the full training data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df_train["Label"]),
    y=df_train["Label"]
)
pos_weight = torch.tensor(class_weights[1], dtype=torch.float32).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

scaler = torch.cuda.amp.GradScaler()

# Compute from the full training data
from sklearn.utils.class_weight import compute_class_weight


if args.do_train:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report

    train_loader = DataLoader(
        ChunkedDataset(df_train), batch_size=args.train_batch_size,
        shuffle=True, collate_fn=chunk_collate_fn
    )
    val_loader = DataLoader(
        ChunkedDataset(df_val), batch_size=args.train_batch_size,
        shuffle=False, collate_fn=chunk_collate_fn
    )

    losses = []
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float32):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['doc_chunk_lens'])
                loss = criterion(logits, batch['labels'].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))

        # Record & show training loss
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        # --- Validation / Evaluation ---
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['doc_chunk_lens'])
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                val_preds.extend(preds.tolist())
                val_probs.extend(probs.tolist())
                val_labels.extend(batch['labels'].cpu().numpy().tolist())

        auc = roc_auc_score(val_labels, val_probs)
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)

        print(f"🔍 Epoch {epoch} Validation ▶ AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(val_labels, val_preds)
        cm_df = pd.DataFrame(cm, index=["Not Readmitted", "Readmitted"], columns=["Pred Not", "Pred Yes"])
        print("\n📊 Contingency Table:")
        print(cm_df)

    # --- Plot Loss Curve ---
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(losses)), losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


