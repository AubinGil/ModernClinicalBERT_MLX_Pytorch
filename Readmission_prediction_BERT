#!/usr/bin/env python
import os, argparse, re, csv
import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --- 1. Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--bert_model", type=str, required=True)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# --- 2. Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
use_amp = False  # MPS does not support AMP+GradScaler yet
print(f"Using device: {device}")

# --- 3. Data Cleaning & Loading ---
TEXT_FIELDS = [
    "chief_complaint", "major_procedure", "hpi",
    "brief_course", "discharge_diagnosis", "discharge_instructions"
]

def clean_text(text):
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    return text

def load_dataset(split):
    df = pd.read_csv(os.path.join(args.data_dir, f"{split}_structured.csv"))
    df = df.dropna(subset=["Label"]).copy()
    for col in TEXT_FIELDS:
        df[col] = df[col].fillna("")
    df["TEXT"] = df[TEXT_FIELDS].agg(" ".join, axis=1)
    tok_lens = df["TEXT"].astype(str).str.split().str.len()
    df = df[tok_lens >= 20].copy()
    print(f"{split}: Loaded {len(df)} samples.")
    return df

df_train = load_dataset("train") if args.do_train else None
df_val = load_dataset("val") if args.do_train else None
df_test = load_dataset("test") if args.do_eval else None

# --- 4. Tokenizer & Dataset Class ---
tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

class NotesDataset(Dataset):
    def __init__(self, df):
        self.texts = df["TEXT"].tolist()
        self.labels = df["Label"].astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i],
            padding='max_length',
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[i], dtype=torch.float)
        }

# --- 5. Model Definition ---
class CustomClassifier(torch.nn.Module):
    def __init__(self, base_model_name, dropout=0, freeze_up_to=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        
        # --- Freeze encoder layers up to (and including) freeze_up_to ---
        for name, param in self.bert.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in range(0, freeze_up_to + 1)):
                param.requires_grad = False

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output).squeeze(-1)

model = CustomClassifier(args.bert_model).to(device)


# --- 6. Optimizer, Scheduler, Loss ---
if args.do_train:
    from sklearn.utils.class_weight import compute_class_weight

    pos_weight = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=df_train["Label"])[1],device=device, dtype=torch.float32)
    #pos_weight = torch.tensor(compute_class_weight("balanced", classes=[0, 1], y=df_train["Label"])[1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(NotesDataset(df_train), batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(NotesDataset(df_val), batch_size=args.train_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

# --- 7. Training/Evaluation Loop ---
def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    losses, preds, labels = [], [], []
    loop = tqdm(loader, desc="Train" if train else "Eval")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.autocast(device_type="mps", enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, lbls)

        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()

        losses.append(loss.item())
        prob = torch.sigmoid(logits.detach().cpu())
        pred = (prob >= 0.5).int().numpy()
        preds.extend(pred.tolist())
        labels.extend(lbls.cpu().numpy().tolist())
        loop.set_postfix(loss=loss.item())

    return {
        "loss": np.mean(losses),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, preds),
        "labels": labels,
        "preds": preds }

# --- 8. Training Loop with Early Stopping ---
if args.do_train:
    metrics_log = []
    best_auc, patience, ctr = 0.0, 2, 0

    for epoch in range(1, args.num_train_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_train_epochs}")
        tr_metrics = run_epoch(train_loader, train=True)
        va_metrics = run_epoch(val_loader, train=False)

        print(f"Train ▶ loss {tr_metrics['loss']:.4f} | acc {tr_metrics['acc']:.4f} | auc {tr_metrics['auc']:.4f}")
        print(f" Val  ▶ loss {va_metrics['loss']:.4f} | acc {va_metrics['acc']:.4f} | auc {va_metrics['auc']:.4f}")

        metrics_log.append({**{"epoch": epoch}, **tr_metrics, **{f"val_{k}": v for k, v in va_metrics.items()}})

        if va_metrics["auc"] > best_auc:
            best_auc = va_metrics["auc"]
            ctr = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            tokenizer.save_pretrained(args.output_dir)
        else:
            ctr += 1
            if ctr >= patience:
                print("🔸 Early stopping triggered.")
                break

    # Save training log
    with open(os.path.join(args.output_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)

# --- 9. Final Test ---
if args.do_eval:
    test_loader = DataLoader(NotesDataset(df_test), batch_size=args.train_batch_size, shuffle=False)
    test_metrics = run_epoch(test_loader, train=False)
    print(f"\nTest ▶ loss {test_metrics['loss']:.4f} | acc {test_metrics['acc']:.4f} | auc {test_metrics['auc']:.4f}")

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Readmitted", "Readmitted"])
    disp.plot(cmap="Blues")
    plt.title("Test Set Confusion Matrix")
    plt.grid(False)
    plt.show()

    print("\nClassification Report:")
    print(classification_report(test_metrics["labels"], test_metrics["preds"], target_names=["Not Readmitted", "Readmitted"]))

# --- 10. Plotting ---
log_df = pd.DataFrame(metrics_log)
plt.figure(figsize=(8, 4))
plt.plot(log_df["epoch"], log_df["loss"], label="Train Loss")
plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(log_df["epoch"], log_df["val_f1"], label="Val F1")
plt.title("Validation F1 Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\nDone. Best Val AUROC: {best_auc:.4f}")
