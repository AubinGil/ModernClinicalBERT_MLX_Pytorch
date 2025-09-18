import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from mlx.utils import tree_flatten
import mlx.core as mx
from mlx.optimizers import Adam

from transformers import AutoTokenizer
from model import load_model
from HierarchicalBERT import HierarchicalBERTa

EPOCHS = 2
BATCH_SIZE = 1
MAX_LEN = 384
STRIDE = 128
LEARNING_RATE = 1e-5

LOG_PATH = "training_log.csv"
BATCH_LOG_PATH = "batch_loss_log.csv"

log_columns = ["epoch", "train_loss", "val_loss", "val_acc", "val_auc", "val_f1", "tn", "fp", "fn", "tp"]
log_rows = []
batch_logs = []
global_step = 0

console = Console()

progress_bar = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
    transient=True
)

def chunk_long_text(tokenizer, text, max_len=512, stride=128):
    toks = tokenizer(text, return_tensors="np", truncation=False, add_special_tokens=True)
    ids = toks["input_ids"][0]
    n = len(ids)

    windows, masks = [], []
    start = 0
    while start < n:
        end = min(start + max_len, n)
        window = ids[start:end]
        pad = max_len - len(window)
        if pad:
            window = np.pad(window, (0, pad), constant_values=tokenizer.pad_token_id)
        windows.append(window)
        masks.append((window != tokenizer.pad_token_id).astype(np.int32))
        if end == n:
            break
        start += max_len - stride
    return windows, masks

def prepare_batch(docs, labels, tokenizer, max_len=512, stride=128):
    chunks, masks, c2d = [], [], []
    for i, text in enumerate(docs):
        ch, msk = chunk_long_text(tokenizer, text, max_len=max_len, stride=stride)
        chunks.extend(ch)
        masks.extend(msk)
        c2d.extend([i] * len(ch))

    input_ids = mx.array(np.array(chunks))
    attention_mask = mx.array(np.array(masks))
    chunk2doc = mx.array(np.array(c2d))
    batch_labels = mx.array(labels)

    return input_ids, attention_mask, chunk2doc, batch_labels

def binary_cross_entropy(logits, targets, epsilon=1e-8):
    probs = mx.sigmoid(logits)
    return -mx.mean(targets * mx.log(probs + epsilon) + (1 - targets) * mx.log(1 - probs + epsilon))

def evaluate(model, docs, labels, tokenizer, batch_size):
    all_preds, all_targets, all_probs = [], [], []
    total_loss = 0.0

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_labels = mx.array(labels[i:i+batch_size])
        token_chunks, attn_chunks, c2d_ids, batch_lbls = prepare_batch(batch_docs, batch_labels, tokenizer)

        logits = model.forward_chunks(token_chunks, attn_chunks, c2d_ids, len(batch_docs))
        loss = binary_cross_entropy(logits, batch_lbls)
        total_loss += loss.item()

        probs = np.array(mx.sigmoid(logits))
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(np.array(batch_lbls).tolist())

    acc = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    return total_loss / max(1, len(docs) // batch_size), acc, auc, f1, cm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="hierarchical_binary.npz")
    args = parser.parse_args()

    backbone, tokenizer = load_model(args.bert_model, args.weights_path)
    hidden = backbone.pooler.weight.shape[0]

    model = HierarchicalBERTa(
        base_bert=backbone,
        hidden_size=hidden,
        num_labels=1,
        doc_layers=2,
        doc_heads=4,
        cls_pool="first"
    )
    optimizer = Adam(learning_rate=LEARNING_RATE)
    optimizer_state = optimizer.state

    df = pd.read_csv(os.path.join(args.data_dir, "test.csv")).dropna(subset=["TEXT", "Label"])
    texts = df["TEXT"].tolist()
    labels = df["Label"].astype(np.float32).tolist()

    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    train_docs = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_docs = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        random.shuffle(train_idx)
        task_id = progress_bar.add_task(f"Epoch {epoch+1}/{EPOCHS}", total=len(train_docs))

        with progress_bar:
            for i in range(0, len(train_docs), BATCH_SIZE):
                batch_docs = train_docs[i:i+BATCH_SIZE]
                batch_lbls = train_labels[i:i+BATCH_SIZE]
                token_chunks, attn_chunks, c2d_ids, batch_lbls = prepare_batch(batch_docs, batch_lbls, tokenizer)

                def loss_fn(mdl):
                    logits = mdl.forward_chunks(token_chunks, attn_chunks, c2d_ids, len(batch_docs))
                    return binary_cross_entropy(logits, batch_lbls)

                loss, grads = mx.value_and_grad(loss_fn)(model)
                optimizer.update(model, grads)
                loss_value = loss.item()
                total_loss += loss_value

                batch_logs.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch": i // BATCH_SIZE + 1,
                    "loss": loss_value,
                    "learning_rate": LEARNING_RATE
                })
                global_step += 1

                progress_bar.update(task_id, advance=BATCH_SIZE, description=f"Epoch {epoch+1} | Loss: {loss_value:.4f}")

        val_loss, val_acc, val_auc, val_f1, val_cm = evaluate(model, val_docs, val_labels, tokenizer, BATCH_SIZE)
        train_loss = total_loss / max(1, len(train_docs) // BATCH_SIZE)

        console.print(f"\n[bold green]Epoch {epoch+1}/{EPOCHS}[/bold green]")
        console.print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        console.print(f"Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f} | F1 Score: {val_f1:.4f}")
        console.print(f"Confusion Matrix: {val_cm.tolist()}")

        log_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "tn": val_cm[0][0],
            "fp": val_cm[0][1],
            "fn": val_cm[1][0],
            "tp": val_cm[1][1]
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            params = dict(tree_flatten(model.parameters()))
            mx.savez(args.output, **params)
            console.print("[bold yellow]\U0001f525 Best model saved![/bold yellow]")

    pd.DataFrame(log_rows, columns=log_columns).to_csv(LOG_PATH, index=False)
    pd.DataFrame(batch_logs).to_csv(BATCH_LOG_PATH, index=False)
    console.print(f"[blue]Training log saved to {LOG_PATH}[/blue]")
    console.print(f"[blue]Batch loss log saved to {BATCH_LOG_PATH}[/blue]")
