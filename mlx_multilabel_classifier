from mlx.utils import tree_flatten 
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from transformers import AutoTokenizer
from model import load_model
from tqdm.auto import tqdm, trange
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_recall_fscore_support

def top_k_labels(y_true, k=8):
    label_counts = np.sum(y_true, axis=0)
    return np.argsort(-label_counts)[:k]

TOP_K = 8

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(base_model.pooler.weight.shape[0], num_labels)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(pooled_output)

def binary_cross_entropy(logits, targets, epsilon=1e-8):
    probs = mx.sigmoid(logits)
    return -mx.mean(targets * mx.log(probs + epsilon) + (1 - targets) * mx.log(1 - probs + epsilon))

def evaluate(model, inputs, masks, labels, batch_size=32):
    total_loss = 0.0
    all_preds, all_targets = [], []

    for i in range(0, len(inputs), batch_size):
        ids = inputs[i:i+batch_size]
        msk = masks[i:i+batch_size]
        targs = labels[i:i+batch_size]

        logits = model(ids, attention_mask=msk)
        loss = binary_cross_entropy(logits, targs)

        total_loss += loss.item()
        probs = mx.sigmoid(logits)
        preds = (probs > 0.5).astype(mx.int32)

        all_preds.append(np.asarray(preds))
        all_targets.append(np.asarray(targs))

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / (len(inputs) // batch_size)

    cm = multilabel_confusion_matrix(all_targets, all_preds)
    P, R, F1s, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    S = all_targets.sum(axis=0)

    return avg_loss, micro_f1, macro_f1, cm, P, R, F1s, S

input_ids = mx.array(np.load("processed/input_ids.npy"))
attention_mask = mx.array(np.load("processed/attention_mask.npy"))
labels = mx.array(np.load("processed/labels.npy"))

indices = list(range(len(input_ids)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_inputs = input_ids[train_idx]
train_masks = attention_mask[train_idx]
train_labels = labels[train_idx]

val_inputs = input_ids[val_idx]
val_masks = attention_mask[val_idx]
val_labels = labels[val_idx]

num_labels = labels.shape[1]
base_model, tokenizer = load_model(
    "thomas-sounack/BioClinical-ModernBERT-base",
    "BioClinical-ModernBERT-mlx/weights.npz"
)
model = BertForMultiLabelClassification(base_model, num_labels)

optimizer = Adam(learning_rate=3e-5)
optimizer_state = optimizer.state

epochs = 10
batch_size = 8
best_val_f1 = 0.0
save_every = False

for epoch in trange(epochs, desc="Epochs"):
    total_loss = 0.0
    batch_iterator = tqdm(
        range(0, len(train_inputs), batch_size),
        desc=f"Train {epoch+1}/{epochs}", leave=False
    )

    for i in batch_iterator:
        ids = train_inputs[i:i+batch_size]
        msk = train_masks[i:i+batch_size]
        lbl = train_labels[i:i+batch_size]

        def loss_fn(mdl, ids, msk, lbl):
            logits = mdl(ids, attention_mask=msk)
            return binary_cross_entropy(logits, lbl)

        loss, grads = mx.value_and_grad(loss_fn)(model, ids, msk, lbl)
        optimizer.update(model, grads)
        total_loss += loss.item()

        batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

    v_loss, microF, macroF, cm, P, R, F1s, S = evaluate(
        model, val_inputs, val_masks, val_labels, batch_size=batch_size
    )
    avg_train = total_loss / max(1, len(train_inputs)//batch_size)
    tqdm.write(f"Epoch {epoch+1}/{epochs} ✅ train={avg_train:.4f} val={v_loss:.4f}  microF1={microF:.4f} macroF1={macroF:.4f}")

    for idx in top_k_labels(np.asarray(val_labels), TOP_K):
        tn, fp, fn, tp = cm[idx].ravel()
        tqdm.write(f" label {idx:3d}: P={P[idx]:.2f} R={R[idx]:.2f} F1={F1s[idx]:.2f} n={S[idx]} | TP={tp} FP={fp} FN={fn} TN={tn}")

    if microF > best_val_f1:
        best_val_f1 = microF
        params = {k: v for k, v in model.parameters().items() if isinstance(v, mx.array)}
        mx.savez("bert_multilabel_best.npz", **params)
        tqdm.write("  🔥 New best model saved!")

    if save_every:
        model.save_weights(f"bert_multilabel_epoch{epoch+1}.npz")
