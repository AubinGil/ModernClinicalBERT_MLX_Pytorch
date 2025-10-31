#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, argparse, time, numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

# LoRA
from peft import LoraConfig, get_peft_model


# -------------------------
# Utilities
# -------------------------
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -------------------------
# Dataset with overflow chunking
# -------------------------
class OverflowChunkedTextDS(Dataset):
    """
    Tokenizes a list of texts with truncation + overflow (sliding window).
    Keeps a mapping from each chunk to the original sample index so we can
    aggregate validation metrics per sample.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        stride: int = 256,
        is_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.is_train = is_train

        # HuggingFace fast tokenizer overflow
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=False,
        )

        # overflow_to_sample_mapping maps chunk -> original sample index
        self.overflow_to_sample = enc.pop("overflow_to_sample_mapping")
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]

        # map chunk label from its original sample
        self.labels = [int(labels[i]) for i in self.overflow_to_sample]

        # also keep the original index to aggregate per-sample at eval
        self.sample_ids = self.overflow_to_sample

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "sample_id": torch.tensor(self.sample_ids[idx], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    # dynamic pad
    keys = ["input_ids", "attention_mask"]
    max_len = max(x["input_ids"].shape[0] for x in batch)
    def pad(x, pad_val):
        out = torch.full((len(batch), max_len), pad_val, dtype=x[0].dtype)
        for i, b in enumerate(batch):
            out[i, : b[x[1]].shape[0]] = b[x[1]]
        return out

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.stack([b["labels"] for b in batch])
    sample_ids = torch.stack([b["sample_id"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_id": sample_ids,
    }


# -------------------------
# Train / Eval
# -------------------------
def evaluate(model, dataloader, device) -> Tuple[float, float]:
    model.eval()
    losses, probs_all, labels_all, sample_ids_all = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
            sample_ids = dataloader.dataset.sample_ids
            out = model(**batch)
            losses.append(out.loss.detach().float().item())
            p = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs_all.extend(p)
            labels_all.extend(batch["labels"].detach().cpu().numpy())
            # for per-sample aggregation we need the sample ids from the actual batch
            # we recompute below using the collated tensor the dataloader provided:
            # dataloader gives us batch["sample_id"] too:
            # (we removed above for device move, so grab it from original batch dict)
    # The easy way: iterate again to gather sample_ids (we kept them on CPU)
    for batch in dataloader:
        sample_ids_all.extend(batch["sample_id"].numpy().tolist())

    # Aggregate chunk probs to per-sample by mean
    df = pd.DataFrame(
        {"sid": sample_ids_all, "prob": probs_all, "y": labels_all}
    )
    agg = df.groupby("sid").agg(prob=("prob", "mean"), y=("y", "first")).reset_index()
    try:
        auroc = roc_auc_score(agg["y"].values, agg["prob"].values)
    except Exception:
        auroc = float("nan")
    val_loss = float(np.mean(losses)) if len(losses) else float("inf")
    model.train()
    return val_loss, auroc


def train(args):
    seed_everything(42)
    device = pick_device()
    print(f"[device] {device}")

    # Load data
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    text_col = args.text_col
    label_col = args.label_col
    assert text_col in train_df.columns and label_col in train_df.columns, "Missing text/label columns."

    # Tokenizer / model
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2)

    # LoRA
    if args.use_lora:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    # Datasets
    train_ds = OverflowChunkedTextDS(
        texts=train_df[text_col].astype(str).tolist(),
        labels=train_df[label_col].astype(int).tolist(),
        tokenizer=tok,
        max_length=args.max_length,
        stride=args.stride,
        is_train=True,
    )
    val_ds = OverflowChunkedTextDS(
        texts=val_df[text_col].astype(str).tolist(),
        labels=val_df[label_col].astype(int).tolist(),
        tokenizer=tok,
        max_length=args.max_length,
        stride=args.stride,
        is_train=False,
    )

    # Dataloaders
    collate = lambda b: collate_fn(b, pad_token_id=tok.pad_token_id or 0)
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # Optimizer / Scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, math.ceil(len(train_dl) * args.num_epochs))
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # MPS detail (safe; ignored if not MPS/CUDA)
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    SAVE_DIR = args.output_dir or "outputs_lora"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Train loop with early stopping
    model.train()
    best_metric = None
    best_step = -1
    patience_left = args.early_stop_patience
    global_step = 0

    def maybe_improved(curr, best, mode):
        if best is None:
            return True
        if mode == "auroc":
            return curr > best + args.early_stop_min_delta
        else:
            return curr < best - args.early_stop_min_delta

    for epoch in range(max(1, args.num_epochs)):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items() if k != "sample_id"}
            out = model(**batch)
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"[step {global_step}] loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.6f}")

            do_eval = (global_step % args.eval_every == 0) or (args.max_steps > 0 and global_step >= args.max_steps)
            if do_eval:
                val_loss, val_auroc = evaluate(model, val_dl, device)
                metric_val = val_auroc if args.early_stop_metric == "auroc" else val_loss
                print(f"[eval @ {global_step}] val_loss={val_loss:.4f} val_auroc={val_auroc:.4f}")

                if maybe_improved(metric_val, best_metric, args.early_stop_metric):
                    best_metric = metric_val
                    best_step = global_step
                    patience_left = args.early_stop_patience
                    model.save_pretrained(os.path.join(SAVE_DIR, "best"))
                    tok.save_pretrained(os.path.join(SAVE_DIR, "best"))
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"[early stop] best {args.early_stop_metric}={best_metric:.4f} @ step {best_step}")
                        return

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[done] reached max_steps={args.max_steps}")
                return

    print(f"[done] training finished. Best step {best_step} (metric={best_metric:.4f}). Weights in {SAVE_DIR}/best")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser("Train GatorTron-base with LoRA (MPS-friendly) on cleaned CSVs")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--text_col", default="text")     # use 'text' from your lora-shim
    ap.add_argument("--label_col", default="label")   # use 'label' from your lora-shim
    ap.add_argument("--model_id", default="UFNLP/gatortron-base")
    ap.add_argument("--output_dir", default="outputs_lora")

    # Tokenization / chunking
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)

    # Training
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="if >0, overrides epochs")
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging / eval / early stop
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--early_stop_metric", choices=["auroc", "loss"], default="auroc")
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--early_stop_min_delta", type=float, default=5e-4)
    return ap.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

