"""Fine-tunes BERTimbau (neuralmind/bert-base-portuguese-cased) on a GPU.

- Training: balanced subsample of the train split (all positives + as many negatives).
- Model selection: F1 on the validation split after each epoch; the best epoch is kept.
- Test: scored once, with the selected checkpoint.

Writes artifacts/bert/ (weights + tokenizer), artifacts/bert_metrics.json and
artifacts/preds/bert_{val,test}.npy (positive-class probabilities, used by the ensemble).

Runs on a GPU only (Google Colab): see colab/train_bert_colab.ipynb.
Usage: python finetune_bert.py [--epochs 3] [--max-len 64]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import SEED, load_splits

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
OUT = Path("artifacts")


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4)}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model, tok, texts, max_len, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    if DEVICE.type != "cuda":
        raise SystemExit("No GPU found. Train on Google Colab (Runtime > Change runtime type > GPU), never on a local CPU.")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    train, val, test, stats = load_splits()

    pos = train[train.label == 1]
    neg = train[train.label == 0].sample(n=len(pos), random_state=SEED)
    sub = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"balanced training subsample: {len(sub)} ({len(pos)} positive)", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(DEVICE)
    enc = tok(sub["text"].tolist(), padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(sub["label"].values))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    scaler = torch.amp.GradScaler("cuda")
    best = {"val_f1": -1}
    history = []
    model.train()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        for step, (ids, att, y) in enumerate(dl):
            ids, att, y = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                done = (step + 1) / len(dl)
                eta = (time.time() - t0) / done * (1 - done) / 60
                print(f"epoch {ep} step {step}/{len(dl)} loss {loss.item():.3f} "
                      f"({(time.time() - t0) / 60:.1f} min, epoch ETA {eta:.1f} min)", flush=True)
        val_prob = predict_proba(model, tok, val["text"].tolist(), args.max_len)
        m = metrics(val["label"].values, val_prob)
        history.append({"epoch": ep, "minutes": round((time.time() - t0) / 60, 1), **{f"val_{k}": v for k, v in m.items()}})
        print(f"epoch {ep} done: val {m}", flush=True)
        if m["f1"] > best["val_f1"]:
            best = {"val_f1": m["f1"], "epoch": ep}
            model.save_pretrained(OUT / "bert")
            tok.save_pretrained(OUT / "bert")
            (OUT / "preds").mkdir(parents=True, exist_ok=True)
            np.save(OUT / "preds" / "bert_val.npy", val_prob)

    # Single test evaluation with the checkpoint selected on validation.
    model = AutoModelForSequenceClassification.from_pretrained(OUT / "bert").to(DEVICE)
    test_prob = predict_proba(model, tok, test["text"].tolist(), args.max_len)
    np.save(OUT / "preds" / "bert_test.npy", test_prob)
    result = {
        "model": BASE_MODEL, "max_len": args.max_len, "batch": args.batch, "lr": args.lr,
        "epochs_run": args.epochs, "selected_epoch": best["epoch"], "train_subsample": len(sub),
        **stats, "history": history,
        "val": metrics(val["label"].values, np.load(OUT / "preds" / "bert_val.npy")),
        "test": metrics(test["label"].values, test_prob),
        "total_minutes": round((time.time() - t_start) / 60, 1),
    }
    (OUT / "bert_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
