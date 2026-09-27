"""Scores the LLM judge on the validation sample and the 333-sentence test set.

Results are cached in artifacts/preds/llm_{split}.jsonl, so reruns only call the API
for texts not scored yet (no double spending). Requires ANTHROPIC_API_KEY.

Usage: python eval_llm.py [--val-sample 400] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.llm_judge import MODEL, get_client, judge
from data import SEED, load_splits

PREDS = Path("artifacts/preds")


def score_split(name, texts, client, workers):
    path = PREDS / f"llm_{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["i"]] = row
    todo = [i for i in range(len(texts)) if i not in done]
    print(f"{name}: {len(done)} cached, {len(todo)} to score with {MODEL}")
    with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge, texts[i], client): i for i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            j = fut.result()
            row = {"i": i, "label": j.label, "confidence": j.confidence, "category": j.category,
                   "refused": j.refused, "model": j.model, "explanation": j.explanation}
            done[i] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")
    return [done[i] for i in range(len(texts))]


def report(y, rows):
    pred = np.array([r["label"] for r in rows])
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4),
            "refusals": sum(r["refused"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-sample", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    PREDS.mkdir(parents=True, exist_ok=True)
    _, val, test, _ = load_splits()
    val_idx = np.random.default_rng(SEED).choice(len(val), size=min(args.val_sample, len(val)), replace=False)
    np.save(PREDS / "llm_val_idx.npy", val_idx)
    client = get_client()
    val_rows = score_split("val", val["text"].iloc[val_idx].tolist(), client, args.workers)
    test_rows = score_split("test", test["text"].tolist(), client, args.workers)
    result = {"model": MODEL,
              "val_sample": report(val["label"].values[val_idx], val_rows),
              "test": report(test["label"].values, test_rows)}
    Path("artifacts/llm_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
