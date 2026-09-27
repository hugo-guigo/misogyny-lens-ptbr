"""Exports the fine-tuned BERTimbau to ONNX (fp32 and dynamic int8) and measures both.

No training: this converts the trained weights and runs inference on the 333-sentence
test set to check that int8 quantization keeps the accuracy.

Writes artifacts/onnx/{model.onnx, model_quantized.onnx} (the layout Transformers.js
expects under onnx/) and artifacts/onnx_metrics.json.
Usage: python export_onnx.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import load_splits

SRC = Path("artifacts/bert")
OUT = Path("artifacts/onnx")
MAX_LEN = 64


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4)}


def ort_proba(session, enc):
    feeds = {i.name: enc[i.name] for i in session.get_inputs()}
    logits = session.run(["logits"], feeds)[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def timed(fn, texts, tok, batch=1, rounds=50):
    """Median latency of single-sentence inference, in ms."""
    times = []
    for text in texts[:rounds]:
        t0 = time.perf_counter()
        fn(tok([text], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np"))
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()

    dummy = tok(["texto de exemplo"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    axes["logits"] = {0: "batch"}
    torch.onnx.export(model, tuple(dummy[n] for n in names), OUT / "model.onnx",
                      input_names=names, output_names=["logits"], dynamic_axes=axes,
                      opset_version=17, dynamo=False)
    quantize_dynamic(OUT / "model.onnx", OUT / "model_quantized.onnx", weight_type=QType)

    _, _, test, _ = load_splits(verbose=False)
    texts, y = test["text"].tolist(), test["label"].values
    enc_np = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
    enc_np = {k: v.astype(np.int64) for k, v in enc_np.items()}

    with torch.inference_mode():
        enc_pt = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        torch_prob = torch.softmax(model(**enc_pt).logits, -1)[:, 1].numpy()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sessions = {name: ort.InferenceSession(str(OUT / f), opts, providers=["CPUExecutionProvider"])
                for name, f in (("onnx_fp32", "model.onnx"), ("onnx_int8", "model_quantized.onnx"))}
    probs = {"torch_fp32": torch_prob, **{k: ort_proba(s, enc_np) for k, s in sessions.items()}}

    as_np = lambda e: {k: v.astype(np.int64) for k, v in e.items()}
    latency = {
        "torch_fp32": timed(lambda e: model(**{k: torch.from_numpy(v) for k, v in e.items()}), texts, tok),
        **{k: timed(lambda e, s=s: ort_proba(s, as_np(e)), texts, tok) for k, s in sessions.items()},
    }
    size_mb = {"onnx_fp32": round((OUT / "model.onnx").stat().st_size / 1e6, 1),
               "onnx_int8": round((OUT / "model_quantized.onnx").stat().st_size / 1e6, 1),
               "torch_fp32": round((SRC / "model.safetensors").stat().st_size / 1e6, 1)}
    result = {
        "test": {k: scores(y, p) for k, p in probs.items()},
        "max_abs_prob_diff_vs_torch": {k: round(float(np.abs(p - torch_prob).max()), 5) for k, p in probs.items()},
        "label_agreement_vs_torch": {k: round(float(((p >= 0.5) == (torch_prob >= 0.5)).mean()), 4) for k, p in probs.items()},
        "size_mb": size_mb,
        "median_latency_ms_single_sentence_2_threads": latency,
    }
    Path("artifacts/onnx_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


QType = QuantType.QInt8

if __name__ == "__main__":
    main()
