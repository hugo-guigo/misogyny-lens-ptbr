"""BERTimbau classifier with an occlusion explanation.

Occlusion: remove one word at a time and measure how much the probability of
"misogyny traits" drops. A large drop means the word mattered. It costs one extra
forward pass per word, all batched into a single call, which is cheap for short texts.
Unlike the SVM split, this is an estimate of importance, not an exact decomposition.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

WORD = re.compile(r"\w+", re.UNICODE)
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "bert"


class BertModel:
    def __init__(self, model_dir: Path = DEFAULT_DIR, max_len: int = 64, threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.max_len = max_len

    def proba(self, texts: list[str]) -> np.ndarray:
        with self.torch.inference_mode():
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            return self.torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].numpy()

    def explain(self, text: str, max_words: int = 40) -> dict:
        words = [(m.start(), m.end()) for m in WORD.finditer(text)][:max_words]
        variants = [text] + [text[:s] + text[e:] for s, e in words]
        probs = self.proba(variants)
        base = float(probs[0])
        return {
            "probability": base,
            "label": int(base >= 0.5),
            "words": [{"text": text[s:e], "start": s, "end": e, "importance": round(base - float(p), 4)}
                      for (s, e), p in zip(words, probs[1:])],
        }
