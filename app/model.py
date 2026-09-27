"""Inference with a linear SVM stored as plain JSON (no pickle, no scikit-learn).

decision(x) = w_tfidf . tfidf(x) + w_lex . scale(lexicon(x)) + b

Because the model is linear, the decision splits exactly into one term per lemma plus
one term for the lexicon features plus the bias. The explanation is that split, not an
approximation.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app import lexicon as lex
from app.text import Token, analyze

# Same token pattern the TF-IDF vectorizer used in training.
TOKEN_PATTERN = re.compile(r"(?u)\b[a-záéíóúâêôãõçàü\-]{3,}\b")
DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "model.json"


@dataclass
class Explanation:
    decision: float
    label: int
    bias: float
    lexicon_contribution: float
    lexicon_features: dict[str, float]
    tokens: list[dict]


class LinearTextModel:
    def __init__(self, artifact: dict):
        self.vocab: dict[str, int] = artifact["vocabulary"]
        self.idf = np.asarray(artifact["idf"])
        self.w_tfidf = np.asarray(artifact["coef_tfidf"])
        self.w_lex = np.asarray(artifact["coef_lexicon"])
        self.scaler_mean = np.asarray(artifact["scaler_mean"])
        self.scaler_scale = np.asarray(artifact["scaler_scale"])
        self.bias = float(artifact["intercept"])
        self.lexicon: dict[str, dict] = artifact["lexicon"]
        self.meta = artifact.get("meta", {})

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT) -> "LinearTextModel":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def tfidf(self, lemmas: list[str]) -> dict[int, float]:
        """Sparse TF-IDF: sublinear tf, idf weighting, L2 normalization."""
        terms = TOKEN_PATTERN.findall(" ".join(lemmas))
        counts = Counter(self.vocab[t] for t in terms if t in self.vocab)
        vec = {j: (1.0 + math.log(c)) * self.idf[j] for j, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {j: v / norm for j, v in vec.items()} if norm > 0 else {}

    def lexicon_scaled(self, lemmas: list[str]) -> np.ndarray:
        return (lex.features(lemmas, self.lexicon) - self.scaler_mean) / self.scaler_scale

    def decision_from_lemmas(self, lemmas: list[str]) -> float:
        x = self.tfidf(lemmas)
        return (sum(self.w_tfidf[j] * v for j, v in x.items())
                + float(self.w_lex @ self.lexicon_scaled(lemmas)) + self.bias)

    def explain(self, text: str, tokens: list[Token] | None = None) -> Explanation:
        tokens = analyze(text) if tokens is None else tokens
        lemmas = [t.lemma for t in tokens if t.lemma]
        x = self.tfidf(lemmas)

        # Each vocabulary term's contribution w_j * x_j is split evenly across its
        # occurrences, then summed back per lemma (a lemma can hold several terms).
        term_counts = Counter(TOKEN_PATTERN.findall(" ".join(lemmas)))
        per_term = {t: self.w_tfidf[self.vocab[t]] * x[self.vocab[t]] / n
                    for t, n in term_counts.items() if t in self.vocab}
        occurrences = Counter(lemmas)
        per_lemma = {l: sum(per_term.get(t, 0.0) for t in TOKEN_PATTERN.findall(l)) for l in occurrences}

        raw_lex = lex.features(lemmas, self.lexicon)
        lex_contrib = float(self.w_lex @ ((raw_lex - self.scaler_mean) / self.scaler_scale))
        decision = sum(per_lemma[l] * n for l, n in occurrences.items()) + lex_contrib + self.bias

        out_tokens = []
        for t in tokens:
            info = self.lexicon.get(t.lemma) if t.lemma else None
            out_tokens.append({
                "text": t.text, "start": t.start, "end": t.end, "lemma": t.lemma,
                "contribution": per_lemma.get(t.lemma, 0.0) if t.lemma else 0.0,
                "in_vocabulary": bool(t.lemma and any(p in self.vocab for p in TOKEN_PATTERN.findall(t.lemma))),
                "lexicon": info["polarity"] if info else None,
            })
        return Explanation(
            decision=decision, label=int(decision > 0), bias=self.bias,
            lexicon_contribution=lex_contrib,
            lexicon_features=dict(zip(lex.FEATURE_NAMES, raw_lex.round(4).tolist())),
            tokens=out_tokens,
        )
