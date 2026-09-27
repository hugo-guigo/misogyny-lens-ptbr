"""Linear SVM (scikit-learn, saved with joblib) with an exact per-word explanation.

decision(x) = w_tfidf . tfidf(x) + w_lex . scale(lexicon(x)) + b

Because the model is linear, the decision splits exactly into one term per lemma, one
term for the lexicon features and the bias.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from app import lexicon as lex
from app.text import Token, analyze

DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "svm.joblib"


@dataclass
class Explanation:
    decision: float
    label: int
    bias: float
    lexicon_contribution: float
    lexicon_features: dict[str, float]
    tokens: list[dict]


class LinearTextModel:
    """Wraps the fitted TfidfVectorizer, StandardScaler and LinearSVC."""

    def __init__(self, bundle: dict):
        self.vectorizer = bundle["vectorizer"]
        self.scaler = bundle["scaler"]
        self.svm = bundle["svm"]
        self.lexicon: dict[str, dict] = bundle["lexicon"]
        self.meta = bundle.get("meta", {})
        coef = self.svm.coef_.ravel()
        self.n_tfidf = len(self.vectorizer.vocabulary_)
        self.w_tfidf, self.w_lex = coef[:self.n_tfidf], coef[self.n_tfidf:]
        self.bias = float(self.svm.intercept_[0])
        self.vocab = self.vectorizer.vocabulary_
        self.token_pattern = re.compile(self.vectorizer.token_pattern)

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT) -> "LinearTextModel":
        # joblib uses pickle: only load artifacts produced by train_svm.py in this repo.
        return cls(joblib.load(path))

    def features(self, lemma_docs: list[list[str]]):
        tfidf = self.vectorizer.transform(" ".join(l) for l in lemma_docs)
        lexf = self.scaler.transform(np.vstack([lex.features(l, self.lexicon) for l in lemma_docs]))
        return hstack([tfidf, csr_matrix(lexf)]).tocsr()

    def decision_from_lemmas(self, lemmas: list[str]) -> float:
        return float(self.svm.decision_function(self.features([lemmas]))[0])

    def explain(self, text: str, tokens: list[Token] | None = None) -> Explanation:
        tokens = analyze(text) if tokens is None else tokens
        lemmas = [t.lemma for t in tokens if t.lemma]
        x = self.vectorizer.transform([" ".join(lemmas)])
        xv = dict(zip(x.indices, x.data))

        # Each vocabulary term's contribution w_j * x_j is split evenly across its
        # occurrences, then summed back per lemma (a lemma can hold several terms).
        term_counts = Counter(self.token_pattern.findall(" ".join(lemmas)))
        per_term = {t: self.w_tfidf[self.vocab[t]] * xv.get(self.vocab[t], 0.0) / n
                    for t, n in term_counts.items() if t in self.vocab}
        occurrences = Counter(lemmas)
        per_lemma = {l: sum(per_term.get(t, 0.0) for t in self.token_pattern.findall(l)) for l in occurrences}

        raw_lex = lex.features(lemmas, self.lexicon)
        lex_contrib = float(self.w_lex @ self.scaler.transform(raw_lex[None, :])[0])
        decision = sum(per_lemma[l] * n for l, n in occurrences.items()) + lex_contrib + self.bias

        out_tokens = []
        for t in tokens:
            info = self.lexicon.get(t.lemma) if t.lemma else None
            out_tokens.append({
                "text": t.text, "start": t.start, "end": t.end, "lemma": t.lemma,
                "contribution": float(per_lemma.get(t.lemma, 0.0)) if t.lemma else 0.0,
                "in_vocabulary": bool(t.lemma and any(p in self.vocab for p in self.token_pattern.findall(t.lemma))),
                "lexicon": info["polarity"] if info else None,
            })
        return Explanation(
            decision=float(decision), label=int(decision > 0), bias=self.bias,
            lexicon_contribution=lex_contrib,
            lexicon_features=dict(zip(lex.FEATURE_NAMES, raw_lex.round(4).tolist())),
            tokens=out_tokens,
        )
