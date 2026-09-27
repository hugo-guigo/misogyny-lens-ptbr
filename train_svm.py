"""Trains the linear SVM (TF-IDF of spaCy lemmas + 8 lexicon features) with scikit-learn.

Fits on the train split, reports validation and test, and writes:
- artifacts/svm.joblib            vectorizer, scaler, LinearSVC and lexicon
- artifacts/svm_metrics.json      metrics, bootstrap CI and the lexicon ablation
- artifacts/preds/svm_{val,test}.npy  decision scores, used by the ensemble

Usage: python train_svm.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app import lexicon as lex
from app.model import LinearTextModel
from app.text import SPACY_MODEL, lemmas_batch
from data import COMMIT, SEED, load_lexicon, load_splits

OUT = Path("artifacts")


def scores(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4)}


def bootstrap_f1(y, pred_a, pred_b, n=2000, seed=SEED):
    """Paired bootstrap: 95% CI of F1 for both models and of the delta (b - a)."""
    rng = np.random.default_rng(seed)
    fa, fb, delta, wins = [], [], [], 0
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        a, b = f1_score(y[idx], pred_a[idx]), f1_score(y[idx], pred_b[idx])
        fa.append(a); fb.append(b); delta.append(b - a); wins += b > a
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"f1_ci_a": ci(fa), "f1_ci_b": ci(fb), "delta_mean": round(float(np.mean(delta)), 4),
            "delta_ci": ci(delta), "p_one_sided": round(1 - wins / len(delta), 4)}


def main():
    t0 = time.time()
    train, val, test, stats = load_splits()
    lexicon = load_lexicon()
    print(f"lemmatizing with spaCy {SPACY_MODEL} ...")
    L = {name: lemmas_batch(df["text"].tolist()) for name, df in (("train", train), ("val", val), ("test", test))}
    y = {"train": train["label"].values, "val": val["label"].values, "test": test["label"].values}

    vec = TfidfVectorizer(min_df=5, max_df=0.95, max_features=10_000, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-záéíóúâêôãõçàü\-]{3,}\b")
    tfidf = {"train": vec.fit_transform(" ".join(l) for l in L["train"])}
    for s in ("val", "test"):
        tfidf[s] = vec.transform(" ".join(l) for l in L[s])
    lexf = {s: np.vstack([lex.features(l, lexicon) for l in L[s]]) for s in L}
    scaler = StandardScaler().fit(lexf["train"])
    X = {s: hstack([tfidf[s], csr_matrix(scaler.transform(lexf[s]))]).tocsr() for s in L}

    svm = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED).fit(X["train"], y["train"])
    base = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED).fit(tfidf["train"], y["train"])

    bundle = {"vectorizer": vec, "scaler": scaler, "svm": svm, "lexicon": lexicon,
              "meta": {"spacy_model": SPACY_MODEL, "data_commit": COMMIT}}
    OUT.mkdir(exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    joblib.dump(bundle, OUT / "svm.joblib", compress=3)

    # The explanation path must reproduce sklearn's decision function.
    model = LinearTextModel(bundle)
    dec = {s: svm.decision_function(X[s]) for s in ("val", "test")}
    assert max(abs(model.decision_from_lemmas(l) - d) for l, d in zip(L["test"], dec["test"])) < 1e-9
    for s in ("val", "test"):
        np.save(OUT / "preds" / f"svm_{s}.npy", dec[s])

    metrics = {
        "model": "LinearSVC(class_weight=balanced) on TF-IDF of spaCy lemmas + 8 lexicon features",
        "spacy_model": SPACY_MODEL, "data_commit": COMMIT, **stats,
        "vocabulary_size": len(vec.vocabulary_),
        "val": scores(y["val"], (dec["val"] > 0).astype(int)),
        "test": scores(y["test"], (dec["test"] > 0).astype(int)),
        "test_tfidf_only": scores(y["test"], base.predict(tfidf["test"])),
        "bootstrap_lexicon_vs_tfidf_only": bootstrap_f1(y["test"], base.predict(tfidf["test"]), (dec["test"] > 0).astype(int)),
        "train_seconds": round(time.time() - t0, 1),
    }
    (OUT / "svm_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
