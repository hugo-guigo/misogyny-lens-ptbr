"""Trains the linear SVM and exports it as artifacts/model.json + artifacts/metrics.json.

Data comes from the research repo (hugo-guigo/lexico-misoginia-ptbr), pinned to one commit:
- unified training corpus (HateBR, ToLD-BR, Portuguese Hate Speech)
- 333-sentence labeled test set (ToLD-BR)
- leak-free misogyny lexicon V3

Leak control: the test sentences also exist in the unified corpus, so they (and their
duplicates) are removed from training before fitting. Without this step F1 is inflated
by memorization (about 0.87 in the research run).

Usage: python train.py
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app import lexicon as lex
from app.model import LinearTextModel
from app.text import REGEX_NON_ALPHA, SPACY_MODEL, clean, lemmas_batch

COMMIT = "3cd80edb467bf6282a23880f3f62b482cd67113b"
BASE = f"https://raw.githubusercontent.com/hugo-guigo/lexico-misoginia-ptbr/{COMMIT}"
FILES = {
    "train": "corpus_unificado_final.csv",
    "test": "outputs/corpus_teste_frases.csv",
    "lexicon": "outputs/lexico_misoginia_v3_semente_pmi_leakfree.csv",
}
DATA = Path("data")
ARTIFACTS = Path("artifacts")
SEED = 42


def fetch(key: str) -> Path:
    path = DATA / Path(FILES[key]).name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print(f"downloading {FILES[key]}")
        urllib.request.urlretrieve(f"{BASE}/{FILES[key]}", path)
    return path


def normalized(text: str) -> str:
    return " ".join(clean(text).split())


def load_data():
    train = pd.read_csv(fetch("train"))
    train = train[~train["text"].astype(str).str.fullmatch(r"\s*\d+\s*", na=False)].copy()
    train["text"] = train["text"].astype(str)
    train = train[train["text"].str.replace(REGEX_NON_ALPHA, "", regex=True).str.len() >= 3]
    train = train.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    n_dedup = len(train)

    test = pd.read_csv(fetch("test")).dropna(subset=["text", "label"])
    test = test[test["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    test["label"] = test["label"].astype(int)

    # Leak control: drop every training row whose normalized text is a test sentence.
    test_keys = set(test["text"].map(normalized))
    leaked = train["text"].map(normalized).isin(test_keys)
    train = train[~leaked].reset_index(drop=True)
    print(f"train: {n_dedup} after dedup, {int(leaked.sum())} leaked rows removed, {len(train)} used")
    print(f"test:  {len(test)} labeled ({int(test['label'].sum())} positive)")
    return train, test, n_dedup, int(leaked.sum())


def load_lexicon() -> dict[str, dict]:
    df = pd.read_csv(fetch("lexicon"))
    df["gender_assoc"] = df.get("gender_assoc", 0.0)
    return {r.term: {"polarity": r.polarity, "score": float(r.score_norm),
                     "gender": float(r.gender_assoc) if pd.notna(r.gender_assoc) else 0.0}
            for r in df.itertuples()}


def bootstrap(y, pred_a, pred_b, n=2000, seed=SEED):
    """Paired bootstrap on the test set: 95% CI of F1 for each model and of the delta."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    fa, fb, delta, wins = [], [], [], 0
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        a = f1_score(y[idx], pred_a[idx], zero_division=0)
        b = f1_score(y[idx], pred_b[idx], zero_division=0)
        fa.append(a); fb.append(b); delta.append(b - a); wins += b > a
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"f1_ci_baseline": ci(fa), "f1_ci_lexicon": ci(fb), "delta_mean": round(float(np.mean(delta)), 4),
            "delta_ci": ci(delta), "p_one_sided": round(1 - wins / len(delta), 4), "resamples": len(delta)}


def scores(y, pred):
    return {"f1": round(f1_score(y, pred), 4), "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4)}


def main():
    t0 = time.time()
    train, test, n_dedup, n_leaked = load_data()
    lexicon = load_lexicon()

    print(f"lemmatizing with spaCy {SPACY_MODEL} ...")
    train_lemmas = lemmas_batch(train["text"].tolist())
    test_lemmas = lemmas_batch(test["text"].tolist())

    vec = TfidfVectorizer(min_df=5, max_df=0.95, max_features=10_000, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-záéíóúâêôãõçàü\-]{3,}\b")
    X_tr_tfidf = vec.fit_transform(" ".join(l) for l in train_lemmas)
    X_te_tfidf = vec.transform(" ".join(l) for l in test_lemmas)
    lex_tr = np.vstack([lex.features(l, lexicon) for l in train_lemmas])
    lex_te = np.vstack([lex.features(l, lexicon) for l in test_lemmas])
    scaler = StandardScaler().fit(lex_tr)
    X_tr = hstack([X_tr_tfidf, csr_matrix(scaler.transform(lex_tr))]).tocsr()
    X_te = hstack([X_te_tfidf, csr_matrix(scaler.transform(lex_te))]).tocsr()
    y_tr, y_te = train["label"].values, test["label"].values

    baseline = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED).fit(X_tr_tfidf, y_tr)
    model = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED).fit(X_tr, y_tr)
    pred_base, pred_lex = baseline.predict(X_te_tfidf), model.predict(X_te)

    n_tfidf = X_tr_tfidf.shape[1]
    coef = model.coef_.ravel()
    vocab = {term: int(i) for term, i in vec.vocabulary_.items()}
    artifact = {
        "vocabulary": vocab,
        "idf": vec.idf_.tolist(),
        "coef_tfidf": coef[:n_tfidf].tolist(),
        "coef_lexicon": coef[n_tfidf:].tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "intercept": float(model.intercept_[0]),
        "lexicon": lexicon,
        "meta": {"spacy_model": SPACY_MODEL, "data_commit": COMMIT},
    }

    # The JSON model must reproduce scikit-learn exactly.
    exported = LinearTextModel(artifact)
    ours = np.array([exported.decision_from_lemmas(l) for l in test_lemmas])
    max_diff = float(np.abs(ours - model.decision_function(X_te)).max())
    assert max_diff < 1e-8, f"exported model diverges from sklearn: {max_diff}"

    metrics = {
        "model": "LinearSVC (class_weight=balanced) on TF-IDF of spaCy lemmas + 8 lexicon features",
        "spacy_model": SPACY_MODEL,
        "data_commit": COMMIT,
        "train_docs_after_dedup": n_dedup,
        "leaked_rows_removed": n_leaked,
        "train_docs": len(train),
        "train_positive_rate": round(float(y_tr.mean()), 4),
        "test_docs": len(test),
        "test_positive": int(y_te.sum()),
        "vocabulary_size": n_tfidf,
        "test_lexicon": scores(y_te, pred_lex),
        "test_baseline_tfidf_only": scores(y_te, pred_base),
        "bootstrap": bootstrap(y_te, pred_base, pred_lex),
        "export_max_abs_diff": max_diff,
        "train_seconds": round(time.time() - t0, 1),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "model.json").write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
