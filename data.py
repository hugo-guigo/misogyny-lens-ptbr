"""Data loading shared by every model, with one fixed train / validation / test split.

Sources (research repo hugo-guigo/lexico-misoginia-ptbr, pinned commit):
- unified training corpus (HateBR, ToLD-BR, Portuguese Hate Speech)
- 333-sentence labeled test set (ToLD-BR)
- leak-free misogyny lexicon V3

Rules that keep the test number honest:
1. Leak control: test sentences also exist in the unified corpus, so every training row
   whose normalized text matches a test sentence is dropped.
2. Model selection (epochs, ensemble weights, thresholds) uses only the validation split.
3. The test set is scored once per final model.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.text import REGEX_NON_ALPHA, clean

COMMIT = "3cd80edb467bf6282a23880f3f62b482cd67113b"
BASE = f"https://raw.githubusercontent.com/hugo-guigo/lexico-misoginia-ptbr/{COMMIT}"
FILES = {
    "train": "corpus_unificado_final.csv",
    "test": "outputs/corpus_teste_frases.csv",
    "lexicon": "outputs/lexico_misoginia_v3_semente_pmi_leakfree.csv",
}
DATA = Path(__file__).resolve().parent / "data"
SEED = 42
VAL_FRACTION = 0.1


def fetch(key: str) -> Path:
    path = DATA / Path(FILES[key]).name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print(f"downloading {FILES[key]}")
        urllib.request.urlretrieve(f"{BASE}/{FILES[key]}", path)
    return path


def normalized(text: str) -> str:
    return " ".join(clean(text).split())


def load_splits(verbose: bool = True):
    """Returns (train, val, test) DataFrames with columns text, label; plus a stats dict."""
    corpus = pd.read_csv(fetch("train"))
    corpus = corpus[~corpus["text"].astype(str).str.fullmatch(r"\s*\d+\s*", na=False)].copy()
    corpus["text"] = corpus["text"].astype(str)
    corpus = corpus[corpus["text"].str.replace(REGEX_NON_ALPHA, "", regex=True).str.len() >= 3]
    corpus = corpus.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    n_dedup = len(corpus)

    test = pd.read_csv(fetch("test")).dropna(subset=["text", "label"])
    test = test[test["text"].astype(str).str.strip() != ""].reset_index(drop=True)
    test["label"] = test["label"].astype(int)
    test = test[["text", "label"]]

    leaked = corpus["text"].map(normalized).isin(set(test["text"].map(normalized)))
    corpus = corpus[~leaked].reset_index(drop=True)[["text", "label"]]

    train, val = train_test_split(corpus, test_size=VAL_FRACTION, stratify=corpus["label"], random_state=SEED)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    stats = {"docs_after_dedup": n_dedup, "leaked_rows_removed": int(leaked.sum()),
             "train_docs": len(train), "val_docs": len(val), "test_docs": len(test),
             "train_positive_rate": round(float(train["label"].mean()), 4),
             "test_positive": int(test["label"].sum())}
    if verbose:
        print(f"splits: train {len(train)} | val {len(val)} | test {len(test)} "
              f"({stats['leaked_rows_removed']} leaked rows removed)")
    return train, val, test, stats


def load_lexicon() -> dict[str, dict]:
    df = pd.read_csv(fetch("lexicon"))
    return {r.term: {"polarity": r.polarity, "score": float(r.score_norm),
                     "gender": float(r.gender_assoc) if pd.notna(r.gender_assoc) else 0.0}
            for r in df.itertuples()}
