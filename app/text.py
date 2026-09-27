"""Text preprocessing shared by training and inference.

Same filters as the research pipeline (lowercase, drop URLs, @users, numbers and
non-letters, keep content lemmas of 3+ letters that are not stopwords), with one
change: removed characters become spaces of the same length, so every kept lemma
still points to its exact span in the original text. That is what lets the demo
highlight the words that drove the prediction.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import spacy

# pt_core_news_sm by default; override with SPACY_MODEL to compare lemmatizers.
SPACY_MODEL = os.environ.get("SPACY_MODEL", "pt_core_news_sm")
ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "INTJ"}

REGEX_URL = re.compile(r"https?://\S+|www\.\S+")
REGEX_USER = re.compile(r"@\w+")
REGEX_NUM = re.compile(r"\b\d+\b")
REGEX_NON_ALPHA = re.compile(r"[^a-záéíóúâêôãõçàü ]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    lemma: str | None  # None when the token is filtered out


def _blank(match: re.Match) -> str:
    return " " * len(match.group())


def clean(text: str) -> str:
    """Lowercase and blank out noise while keeping every character offset."""
    lowered = "".join(c.lower() if len(c.lower()) == 1 else c for c in text)
    for rx in (REGEX_URL, REGEX_USER, REGEX_NUM, REGEX_NON_ALPHA):
        lowered = rx.sub(_blank, lowered)
    return lowered


@lru_cache(maxsize=1)
def get_nlp() -> spacy.Language:
    return spacy.load(SPACY_MODEL, disable=["parser", "ner"])


def _keep(tok) -> bool:
    lemma = tok.lemma_.lower()
    return (tok.pos_ in ALLOWED_POS and not tok.is_stop and not tok.is_punct
            and lemma.isalpha() and len(lemma) >= 3)


def tokens_from_doc(original: str, doc) -> list[Token]:
    out = []
    for tok in doc:
        if tok.is_space:
            continue
        start, end = tok.idx, tok.idx + len(tok.text)
        out.append(Token(original[start:end], start, end, tok.lemma_.lower() if _keep(tok) else None))
    return out


def analyze(text: str) -> list[Token]:
    return tokens_from_doc(text, get_nlp()(clean(text)))


def lemmas_batch(texts: list[str], batch_size: int = 256) -> list[list[str]]:
    """Fast path for training: only the kept lemmas of each text."""
    nlp = get_nlp()
    cleaned = (clean(t) for t in texts)
    return [[t.lemma for t in tokens_from_doc(orig, doc) if t.lemma]
            for orig, doc in zip(texts, nlp.pipe(cleaned, batch_size=batch_size))]
