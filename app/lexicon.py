"""The 8 lexicon features from the research pipeline (misogyny lexicon V3, leak-free)."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean_score", "misog_ratio", "match_count", "max_score",
    "sum_score", "gender_weighted", "mean_gender", "net_polarity",
]


def features(lemmas: list[str], lexicon: dict[str, dict]) -> np.ndarray:
    """One 8-dim vector per document.

    0 mean_score       mean score of matched misogyny terms
    1 misog_ratio      share of lemmas that are misogyny terms
    2 match_count      number of matched misogyny terms
    3 max_score        highest matched score
    4 sum_score        total matched score
    5 gender_weighted  sum of score x gender association (gender-anchored signal)
    6 mean_gender      mean gender association of matched terms
    7 net_polarity     (misogyny matches - non-misogyny matches) / len(doc)
    """
    f = np.zeros(len(FEATURE_NAMES))
    n = max(len(lemmas), 1)
    scores, genders, non_count = [], [], 0
    for lemma in lemmas:
        info = lexicon.get(lemma)
        if info is None:
            continue
        if info["polarity"] == "misogino":
            scores.append(info["score"])
            genders.append(info["gender"])
        else:
            non_count += 1
    if scores:
        s, g = np.asarray(scores), np.asarray(genders)
        f[0] = s.mean()
        f[1] = len(s) / n
        f[2] = len(s)
        f[3] = s.max()
        f[4] = s.sum()
        f[5] = (s * g).sum()
        f[6] = g.mean()
    f[7] = (len(scores) - non_count) / n
    return f
