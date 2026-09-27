"""Stacks the SVM and BERTimbau (and the LLM judge, when its predictions exist).

The stacker is a logistic regression fitted on the validation split only, with balanced
class weights: validation has 17% positives while the test set has 40%, so fitting the
validation prior would push the ensemble to under-predict on test.

Writes artifacts/ensemble.joblib and artifacts/ensemble_metrics.json.
Usage: python ensemble.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import SEED, load_splits

P = Path("artifacts/preds")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def scores(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4)}


def bootstrap_ci(y, pred, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[idx], pred[idx]))
        f1.append(f1_score(y[idx], pred[idx], zero_division=0))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"accuracy_ci95": ci(acc), "f1_ci95": ci(f1)}


def main():
    _, val, test, _ = load_splits(verbose=False)
    yv, yt = val["label"].values, test["label"].values
    feats_val = np.column_stack([np.load(P / "svm_val.npy"), logit(np.load(P / "bert_val.npy"))])
    feats_test = np.column_stack([np.load(P / "svm_test.npy"), logit(np.load(P / "bert_test.npy"))])

    stacker = LogisticRegression(class_weight="balanced", C=1.0)
    cv_pred = cross_val_predict(stacker, feats_val, yv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    stacker.fit(feats_val, yv)
    test_pred = stacker.predict(feats_test)

    result = {
        "features": ["svm_decision", "bert_logit"],
        "weights": dict(zip(["svm_decision", "bert_logit"], np.round(stacker.coef_[0], 4).tolist())),
        "intercept": round(float(stacker.intercept_[0]), 4),
        "val_5fold_cv": scores(yv, cv_pred),
        "test": {"svm": scores(yt, (feats_test[:, 0] > 0).astype(int)),
                 "bert": scores(yt, (feats_test[:, 1] > 0).astype(int)),
                 "ensemble": scores(yt, test_pred)},
        "test_ensemble_bootstrap": bootstrap_ci(yt, test_pred),
        "test_bert_bootstrap": bootstrap_ci(yt, (feats_test[:, 1] > 0).astype(int)),
    }
    joblib.dump(stacker, "artifacts/ensemble.joblib")
    Path("artifacts/ensemble_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
