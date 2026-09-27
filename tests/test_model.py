import math

import pytest

from app.model import LinearTextModel

TEXTS = [
    "Lugar de mulher é na cozinha",
    "mulher mulher mulher, sempre a mesma coisa",
    "Hoje o jogo foi ótimo, parabéns ao time",
    "de o a em",
    "Ela é uma cientista brilhante e merece o prêmio",
]


@pytest.fixture(scope="module")
def model():
    return LinearTextModel.load()


@pytest.mark.parametrize("text", TEXTS)
def test_explanation_adds_up_to_the_decision(model, text):
    exp = model.explain(text)
    total = sum(t["contribution"] for t in exp.tokens) + exp.lexicon_contribution + exp.bias
    assert math.isclose(total, exp.decision, abs_tol=1e-9)


@pytest.mark.parametrize("text", TEXTS)
def test_explain_matches_direct_decision(model, text):
    exp = model.explain(text)
    lemmas = [t["lemma"] for t in exp.tokens if t["lemma"]]
    assert math.isclose(exp.decision, model.decision_from_lemmas(lemmas), abs_tol=1e-9)
    assert exp.label == int(exp.decision > 0)


def test_text_without_content_words_falls_back_to_bias_and_lexicon(model):
    exp = model.explain("de o a em")
    assert all(t["contribution"] == 0 for t in exp.tokens)
    assert math.isclose(exp.decision, exp.bias + exp.lexicon_contribution, abs_tol=1e-12)


def test_tfidf_vector_is_l2_normalized(model):
    x = model.vectorizer.transform(["mulher cozinha lugar mulher"])
    assert math.isclose(math.sqrt((x.data ** 2).sum()), 1.0, rel_tol=1e-12)
