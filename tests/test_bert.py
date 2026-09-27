"""BERTimbau inference tests. Skipped when the fine-tuned weights are not present (e.g. in CI)."""
from pathlib import Path

import pytest

BERT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "bert"
pytestmark = pytest.mark.skipif(not (BERT_DIR / "model.safetensors").exists(), reason="fine-tuned BERT not available")


@pytest.fixture(scope="module")
def bert():
    from app.bert_model import BertModel
    return BertModel(BERT_DIR, threads=2)


def test_probabilities_are_valid(bert):
    p = bert.proba(["Lugar de mulher é na cozinha", "Hoje choveu em Goiânia"])
    assert ((p >= 0) & (p <= 1)).all()


def test_occlusion_has_one_entry_per_word_with_spans(bert):
    text = "Mulher no volante, perigo constante"
    out = bert.explain(text)
    assert [w["text"] for w in out["words"]] == ["Mulher", "no", "volante", "perigo", "constante"]
    for w in out["words"]:
        assert text[w["start"]:w["end"]] == w["text"]
    assert out["label"] == int(out["probability"] >= 0.5)
