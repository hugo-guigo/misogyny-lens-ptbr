import pytest
from fastapi.testclient import TestClient

from app.main import MAX_CHARS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["vocabulary_size"] > 1000


def test_analyze_returns_label_and_spans(client):
    text = "Lugar de mulher é na cozinha"
    body = client.post("/api/analyze", json={"text": text}).json()
    assert body["label"] in (0, 1)
    assert body["latency_ms"] >= 0
    for tok in body["tokens"]:
        assert text[tok["start"]:tok["end"]] == tok["text"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "a" * (MAX_CHARS + 1)}, {}])
def test_rejects_invalid_input(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 422


def test_model_card_reports_test_metrics(client):
    card = client.get("/api/model-card").json()
    assert 0 < card["test_lexicon"]["f1"] <= 1
    assert card["leaked_rows_removed"] > 0


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Misogyny Lens" in r.text
