"""LLM judge tests with a fake client: no network, no API key, no cost."""
import json
from types import SimpleNamespace

from app.llm_judge import judge


class FakeClient:
    def __init__(self, stop_reason="end_turn", payload=None):
        self.calls = []
        text = json.dumps(payload or {})
        response = SimpleNamespace(stop_reason=stop_reason, model="claude-opus-5",
                                   content=[SimpleNamespace(type="text", text=text)])
        create = lambda **kw: (self.calls.append(kw), response)[1]
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_parses_structured_output():
    client = FakeClient(payload={"label": 1, "confidence": 0.9, "category": "estereotipizacao",
                                 "explanation": "Reforça papel doméstico."})
    j = judge("Lugar de mulher é na cozinha", client)
    assert (j.label, j.category, j.refused) == (1, "estereotipizacao", False)
    call = client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "Lugar de mulher" in call["messages"][0]["content"]


def test_refusal_is_reported_not_crashed():
    j = judge("qualquer texto", FakeClient(stop_reason="refusal"))
    assert j.refused and j.label == 0
