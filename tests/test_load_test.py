"""bench/load_test.py: o que conta como erro e o que é repetido (docs/specs/kubernetes.md, bloco 4)."""
import asyncio
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
import load_test  # noqa: E402


def rodar(falhas: list[Exception], **kw) -> dict:
    """1 requisição de aquecimento (sempre ok) e 1 medida; as exceções de `falhas` saem em ordem,
    uma por tentativa, e depois o transporte responde 200."""
    falhas = [None] + falhas

    def handler(request: httpx.Request) -> httpx.Response:
        f = falhas.pop(0) if falhas else None
        if f is not None:
            raise f
        return httpx.Response(200, json={"latency_ms": 1.0})
    return asyncio.run(load_test.run("http://lens", 1, 1, 0, transport=httpx.MockTransport(handler), **kw))


def test_conexao_fechada_e_repetida_uma_vez_com_retry_closed():
    r = rodar([httpx.ReadError("fechada")], retry_closed=True)
    assert r["errors"] == 0 and r["retried"] == 1 and r["retried_kinds"] == {"ReadError": 1}


def test_sem_retry_closed_a_conexao_fechada_conta_como_erro():
    r = rodar([httpx.RemoteProtocolError("sem resposta")])
    assert r["errors"] == 1 and r["error_kinds"] == {"RemoteProtocolError": 1} and r["retried"] == 0


def test_conexao_recusada_nunca_e_repetida():
    # ConnectError quer dizer que não há servidor: é o que a contraprova sem probes precisa mostrar.
    r = rodar([httpx.ConnectError("recusada")], retry_closed=True)
    assert r["errors"] == 1 and r["error_kinds"] == {"ConnectError": 1} and r["retried"] == 0


def test_repete_no_maximo_uma_vez():
    r = rodar([httpx.ReadError("1"), httpx.ReadError("2")], retry_closed=True)
    assert r["errors"] == 1 and r["retried"] == 1
