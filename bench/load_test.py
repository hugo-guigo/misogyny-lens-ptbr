"""Load test for /api/analyze: N requests with C concurrent clients, reports latency percentiles.

Usage:
    python bench/load_test.py --url http://localhost:8000 --requests 500 --concurrency 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter

import httpx

SENTENCES = [
    "Lugar de mulher é na cozinha",
    "Mulher no volante, perigo constante",
    "Ela é uma cientista brilhante e merece o prêmio",
    "O juiz roubou o jogo de ontem, que vergonha",
    "Hoje choveu muito aqui em Goiânia e o trânsito parou",
    "Essas feministas só sabem reclamar",
    "Parabéns pela aprovação no mestrado, você merece demais",
]


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# A kept-alive connection that the server closed under the request (a pod shutting down during a
# rolling update closes its idle connections). /api/analyze changes nothing, so retrying is safe; an
# endpoint that writes would need an idempotency key first. A refused connection (ConnectError) is
# never retried: it means no server is there, which is what the test must catch.
CLOSED_CONNECTION = (httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError)


async def run(url: str, n: int, concurrency: int, seed: int, keepalive: bool = True,
              retry_closed: bool = False, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    rng = random.Random(seed)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(n):
        queue.put_nowait(rng.choice(SENTENCES))
    client_ms, server_ms, errors = [], [], 0
    error_kinds: Counter[str] = Counter()  # e.g. "ConnectError" or "HTTP 503": tells a refused
    # connection apart from a closed one or a server error (docs/specs/kubernetes.md, bloco 3)

    retried_kinds: Counter[str] = Counter()  # closed connections retried once (retry_closed)

    async def worker(client: httpx.AsyncClient):
        nonlocal errors
        while not queue.empty():
            text = queue.get_nowait()
            t0 = time.perf_counter()  # latency includes the retry, as a client would see it
            for attempt in (1, 2):
                try:
                    r = await client.post(f"{url}/api/analyze", json={"text": text})
                    r.raise_for_status()
                    client_ms.append((time.perf_counter() - t0) * 1000)
                    server_ms.append(r.json()["latency_ms"])
                except httpx.HTTPStatusError as e:
                    errors += 1
                    error_kinds[f"HTTP {e.response.status_code}"] += 1
                except CLOSED_CONNECTION as e:
                    if retry_closed and attempt == 1:
                        retried_kinds[type(e).__name__] += 1
                        continue
                    errors += 1
                    error_kinds[type(e).__name__] += 1
                except httpx.HTTPError as e:
                    errors += 1
                    error_kinds[type(e).__name__] += 1
                break

    # keepalive=False opens a new connection per request. Behind a Kubernetes Service the pod is
    # chosen per connection, so kept-alive connections opened before a scale-up never reach the
    # new pods (docs/specs/kubernetes.md, bloco 3).
    limits = httpx.Limits() if keepalive else httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=30, limits=limits, transport=transport) as client:
        try:
            await client.post(f"{url}/api/analyze", json={"text": "aquecimento"})  # warm-up
        except httpx.HTTPError:
            pass  # the run below counts the failures
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.perf_counter() - t0

    def summarize(v: list[float]) -> dict | None:
        if not v:  # every request failed: still report errors and error_kinds
            return None
        return {"p50": round(percentile(v, 0.5), 2), "p95": round(percentile(v, 0.95), 2),
                "p99": round(percentile(v, 0.99), 2), "mean": round(statistics.mean(v), 2)}
    return {"url": url, "requests": n, "concurrency": concurrency, "keepalive": keepalive, "retry_closed": retry_closed,
            "errors": errors, "retried": sum(retried_kinds.values()), "retried_kinds": dict(retried_kinds),
            "error_kinds": dict(error_kinds), "elapsed_s": round(elapsed, 1),
            "throughput_rps": round(len(client_ms) / elapsed, 1),
            "client_latency_ms": summarize(client_ms), "server_latency_ms": summarize(server_ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-keepalive", action="store_true", help="new connection per request")
    ap.add_argument("--retry-closed", action="store_true",
                    help="retry once when the server closed a kept-alive connection under the request")
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args.url.rstrip("/"), args.requests, args.concurrency, args.seed,
                                     keepalive=not args.no_keepalive, retry_closed=args.retry_closed)),
                     indent=2))


if __name__ == "__main__":
    main()
