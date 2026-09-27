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


async def run(url: str, n: int, concurrency: int, seed: int) -> dict:
    rng = random.Random(seed)
    queue: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(n):
        queue.put_nowait(rng.choice(SENTENCES))
    client_ms, server_ms, errors = [], [], 0

    async def worker(client: httpx.AsyncClient):
        nonlocal errors
        while not queue.empty():
            text = queue.get_nowait()
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{url}/api/analyze", json={"text": text})
                r.raise_for_status()
                client_ms.append((time.perf_counter() - t0) * 1000)
                server_ms.append(r.json()["latency_ms"])
            except httpx.HTTPError:
                errors += 1

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{url}/api/analyze", json={"text": "aquecimento"})  # warm-up
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.perf_counter() - t0

    summarize = lambda v: {"p50": round(percentile(v, 0.5), 2), "p95": round(percentile(v, 0.95), 2),
                           "p99": round(percentile(v, 0.99), 2), "mean": round(statistics.mean(v), 2)}
    return {"url": url, "requests": n, "concurrency": concurrency, "errors": errors,
            "throughput_rps": round(len(client_ms) / elapsed, 1),
            "client_latency_ms": summarize(client_ms), "server_latency_ms": summarize(server_ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(json.dumps(asyncio.run(run(args.url.rstrip("/"), args.requests, args.concurrency, args.seed)), indent=2))


if __name__ == "__main__":
    main()
