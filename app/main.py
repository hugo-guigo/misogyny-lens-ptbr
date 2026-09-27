"""HTTP API and static demo page.

GET  /api/health      liveness plus model info
GET  /api/model-card  test metrics produced by train.py
POST /api/analyze     prediction with a per-word explanation
GET  /                demo page (static/)
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import LinearTextModel
from app.text import SPACY_MODEL, analyze, get_nlp

ROOT = Path(__file__).resolve().parent.parent
MAX_CHARS = 1000
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = LinearTextModel.load(ROOT / "artifacts" / "model.json")
    trained_with = model.meta.get("spacy_model")
    if trained_with != SPACY_MODEL:
        raise RuntimeError(f"model trained with {trained_with} but server loads {SPACY_MODEL}")
    get_nlp()  # load spaCy at startup, not on the first request
    state["model"] = model
    state["metrics"] = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Misogyny Lens PT-BR", version="1.0.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class TokenOut(BaseModel):
    text: str
    start: int
    end: int
    lemma: str | None
    contribution: float
    in_vocabulary: bool
    lexicon: str | None


class AnalyzeResponse(BaseModel):
    label: int
    label_name: str
    decision: float
    bias: float
    lexicon_contribution: float
    lexicon_features: dict[str, float]
    tokens: list[TokenOut]
    latency_ms: float


@app.get("/api/health")
def health():
    model: LinearTextModel = state["model"]
    return {"status": "ok", "spacy_model": SPACY_MODEL, "vocabulary_size": len(model.vocab),
            "data_commit": model.meta.get("data_commit")}


@app.get("/api/model-card")
def model_card():
    return state["metrics"]


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")
    t0 = time.perf_counter()
    exp = state["model"].explain(req.text, analyze(req.text))
    latency = (time.perf_counter() - t0) * 1000
    return AnalyzeResponse(
        label=exp.label,
        label_name="misogyny traits" if exp.label else "no misogyny traits",
        decision=exp.decision, bias=exp.bias,
        lexicon_contribution=exp.lexicon_contribution,
        lexicon_features=exp.lexicon_features,
        tokens=exp.tokens, latency_ms=round(latency, 3),
    )


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")
