"""HTTP API and static demo page.

GET  /api/health      status plus which models are loaded (kept for the CI and the README)
GET  /api/live        the process answers (Kubernetes startup and liveness probes)
GET  /api/ready       200 only when the models this deploy requires are loaded (readiness probe)
GET  /api/model-card  test metrics of every model
POST /api/analyze     ensemble verdict + SVM exact explanation + BERT occlusion
GET  /                demo page (static/)

BERTimbau is optional: without artifacts/bert the API serves the SVM alone. A deploy that must
serve BERT sets REQUIRE_BERT=1: if the weights are missing, the process stays alive but not ready,
so Kubernetes sends it no traffic and restarting it would not help.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import LinearTextModel
from app.text import SPACY_MODEL, analyze, get_nlp

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MAX_CHARS = 1000
state: dict = {}


def _read_json(name: str) -> dict | None:
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = LinearTextModel.load(ART / "svm.joblib")
    trained_with = model.meta.get("spacy_model")
    if trained_with != SPACY_MODEL:
        raise RuntimeError(f"model trained with {trained_with} but server loads {SPACY_MODEL}")
    get_nlp()  # load spaCy at startup, not on the first request
    state["svm"] = model
    state["bert"] = None
    state["ensemble"] = None
    # The fine-tuned weights (436 MB) live on the Hugging Face Hub, not in git.
    bert_repo = os.environ.get("BERT_REPO")
    if bert_repo and not (ART / "bert" / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(bert_repo, local_dir=ART / "bert")
    if os.environ.get("ENABLE_BERT", "1") == "1" and (ART / "bert" / "model.safetensors").exists():
        from app.bert_model import BertModel
        state["bert"] = BertModel(ART / "bert", threads=int(os.environ.get("TORCH_THREADS", "2")))
        if (ART / "ensemble.joblib").exists():
            state["ensemble"] = joblib.load(ART / "ensemble.joblib")
    state["require_bert"] = os.environ.get("REQUIRE_BERT", "0") == "1"
    state["card"] = {"svm": _read_json("svm_metrics.json"), "bert": _read_json("bert_metrics.json"),
                     "ensemble": _read_json("ensemble_metrics.json")}
    yield
    state.clear()


app = FastAPI(title="Misogyny Lens PT-BR", version="2.0.0", lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


class TokenOut(BaseModel):
    text: str
    start: int
    end: int
    lemma: str | None
    contribution: float
    in_vocabulary: bool
    lexicon: str | None


class WordImportance(BaseModel):
    text: str
    start: int
    end: int
    importance: float


class AnalyzeResponse(BaseModel):
    label: int
    label_name: str
    verdict_model: str
    probability: float | None
    svm_decision: float
    svm_label: int
    bert_probability: float | None
    bias: float
    lexicon_contribution: float
    lexicon_features: dict[str, float]
    tokens: list[TokenOut]
    bert_words: list[WordImportance] | None
    latency_ms: float


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@app.get("/api/health")
def health():
    svm: LinearTextModel = state["svm"]
    return {"status": "ok", "spacy_model": SPACY_MODEL, "vocabulary_size": len(svm.vocab),
            "bert_loaded": state["bert"] is not None, "ensemble_loaded": state["ensemble"] is not None,
            "data_commit": svm.meta.get("data_commit")}


# The probes are async on purpose: sync routes run in the threadpool, where under load they could wait
# behind /api/analyze requests and time out, and the liveness probe would restart a healthy pod.
@app.get("/api/live")
async def live():
    # No model work here: a slow or failing model must not get a healthy process restarted.
    return {"status": "alive"}


@app.get("/api/ready")
async def ready():
    missing = []
    if state.get("svm") is None:
        missing.append("svm")
    if state.get("require_bert") and state.get("bert") is None:
        missing.append("bert")
    if missing:
        return JSONResponse(status_code=503, content={"status": "not ready", "missing": missing})
    return {"status": "ready"}


@app.get("/api/model-card")
def model_card():
    return state["card"]


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_text(req: TextIn):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")
    t0 = time.perf_counter()
    exp = state["svm"].explain(req.text, analyze(req.text))

    bert_p, bert_words, prob, label, verdict = None, None, None, exp.label, "svm"
    if state["bert"] is not None:
        occ = state["bert"].explain(req.text)
        bert_p, bert_words = occ["probability"], occ["words"]
        label, verdict = occ["label"], "bert"
        if state["ensemble"] is not None:
            feats = np.array([[exp.decision, _logit(bert_p)]])
            prob = float(state["ensemble"].predict_proba(feats)[0, 1])
            label, verdict = int(prob >= 0.5), "ensemble"
        else:
            prob = bert_p

    return AnalyzeResponse(
        label=label, label_name="misogyny traits" if label else "no misogyny traits",
        verdict_model=verdict, probability=prob,
        svm_decision=exp.decision, svm_label=exp.label, bert_probability=bert_p,
        bias=exp.bias, lexicon_contribution=exp.lexicon_contribution,
        lexicon_features=exp.lexicon_features, tokens=exp.tokens, bert_words=bert_words,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")
