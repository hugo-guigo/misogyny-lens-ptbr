---
title: Misogyny Lens PT-BR
emoji: 🔎
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Misogyny Lens PT-BR

**Live demo: https://huggingface.co/spaces/hugo-guigo/misogyny-lens-ptbr** (API docs at `/docs`)

A research demo that flags traits of misogyny in Brazilian Portuguese text and shows, word by word, what drove each prediction. It turns my undergraduate research ([lexico-misoginia-ptbr](https://github.com/hugo-guigo/lexico-misoginia-ptbr)) into a tested, containerized API with a live page.

> ⚠ The models make mistakes and reflect biases of their training data (social media comments). They must not be used to moderate or judge people.

## Results (333-sentence held-out test set, scored once per final model)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Linear SVM (TF-IDF of lemmas + lexicon) | 79.0% | 0.696 | 82.5% | 60.2% |
| BERTimbau fine-tuned | 87.4% | 0.819 | 96.0% | 71.4% |
| **Ensemble (SVM + BERTimbau)** | **88.0%** | **0.831** | **95.2%** | **73.7%** |

Ensemble 95% bootstrap CI (2,000 resamples): accuracy 84.7%–91.3%, F1 0.778–0.879.

How the number stays honest:

- **Leak control.** The test sentences also live in the unified training corpus. Every training row whose normalized text matches a test sentence is dropped (371 rows). Without this, the research run reached F1 ≈ 0.87 by memorization.
- **Model selection on validation only.** A stratified 10% validation split picks the BERT epoch (epoch 2 of 3 won) and fits the ensemble weights. The test set is not used for any choice.
- **Distribution shift, measured.** Validation (17% positives, mixed sources) is harder than the test set (40% positives, ToLD-BR): the ensemble scores F1 0.67 on validation. The ensemble uses balanced class weights so the validation prior does not bias it on test.

## What the page shows

- the ensemble verdict and the probability of each model;
- **BERT occlusion:** each word is removed in turn and the drop in probability is its importance;
- **SVM exact split:** the linear model's score decomposes exactly into words + lexicon + bias (tests check it to 1e-9);
- an optional **second opinion from Claude**, with category and explanation, when the server has an API key.

## Models

- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) fine-tuned on a balanced subsample of the train split (11,120 texts), max 64 tokens, lr 2e-5, fp16, 3 epochs on a Colab T4 GPU (1.1 min per epoch). See `colab/train_bert_colab.ipynb`.
- **Linear SVM** (scikit-learn `LinearSVC`, balanced) on TF-IDF of spaCy lemmas (`pt_core_news_sm`) plus the 8 features of the leak-free misogyny lexicon V3. Saved with joblib.
- **Ensemble:** logistic regression over the SVM decision and the BERT logit, fitted on validation. BERT carries most of the weight (0.66 vs 0.22).
- **LLM judge** (`app/llm_judge.py`): Claude with the project's 7-class taxonomy and structured JSON output. `eval_llm.py` scores it on the test set when an API key is available.

## Latency (local, Intel i3-9100F, 2 torch threads, warm server)

| Path | Client p50 | Client p95 | Throughput |
|---|---|---|---|
| SVM only | 6.7 ms | 7.7 ms | 146 req/s |
| Ensemble + BERT occlusion | 146 ms | 188 ms | 7.4 req/s |

With 10 concurrent clients on the SVM path, p95 rises to 282 ms while server-side work stays near 3 ms: inference is CPU-bound in one Python process, so extra clients only queue. Scaling means more processes or instances, not more threads.

## API

```bash
curl -X POST https://hugo-guigo-misogyny-lens-ptbr.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Lugar de mulher é na cozinha"}'
```

Other endpoints: `GET /api/health`, `GET /api/model-card`, `POST /api/llm-explain`. Inputs must have 1 to 1,000 characters.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
uvicorn app.main:app --port 8000
```

Without `artifacts/bert/` the API serves the SVM alone. Set `BERT_REPO` to download the fine-tuned weights from the Hugging Face Hub at startup.

## Training

Training never runs on the serving machine.

1. `python train_svm.py` fits the SVM on the train split and saves its validation and test scores (about 1 minute on a CPU).
2. `colab/train_bert_colab.ipynb` runs `finetune_bert.py` (GPU only) and `ensemble.py` in Google Colab and returns the weights and metrics.

## CI

GitHub Actions runs the test suite, builds the Docker image, starts the container and calls `/api/health` and `/api/analyze` against it. Every green push to `main` is deployed to the Hugging Face Space.

## Limitations

- The test set is small (333 sentences), so the confidence intervals are wide.
- BERT occlusion estimates importance; it is not an exact attribution.
- The ensemble still misses about 1 in 4 positive cases.
- The data comes from social media and carries the biases of its platforms and annotators.

Built by [Hugo Guilherme de Assis Paula](https://hugo-guigo.github.io). Research advised by Prof. Deborah Silva Alves Fernandes (INF/UFG).
