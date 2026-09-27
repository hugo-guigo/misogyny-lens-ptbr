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

A research demo that flags traits of misogyny in Brazilian Portuguese text and explains every prediction word by word. It turns the classifier from my undergraduate research ([lexico-misoginia-ptbr](https://github.com/hugo-guigo/lexico-misoginia-ptbr)) into a tested, containerized API with a live page.

> ⚠ The model makes mistakes and reflects biases of its training data (social media comments). It must not be used to moderate or judge people.

## What it does

Type a sentence and the page shows, as you type:

- the verdict and the decision score;
- each word highlighted by how much it pushed the score up (orange) or down (green);
- the breakdown of the score: top words, the lexicon signal and the model bias;
- server latency and round-trip time.

The model is a linear SVM, so the explanation is exact: the word contributions, the lexicon term and the bias add up to the decision score (the tests check this to 1e-9).

## Model

- **Features:** TF-IDF of spaCy lemmas (`pt_core_news_sm`, content words only) plus the 8 features of the misogyny lexicon V3 (leak-free version).
- **Classifier:** `LinearSVC(class_weight="balanced")`.
- **Export:** the trained model is saved as plain JSON (vocabulary, IDF, weights, scaler, lexicon; 369 KB). The API rebuilds TF-IDF and the decision in NumPy, with no pickle and no scikit-learn at runtime. Training asserts that the export matches scikit-learn (max difference 9e-16).
- **Data:** unified corpus of HateBR, ToLD-BR and Portuguese Hate Speech, pinned to a commit of the research repo. The test set has 333 labeled ToLD-BR sentences (133 positive).
- **Leak control:** the test sentences also live in the unified corpus. Training drops them and their duplicates (371 rows) before fitting. Without this, F1 reaches about 0.87 by memorization.

## Results (333-sentence test set)

| Lemmatizer | Size | F1 | Precision | Recall | Lexicon gain over plain TF-IDF |
|---|---|---|---|---|---|
| `pt_core_news_sm` (deployed) | 16 MB | **0.704** (95% CI 0.632–0.765) | 0.820 | 0.617 | +0.011, p = 0.12 |
| `pt_core_news_md` | 52 MB | 0.712 (95% CI 0.643–0.774) | 0.830 | 0.624 | −0.000, p = 0.56 |
| `pt_core_news_lg` (research run) | 500+ MB | 0.748 | 0.848 | 0.669 | +0.018, p = 0.08 |

Confidence intervals and p-values come from a paired bootstrap with 2,000 resamples.

What the numbers say:

- The deployed model is 0.044 F1 below the research run. The large lemmatizer is 30 times heavier than the small one, which is too much for a free CPU host. The medium one does not close the gap.
- The lexicon's gain over plain TF-IDF is not statistically significant with any lemmatizer. Its value is interpretability, not accuracy.
- Precision is high and recall is low: when the model flags a text it is usually right, but it misses almost 4 in 10 cases.

## Latency (local, one uvicorn process)

`bench/load_test.py`, 7 example sentences, warm server:

| Clients | Throughput | Client p50 | Client p95 | Server p50 | Server p95 |
|---|---|---|---|---|---|
| 1 | 146 req/s | 6.7 ms | 7.7 ms | 2.9 ms | 8.0 ms |
| 10 | 126 req/s | 44.8 ms | 282 ms | 2.9 ms | 8.0 ms |

The work inside the server stays around 3 ms. Adding concurrent clients only adds queueing: inference is CPU-bound in one Python process, so threads compete for the GIL. Scaling means more processes (`--workers`) or instances, not more threads.

## API

```bash
curl -X POST https://hugo-guigo-misogyny-lens-ptbr.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Lugar de mulher é na cozinha"}'
```

Returns the label, the decision score, the bias, the lexicon contribution and its 8 features, and one entry per token with its character offsets, lemma and contribution. Inputs must have 1 to 1,000 characters. Other endpoints: `GET /api/health`, `GET /api/model-card`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                   # 23 tests
uvicorn app.main:app --port 8000            # http://localhost:8000
python train.py                             # retrain: downloads the pinned data, writes artifacts/
python bench/load_test.py --url http://localhost:8000 --concurrency 10
```

Or with Docker:

```bash
docker build -t misogyny-lens . && docker run -p 7860:7860 misogyny-lens
```

## CI

GitHub Actions runs the test suite, builds the Docker image, starts the container and calls `/api/health` and `/api/analyze` against it.

## Layout

```
app/text.py      preprocessing with offset-preserving cleaning (spans map back to the input)
app/lexicon.py   the 8 lexicon features
app/model.py     JSON-backed linear model: TF-IDF, decision, exact explanation
app/main.py      FastAPI app and static page
static/          demo page (plain HTML, CSS and JS)
train.py         data download, leak control, training, bootstrap, export
bench/           load test
tests/           pytest suite
```

## Limitations

- Unigram features miss irony, negation and context.
- Words like "feminista" appear in political speech and can trigger false positives.
- The test set is small (333 sentences), so metrics have wide confidence intervals.
- The training data comes from social media and carries its annotators' and platforms' biases.

Built by [Hugo Guilherme de Assis Paula](https://hugo-guigo.github.io). Research advised by Prof. Deborah Silva Alves Fernandes (INF/UFG).
