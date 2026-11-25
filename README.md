# Misogyny Lens PT-BR

**Live demo (runs in your browser): https://hugo-guigo.github.io/misogyny-lens-ptbr**  
**Model: https://huggingface.co/hugo-guigo/bertimbau-misoginia-ptbr**

A research demo that flags traits of misogyny in Brazilian Portuguese text and shows, word by word, what drove each prediction. It turns my undergraduate research ([lexico-misoginia-ptbr](https://github.com/hugo-guigo/lexico-misoginia-ptbr)) into two deliverables: a quantized BERTimbau that runs entirely in the browser, and a tested, containerized API that serves the full SVM + BERTimbau ensemble.

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

## Browser build (the live demo)

The public demo has no server. `docs/` is a static page on GitHub Pages that downloads a quantized BERTimbau from the Hugging Face Hub and runs it with Transformers.js / ONNX Runtime Web. No text leaves the visitor's machine.

| Version | Size | Accuracy | F1 | Where it was measured |
|---|---|---|---|---|
| PyTorch fp32 | 436 MB | 87.4% | 0.819 | Python |
| ONNX int8, default settings | 110 MB | 79.3% | 0.673 | ONNX Runtime, x86 CPU |
| ONNX uint8 per-channel (deployed) | 110 MB | 85.6% | 0.797 | ONNX Runtime, x86 CPU |
| ONNX uint8 per-channel (deployed) | 110 MB | **85.6%** | **0.791** | **in the browser (WASM)**, `docs/eval.html` |

Default int8 quantization cost 8 points of accuracy. Signed int8 weights saturate on x86 CPUs without VNNI (per-channel int8 collapsed to 39.9%); unsigned per-channel weights keep the model within 1.8 points of fp32 at a quarter of the size. `docs/eval.html` reruns the 333-sentence evaluation in any visitor's browser. Export and quantization: `export_onnx.py`.

## What the API page shows

- the ensemble verdict and the probability of each model;
- **BERT occlusion:** each word is removed in turn and the drop in probability is its importance;
- **SVM exact split:** the linear model's score decomposes exactly into words + lexicon + bias (tests check it to 1e-9);

## Models

- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) fine-tuned on a balanced subsample of the train split (11,120 texts), max 64 tokens, lr 2e-5, fp16, 3 epochs on a Colab T4 GPU (1.1 min per epoch). See `colab/train_bert_colab.ipynb`.
- **Linear SVM** (scikit-learn `LinearSVC`, balanced) on TF-IDF of spaCy lemmas (`pt_core_news_sm`) plus the 8 features of the leak-free misogyny lexicon V3. Saved with joblib.
- **Ensemble:** logistic regression over the SVM decision and the BERT logit, fitted on validation. BERT carries most of the weight (0.66 vs 0.22).

## Latency (local, Intel i3-9100F, 2 torch threads, warm server)

| Path | Client p50 | Client p95 | Throughput |
|---|---|---|---|
| SVM only | 6.7 ms | 7.7 ms | 146 req/s |
| Ensemble + BERT occlusion | 146 ms | 188 ms | 7.4 req/s |

With 10 concurrent clients on the SVM path, p95 rises to 282 ms while server-side work stays near 3 ms: inference is CPU-bound in one Python process, so extra clients only queue. Scaling means more processes or instances, not more threads.

## API (Docker)

The full ensemble ships as a Docker image instead of a hosted endpoint. Measured in a container with BERT
loaded: 437 MiB idle and 619 MiB at peak under 10 concurrent clients. The `pip install` layer alone is 1.65 GB, most of it PyTorch.

```bash
docker build -t misogyny-lens . && docker run -p 8000:7860 -e BERT_REPO=hugo-guigo/bertimbau-misoginia-ptbr misogyny-lens
```


```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Lugar de mulher é na cozinha"}'
```

Other endpoints: `GET /api/health`, `GET /api/model-card`, `GET /api/live` and `GET /api/ready` (the
Kubernetes probes; with `REQUIRE_BERT=1`, ready returns 503 until BERT is loaded). Inputs must have 1 to
1,000 characters.

## Kubernetes (local kind cluster)

A local study cluster, not a hosted service: [kind](https://kind.sigs.k8s.io) (Kubernetes in a Docker
container, one node), manifests with Kustomize. Decisions and every measurement are in
[docs/specs/kubernetes.md](docs/specs/kubernetes.md).

```bash
kind create cluster --config k8s/kind.yaml
docker build -t misogyny-lens:base . && docker build -f Dockerfile.cluster -t misogyny-lens:cluster .
kind load docker-image misogyny-lens:cluster --name lens       # took ~8 min on the i3
kubectl apply -k k8s/metrics-server && kubectl apply -k k8s/base
kubectl port-forward svc/lens 8080:80
```

- **Image:** `Dockerfile.cluster` adds the fine-tuned BERT (436 MB) to the API image, so a new pod does not
  download it. The browser model (`artifacts/onnx`, 521 MB) left the API image: the `COPY artifacts/` layer
  went from 546 MB to 152 kB.
- **Probes:** the port only opens after SVM, spaCy and BERT load. A startup probe on `/api/live` holds the
  other two for up to 120 s, liveness restarts a hung process, readiness on `/api/ready` keeps a pod out of
  the Service until its models are loaded. Without probes the pod was "ready" 2 s after creation and the API
  answered 12 s after the container started; with them, Ready means the API answers (18 s from scheduling).
  A version without BERT and `REQUIRE_BERT=1` stayed alive, out of the Service, and the rollout stopped while
  the old pod kept serving.
- **Graceful shutdown:** with `sh -c "uvicorn ..."` as PID 1, SIGTERM never reached uvicorn and the pod took
  32 s to die (grace period, then SIGKILL). With `exec`, uvicorn is PID 1 and the pod leaves in 9 s (5 s
  `preStop` so the Service drops it first, then a clean shutdown).
- **Resources:** requests 1 CPU / 512 Mi, limits 2 CPU / 1 Gi, from a run without limits (peak 3.6 cores,
  619 MiB). The 2-core limit did not cost throughput (4.0 vs 4.1 req/s, one run each).
- **Rolling update under load** (`maxSurge: 1`, `maxUnavailable: 0`; 600 requests from a Job inside the
  cluster, through the Service): 0 errors in 2 runs. The same rollout without probes and `preStop`: 472
  connection errors (an inflated count, since a refused connection fails instantly and the clients do not
  wait, but it shows the outage).
- **HPA** (1 to 3 replicas at 70% of the CPU request): it scaled out in 34 s and back in ~90 s after the
  load. The third replica stays Pending: the system pods reserve 1,050m of the node's 4 cores. The new pod
  first received **0 requests**: a Service picks the pod per connection, and the load test's keep-alive
  connections were opened before the scale-out. With a new connection per request the new pod took 591 of
  1,500 requests and throughput went from 5.8 to 8.0 req/s, with the i3 then at 100%. Spreading kept-alive
  traffic needs per-request (layer 7) balancing, which this cluster does not have.

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

GitHub Actions runs the test suite, builds the Docker image, starts the container and calls `/api/health` and `/api/analyze` against it.
A second job builds the SVM-only image, creates a kind cluster, deploys `k8s/ci`, runs a smoke test through the Service and
a rolling update under load that must finish with 0 errors (`k8s/ci/testar.sh`). It does not test BERT, which is not in that
image. GitHub Pages serves `docs/` from `main`.

## Limitations

- The test set is small (333 sentences), so the confidence intervals are wide.
- BERT occlusion estimates importance; it is not an exact attribution.
- The ensemble still misses about 1 in 4 positive cases.
- The data comes from social media and carries the biases of its platforms and annotators.
- The Kubernetes numbers come from one node on one PC (i3-9100F, 4 cores), mostly one run per setting. The
  ensemble was saved with scikit-learn 1.6.1 and the image uses 1.9.1 (the API logs an `InconsistentVersionWarning`).

Built by [Hugo Guilherme de Assis Paula](https://hugo-guigo.github.io). Research advised by Prof. Deborah Silva Alves Fernandes (INF/UFG).
