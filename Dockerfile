FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    TORCH_THREADS=2 \
    HF_HOME=/home/app/.cache/huggingface

# Set BERT_REPO (e.g. hugo-guigo/bertimbau-misoginia-ptbr) to download the fine-tuned
# BERTimbau at startup. Without it the API serves the SVM alone.

# Non-root user (required by Hugging Face Spaces, good practice anywhere).
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/service

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app/ app/
COPY --chown=app:app static/ static/
COPY --chown=app:app artifacts/ artifacts/

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
