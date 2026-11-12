# ---- Stage 1: build the React frontend ---------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python API (serves the API and the built frontend) -------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Pre-download the tokenizer so the container needs no extra downloads at runtime.
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY --from=frontend /frontend/dist ./frontend/dist

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Credentials/model names are injected at runtime (docker run --env-file .env); none are baked in.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
