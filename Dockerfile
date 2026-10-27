# Stage 1: build the React dashboard
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: API (serves the built dashboard too, so one service is enough on a free host)
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY backend/ backend/
COPY experiments/ experiments/
COPY models/ models/
COPY --from=web /web/dist frontend/dist
ENV HF_HOME=/app/hf_cache
# Pre-download the 17 MB base model so the first /predict is fast
RUN python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; T.from_pretrained('prajjwal1/bert-tiny'); M.from_pretrained('prajjwal1/bert-tiny', num_labels=2)"
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
