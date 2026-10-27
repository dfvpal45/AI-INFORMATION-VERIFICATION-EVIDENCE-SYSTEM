FROM node:20-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_URL=sqlite:///./aivs.db \
    REDIS_URL=redis://localhost:6379/0 \
    ENABLE_TRANSFORMERS=false \
    ENABLE_SENTENCE_TRANSFORMERS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app
RUN chown app:app /app
COPY backend/requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app backend/app ./app
COPY --chown=app:app backend/data ./data
COPY --from=frontend-build --chown=app:app /build/frontend/dist ./frontend/dist
USER app
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
