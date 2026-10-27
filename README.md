# AI Information Verification & Evidence System (AIVES)

A runnable full-stack starter implementation of the architecture in the supplied diagrams: input text/article URL/image → preprocessing/OCR → claim extraction → evidence retrieval → evidence ranking → optional NLI verification → verdict/confidence → explanation/report → PostgreSQL history and feedback.

## What is included

- **Frontend:** React 18, Vite, Tailwind CSS configuration, responsive dashboard, text/URL/image input, evidence cards, verdict and confidence, report history, feedback.
- **API:** Python 3.11, FastAPI, Uvicorn, request validation and API docs.
- **Database:** PostgreSQL by default in Docker Compose; SQLAlchemy schema for claims/reports/feedback. SQLite can be used for quick local backend-only testing.
- **Cache:** best-effort Redis response cache (app works if Redis is unavailable).
- **Preprocessing:** Trafilatura + BeautifulSoup article extraction, OpenCV + Tesseract OCR, spaCy sentence splitting/NER pipeline when model is installed.
- **Evidence:** a curated list of 50 news, fact-checking, science, government and reference publishers; live Google News RSS and public web-search results; plus optional Google Fact Check Tools and NewsAPI integrations. Wikipedia is included as a lower-weight reference source.
- **Ranking:** TF-IDF/cosine-similarity fallback. Optional Sentence-Transformers (`all-MiniLM-L6-v2`) semantic embeddings. `faiss-cpu` is included for extending the index to a larger trusted-source corpus; this demo's small corpus ranks live candidates directly to avoid a fragile first-run model download.
- **Verification:** conservative heuristic baseline by default; optional Hugging Face DeBERTa NLI model when `ENABLE_TRANSFORMERS=true`.
- **Explanation:** evidence-grounded template by default; optional Anthropic API explanation if an API key is supplied.
- **Packaging:** Dockerfiles, Docker Compose, environment templates, GitHub Actions CI workflow.

> **Important accuracy note:** This is an executable prototype, not a guaranteed fact detector. Similarity is not proof, and “Unverified” means the retrieved evidence was insufficient. Claims are sent to public search when looking for evidence. The 25–80% score estimates evidence strength from relevance, source credibility and source agreement; it is not the probability that a verdict is correct. External API keys and model downloads are optional, so the out-of-box baseline may return “Unverified” frequently. Do not present heuristic baseline outputs as authoritative fact-checks.

## Fastest startup (recommended): Docker

Requirements: Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux), with internet access for initial images/packages.

1. Extract the ZIP into a folder.
2. Open a terminal in the extracted `aivs_project` folder.
3. Copy `.env.example` to `.env` (optional; leave API keys blank for baseline mode).
4. Run:

   ```bash
   docker compose up --build
   ```

5. Open the frontend: **http://localhost:5173**
6. API interactive docs: **http://localhost:8000/docs**
7. Health check: **http://localhost:8000/api/health**

First startup may take a few minutes while dependencies install. Stop with `Ctrl+C`; use `docker compose down`. PostgreSQL data persists in the `postgres_data` volume.

## Free single-service deployment: Render

The root `Dockerfile` builds the Vite frontend and serves it from the FastAPI app, so one Render web service handles both the UI and API. The deployment uses a smaller dependency set and disables optional ML downloads by default.

1. Put this project in a Git repository that you control, then create a free Render account and connect that repository.
2. In Render, choose **New → Blueprint** and select the repository. Render reads `render.yaml` and creates the `aivs` free web service.
3. Wait for the first Docker build, then open the service URL. Check `/api/health` and `/docs`.

The free service can sleep when idle and has temporary local disk. The demo's SQLite report history can be lost when Render restarts or redeploys the service. No paid database or persistent disk is configured. Add API credentials only through Render's environment-variable settings; never put secrets in the repository. Availability and free-tier terms are controlled by Render and may change.

## Windows local development (without Docker)

Use Python 3.11 for best compatibility. Install Node.js 20+ and PostgreSQL if you want PostgreSQL; for a simple API smoke test, SQLite can be used.

### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

For SQLite, edit `backend/.env` and set:

```env
DATABASE_URL=sqlite:///./aivs.db
REDIS_URL=redis://localhost:6379/0
```

Start the API from the `backend` folder:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend (new terminal)

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The frontend expects the API at `http://localhost:8000` by default. Set `VITE_API_URL` if your API is elsewhere.

## Enable real external integrations / ML

Create `backend/.env` for local mode or the root `.env` for Compose mode.

- `GOOGLE_FACTCHECK_API_KEY`: Google Fact Check Tools API key. Follow Google's current API setup and quota rules.
- `NEWSAPI_KEY`: NewsAPI key for news article retrieval.
- `ANTHROPIC_API_KEY`: enables an LLM-written explanation grounded in retrieved snippets.
- `ENABLE_SENTENCE_TRANSFORMERS=true`: downloads `sentence-transformers/all-MiniLM-L6-v2` at first use. This can take time and disk space; leave false for TF-IDF fallback.
- `ENABLE_TRANSFORMERS=true`: enables a Hugging Face DeBERTa NLI classifier; first use downloads model weights and can require several GB of RAM/disk. Test model licensing and hardware suitability before deployment.

To use OCR locally on Windows, install Tesseract OCR and either add it to PATH or set `TESSERACT_CMD` to the full `tesseract.exe` path. The Docker image installs Tesseract.

## API routes

- `GET /api/health` — API status.
- `POST /api/verify/text` — JSON body: `{"text":"...","source_url":"https://..."}` (`source_url` optional).
- `POST /api/verify/url?url=https%3A%2F%2F...` — fetch and inspect an article.
- `POST /api/verify/image` — multipart upload with field name `file`.
- `GET /api/history?limit=20` — recent reports.
- `GET /api/history/{id}` — saved report.
- `POST /api/history/{id}/feedback` — JSON body: `{"feedback":"Helpful"}`.

## Troubleshooting

- **Frontend says API offline:** check `http://localhost:8000/api/health`; then ensure backend is running and CORS origins include the frontend address.
- **Docker port conflict:** stop another service using 5173, 8000, 5432 or 6379, or change the host-side port mappings in `docker-compose.yml`.
- **URL fetch fails:** some sites block automated requests, require JavaScript, or disallow scraping. Try a public article URL or paste its text.
- **OCR fails:** confirm Tesseract is installed; the screenshot must contain readable text.
- **Model first-run is slow:** keep optional model flags false for a light baseline. Enable large transformer models only on a machine with suitable RAM and disk.
- **No external evidence:** configure the optional API keys; the small built-in corpus is a starting point, not a comprehensive database.

## Production checklist

Before public deployment, add authentication/rate limits, secrets management, audit logging, stronger SSRF protections for URL fetching, file scanning, database migrations, request-size limits at the proxy, API usage budgets, model evaluation against a labeled test set, monitoring, and human review. Do not expose the development Vite server as the production frontend; build with `npm run build` and serve the generated `dist` directory through a production web server.
