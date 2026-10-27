**** AI Information Verification & Evidence System**
An evidence-based fact-checking pipeline that takes a claim, a news URL or a screenshot, retrieves supporting evidence from multiple sources, and returns a verdict with a confidence score and a plain-language explanation with source links.

Built with: Python 3.11 · FastAPI · React + Tailwind CSS · PostgreSQL · Redis · FAISS · Docker

Table of Contents
Overview
Key Features
System Architecture
How It Works
Tech Stack
Project Structure
Getting Started
Configuration
API Usage
Model Training & Evaluation
Limitations
Roadmap
Contributing
License
Overview
Misinformation spreads quickly through forwarded messages, social media posts and news articles. AIVES helps users check whether a claim is supported by trustworthy evidence instead of relying on a single black-box prediction.

Users can submit:

a text claim,
a news article URL, or
an image (for example, a forwarded-message screenshot).
The system extracts the check-worthy claims, searches for evidence in parallel across fact-checking databases, open knowledge sources and a locally indexed trusted corpus, ranks that evidence, and uses a Natural Language Inference (NLI) model to decide whether each claim is supported, refuted, or has not enough information. A final verdict is produced using source-credibility weights, and a language model writes a short explanation strictly from the retrieved evidence.

Key Features
Multi-input support: text, URL scraping and image OCR.
Check-worthy claim detection: opinions and questions are discarded; only factual sentences are verified.
Parallel multi-source retrieval: published fact-checks, Wikipedia and recent news, plus a trusted-source vector index (government, WHO/UN, etc.).
Semantic evidence ranking: Sentence-BERT embeddings with top-k cosine similarity.
NLI-based verification: DeBERTa-v3 fine-tuned on FEVER/MNLI.
Credibility-weighted verdicts: results are aggregated using source-credibility scores.
Honest uncertainty: low-confidence results are marked "Unverified" instead of guessing.
Grounded explanations: retrieval-augmented LLM summaries that use only the retrieved evidence, to reduce hallucination.
Feedback loop: user feedback is stored in PostgreSQL for later analysis and model improvement.
Fast and scalable: Redis caching for repeated claims and API responses; async FastAPI pipeline.
System Architecture
1. User InputText claim / News URL /Image
2. PreprocessingOCR (Tesseract), cleaning,URL scraping
3. Claim ExtractionspaCy NER + check-worthysentence detection
4a. Fact-check APIsGoogle Fact Check Tools
4b. Open KnowledgeWikipedia + NewsAPI
4c. Trusted CorpusFAISS vector index
5. Evidence RankingSentence-BERT similarity(top-k)
6. Claim VerificationDeBERTa NLI: support /refute / not enough info
Source Credibility Scorer
7. Verdict + ConfidenceAggregation with credibilityweights
8. Explanation + ReportLLM summary with sourcelinks -> Web UI
PostgreSQLclaims, verdicts, userfeedback
Figure 1: Working architecture of the AI Information Verification & Evidence System.

How It Works
Step	Stage	Description
1	User input	The user types a claim, pastes a news URL or uploads an image through the web interface.
2	Preprocessing	Text is extracted from images using OCR; article text is scraped from URLs and boilerplate (menus, ads) is removed. Text is cleaned and split into sentences.
3	Claim extraction	Named entities, numbers, dates and quotations are identified. Only check-worthy factual sentences are kept; opinions and questions are discarded.
4	Evidence retrieval	Each claim is searched in parallel in (a) published fact-checks, (b) Wikipedia and recent news articles, and (c) a locally indexed corpus of trusted sources such as government and WHO/UN websites.
5	Evidence ranking	Claims and evidence passages are converted to embeddings, and the top-k most semantically similar passages are selected using cosine similarity.
6	Verification	An NLI model reads each (evidence, claim) pair and predicts entailment (supports), contradiction (refutes) or neutral (not enough information).
7	Verdict & confidence	Predictions are combined using source-credibility weights to produce a final verdict and a confidence percentage. Low-confidence results are marked "Unverified".
8	Explanation & report	A language model writes a short explanation strictly from the retrieved evidence. The result page shows the verdict, evidence snippets and source links. User feedback is saved in the database.
Tech Stack
Layer	Technology	Specific use in this project
Language	Python 3.11	Main language for the entire AI pipeline, data processing and evaluation scripts.
Frontend	React.js + Tailwind CSS	Web interface for submitting claims and displaying verdicts, evidence cards and confidence meters.
Backend API	FastAPI + Uvicorn	REST API that receives user input, runs the pipeline asynchronously and returns JSON results.
OCR	Tesseract OCR + OpenCV	OpenCV cleans and thresholds the uploaded image; Tesseract extracts text from screenshots.
Web scraping	Trafilatura + BeautifulSoup4	Downloads a news URL and extracts the main article text without ads and navigation.
NLP preprocessing	spaCy (en_core_web_sm)	Sentence splitting, named-entity recognition and detection of numbers/dates for claim extraction.
Embeddings	Sentence-Transformers (all-MiniLM-L6-v2)	Converts claims and evidence passages into 384-dimensional vectors for semantic matching.
Vector search	FAISS	Stores embeddings of the trusted-source corpus and returns the nearest passages in milliseconds.
Verification model	Hugging Face Transformers + DeBERTa-v3 (NLI, fine-tuned on FEVER/MNLI)	Classifies each claim-evidence pair as entailment, contradiction or neutral.
Fact-check source	Google Fact Check Tools API	Finds claims already reviewed by professional fact-checking organisations.
Knowledge sources	Wikipedia API + NewsAPI	Fetches encyclopaedic facts and recent news articles related to the claim.
Explanation	LLM API (Anthropic Claude API)	Writes a short, plain-language explanation using only the retrieved evidence passages (retrieval-augmented, to avoid hallucination).
Database	PostgreSQL	Stores claims, verdicts, evidence links, source-credibility scores and user feedback.
Caching / queue	Redis	Caches repeated claims and API responses to reduce latency and external API usage.
ML framework	PyTorch + scikit-learn	PyTorch runs/fine-tunes the models; scikit-learn computes accuracy, precision, recall and F1-score.
Deployment	Docker + GitHub	Docker packages the app for consistent deployment; GitHub provides version control and collaboration.
Project Structure
This is a suggested layout. Adjust it to match your repository.

aives/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── api/                    # REST routes (verify, feedback, health)
│   │   ├── pipeline/
│   │   │   ├── preprocess.py       # OCR, scraping, cleaning
│   │   │   ├── claim_extraction.py # spaCy NER + check-worthiness
│   │   │   ├── retrieval/
│   │   │   │   ├── factcheck.py    # Google Fact Check Tools
│   │   │   │   ├── open_knowledge.py  # Wikipedia + NewsAPI
│   │   │   │   └── trusted_corpus.py  # FAISS search
│   │   │   ├── ranking.py          # Sentence-BERT top-k
│   │   │   ├── verification.py     # DeBERTa NLI
│   │   │   ├── credibility.py      # Source credibility scorer
│   │   │   ├── verdict.py          # Weighted aggregation
│   │   │   └── explain.py          # LLM explanation (RAG)
│   │   ├── db/                     # SQLAlchemy models, migrations
│   │   └── core/                   # config, cache (Redis), logging
│   ├── scripts/
│   │   ├── build_corpus_index.py   # Build the FAISS index
│   │   ├── finetune_nli.py         # Fine-tune DeBERTa on FEVER/MNLI
│   │   └── evaluate.py             # Accuracy / precision / recall / F1
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/                        # React components (claim form, evidence cards, confidence meter)
│   └── package.json
├── data/
│   ├── corpus/                     # Trusted-source documents
│   └── index/                      # FAISS index files
├── docker-compose.yml
├── .env.example
└── README.md
Getting Started
Prerequisites
Python 3.11
Node.js 18+ and npm
PostgreSQL and Redis (or just Docker)
Tesseract OCR installed and available on your PATH
API keys for Google Fact Check Tools, NewsAPI and the Anthropic API
1. Clone the repository
bash
git clone https://github.com/<your-username>/aives.git
cd aives
2. Run with Docker (recommended)
bash
cp .env.example .env      # then fill in your API keys
docker compose up --build
Frontend: http://localhost:3000
Backend API docs (Swagger): http://localhost:8000/docs
3. Run locally (without Docker)
Backend

bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Build the trusted-corpus vector index
python scripts/build_corpus_index.py

# Start the API
uvicorn app.main:app --reload --port 8000
Frontend

bash
cd frontend
npm install
npm run dev
Configuration
Create a .env file in the project root (see .env.example):

env
# External APIs
GOOGLE_FACT_CHECK_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Data stores
DATABASE_URL=postgresql://user:password@localhost:5432/aives
REDIS_URL=redis://localhost:6379/0

# Pipeline settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
NLI_MODEL_PATH=./models/deberta-v3-nli
TOP_K_EVIDENCE=5
CONFIDENCE_THRESHOLD=0.60     # below this, the verdict is "Unverified"
Never commit your .env file or API keys to version control.

API Usage
POST /api/verify
Submit a claim, URL or image for verification.

bash
# Text claim
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"type": "text", "content": "The WHO declared COVID-19 a pandemic in March 2020."}'

# Image (e.g. forwarded-message screenshot)
curl -X POST http://localhost:8000/api/verify \
  -F "type=image" \
  -F "file=@screenshot.png"
Example response (illustrative)

json
{
  "claim": "The WHO declared COVID-19 a pandemic in March 2020.",
  "verdict": "Supported",
  "confidence": 0.93,
  "explanation": "WHO announced on 11 March 2020 that COVID-19 could be characterised as a pandemic...",
  "evidence": [
    {
      "snippet": "...",
      "source": "who.int",
      "credibility": 0.98,
      "label": "entailment",
      "url": "https://..."
    }
  ]
}
Possible verdicts: Supported, Refuted, Unverified.

POST /api/feedback
Save user feedback on a verdict so it can be reviewed and used to improve the system.

json
{ "claim_id": 123, "agrees_with_verdict": false, "comment": "Source is outdated." }
Model Training & Evaluation
The verification model is DeBERTa-v3 fine-tuned for NLI on the FEVER and MNLI datasets.

bash
# Fine-tune the NLI model
python backend/scripts/finetune_nli.py --epochs 3 --batch-size 16

# Evaluate
python backend/scripts/evaluate.py --model ./models/deberta-v3-nli --dataset fever_dev
Metrics are computed with scikit-learn:

Metric	Value
Accuracy	to be filled
Precision	to be filled
Recall	to be filled
F1-score	to be filled
Limitations
Verdict quality depends on evidence coverage: if no relevant evidence is found, the claim is marked "Unverified".
The trusted corpus must be kept up to date, and the credibility weights reflect design choices that may carry bias.
OCR accuracy drops on low-resolution, stylised or heavily compressed screenshots.
NLI models can struggle with sarcasm, implicit claims and complex multi-hop reasoning.
The default pipeline is tuned for English content.
AIVES is a decision-support tool, not a replacement for professional fact-checkers.
Roadmap
 Multilingual claim verification
 Multi-hop reasoning across several evidence passages
 Browser extension and messaging-app bot integration
 Active learning from stored user feedback
 Image manipulation / reverse-image-search checks
 Admin dashboard for managing sources and credibility scores
Contributing
Contributions are welcome.

Fork the repository
Create a feature branch: git checkout -b feature/your-feature
Commit your changes: git commit -m "Add your feature"
Push to the branch: git push origin feature/your-feature
Open a Pull Request
License
This project is licensed under the MIT License. See the LICENSE file for details. (Change this if you use a different license.)

Acknowledgements
Hugging Face Transformers and Sentence-Transformers
FAISS by Meta AI
FEVER and MNLI datasets
Google Fact Check Tools, Wikipedia and NewsAPI
Anthropic Claude API for evidence-grounded explanations

Claude finished the response
