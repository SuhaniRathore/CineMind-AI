# CineMind AI

AI-powered movie intelligence and discovery platform — a final-year BTech project.

Full planning documentation lives in [`docs/`](./docs): `PROJECT_VISION.md`, `REQUIREMENTS.md`,
`ARCHITECTURE.md`, `ROADMAP.md`, `DEVELOPMENT_RULES.md`, `DATABASE_DESIGN.md`, `API_PLAN.md`,
`ML_NLP_PLAN.md`. Read `ROADMAP.md` and `DEVELOPMENT_RULES.md` before making any changes.

## Project Status

**Milestone 1: Project Foundation** — repository structure, FastAPI skeleton, and a health-check
endpoint only. No auth, database, TMDB integration, ML/NLP, or frontend functionality yet.

## Repository Structure

```
cinemind-ai/
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── api/routes/     # HTTP route handlers (thin, delegate to services)
│   │   ├── core/           # Settings/config
│   │   ├── db/             # DB engine/session, migrations (Milestone 3+)
│   │   ├── models/         # SQLAlchemy ORM models (Milestone 3+)
│   │   ├── schemas/        # Pydantic request/response schemas
│   │   ├── services/       # Business logic (Milestone 4+)
│   │   ├── ml/             # Recommendation algorithms, isolated from API code (Milestone 8+)
│   │   ├── nlp/             # Sentiment/summarization/etc., isolated from API code (Milestone 12+)
│   │   ├── utils/          # Small shared helpers
│   │   └── main.py         # FastAPI app entrypoint
│   ├── tests/               # Backend tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/                # React app (skeleton only for now)
├── ml-experiments/          # Notebooks and exploratory ML work, outside the app package
├── docs/                    # Planning/architecture documentation
├── .gitignore
└── README.md
```

## Running the Backend Locally

Requires Python 3.11+.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env with real local values as needed

uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

### Verify the health endpoint

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok", "app_name": "CineMind AI", "environment": "development"}
```

### Run tests

```bash
cd backend
pytest
```

## Frontend

Not yet implemented — scaffolding arrives at Milestone 9 per `docs/ROADMAP.md`.

## Contributing / Development Rules

All contributors (human or AI) must follow `docs/DEVELOPMENT_RULES.md` — notably: build one
milestone at a time, never hardcode secrets, keep API/services/models/ML/NLP code separated,
and don't make architectural decisions silently.
