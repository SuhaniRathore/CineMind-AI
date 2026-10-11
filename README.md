# CineMind AI

CineMind AI is an AI-powered movie discovery and recommendation platform developed as a final-year BTech Computer Science project.

The application combines a React frontend, FastAPI backend, PostgreSQL database, TMDB movie metadata, and multiple recommendation components.

## Current Project Status

- React frontend and FastAPI backend run locally through Docker Compose.
- Login/authentication and the PostgreSQL connection have been verified.
- The recommendation API returned 10 movies in a local smoke test.
- Backend tests: **166 passed** with 2 existing Pydantic deprecation warnings in the verified run; environment-dependent TMDB key checks are skipped when no key is configured.
- Frontend ESLint and production build both pass.
- Docker Compose services: API, web frontend, and database.

## Features

- User registration and login.
- Movie discovery with posters, metadata, genres, and rating information.
- TMDB movie metadata integration.
- Movie recommendation API.
- Popularity-based fallback recommendations for new users without ratings.
- Content-based recommendation components.
- Collaborative filtering, hybrid recommendation, and Neural Collaborative Filtering (NCF) components.
- Streaming NCF preprocessing and training utilities.
- Persistent Docker storage for database and ML data.

> **Recommendation status:** The live API loads NCF when `USE_DEEP_COLLAB=true`, the saved artifact is complete, and the user has at least `COLLAB_MIN_RATINGS` ratings. New users, users without a mapped training identity, and requests without collaborative signals use content-based or popularity fallback; the API labels the selected method.
>
> **Dataset/training status:** Persistent storage contains saved NCF artifacts and 102 train / 101 validation shards whose metadata records 22,513,700 training and 2,483,966 validation rows. The mounted `ml-25m/ratings.csv` is header-only, so these shards cannot currently be independently rebuilt or validated against their original source. The shard metadata is not proof that a fresh, full MovieLens 25M training run completed or was evaluated; that training status remains unverified. Do not trigger retraining until authentic source data, mapped movie coverage, output paths, overwrite risks, and resource requirements have been reviewed.

## Verified and Incomplete Work

- **Verified locally:** Compose API/database/web services respond; the API health check reports a connected database; the frontend production build succeeds.
- **Implemented, conditional:** NCF inference is wired into the authenticated recommendations endpoint, but falls back when the user is cold-start or the trained mapping has no usable candidates.
- **Incomplete / unverified:** fresh full-dataset NCF training and evaluation; production deployment; semantic search / pgvector; V2/V3 features not specifically marked implemented in their modules or routes.

## Technology Stack

| Component | Technologies |
|---|---|
| Backend | Python 3.11+, FastAPI, Pydantic |
| Database | PostgreSQL, SQLAlchemy, Alembic |
| Frontend | React |
| Movie data | TMDB API |
| Recommendation components | TF-IDF/content-based filtering, SVD, hybrid recommendation, PyTorch NCF |
| Development and deployment | Docker Compose, Git, pytest |

## Repository Structure

```text
CineMind-AI/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── ml/
│   │   ├── nlp/
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
├── ml-experiments/
├── docs/
├── docker-compose.yml
└── README.md
```

## Run with Docker Compose

### Prerequisites

- Docker Desktop
- Git

### Start the application

Run these commands from the project root:

```powershell
docker compose up -d --build
docker compose ps
```

### Local URLs

- Frontend: <http://localhost:5173>
- Backend health: <http://localhost:8000/health>
- Interactive API documentation: <http://localhost:8000/docs>

### Check API health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected fields include `status: ok` and `db_status: connected`.

### Stop the application

```powershell
docker compose down
```

This stops and removes the Compose containers and network while preserving named volumes. **Do not add `-v`** unless you deliberately intend to delete persistent volumes.

## Run Backend Tests Locally

From the project root, if the local virtual environment already exists:

```powershell
Set-Location .\backend
.\.venv\Scripts\python.exe -m pytest -q
```

For a first-time local setup, create and activate a virtual environment, then install dependencies:

```powershell
Set-Location .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure environment variables using `.env.example` as a template before starting services. Keep API keys, database credentials, and other secrets private; never commit a real `.env` file.

## Documentation

The `docs/` directory contains planning and technical documentation:

- `PROJECT_VISION.md`
- `REQUIREMENTS.md`
- `ARCHITECTURE.md`
- `DATABASE_DESIGN.md`
- `API_PLAN.md`
- `ML_NLP_PLAN.md`
- `ROADMAP.md`
- `DEVELOPMENT_RULES.md`

## Development Guidelines

- Keep API routes, services, database models, and ML components separated.
- Never hardcode or commit secrets.
- Run relevant tests after code changes.
- Preserve existing database and ML artifacts when changing Docker configuration.
- Document model-training status and limitations accurately.

See `docs/DEVELOPMENT_RULES.md` for the project's development guidelines.
