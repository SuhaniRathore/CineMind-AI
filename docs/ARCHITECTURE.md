# ARCHITECTURE.md — CineMind AI

## Guiding Principle

Start as a **modular monolith** (single FastAPI service, cleanly separated internal modules), not microservices. Microservices would over-engineer a BTech project and make it harder to explain and deploy. ML/NLP logic is kept in its own internal package so it *could* be split into a separate service later if needed, without a rewrite.

## High-Level Components

```
                          ┌─────────────────────┐
                          │      React SPA       │
                          │  (frontend, browser)  │
                          └──────────┬───────────┘
                                     │ REST (JSON) over HTTPS
                                     ▼
                          ┌─────────────────────┐
                          │     FastAPI App       │
                          │  ┌─────────────────┐  │
                          │  │  API Routers     │  │  ← auth, movies, ratings,
                          │  ├─────────────────┤  │     reviews, recommend, search
                          │  │  Service Layer   │  │  ← business logic
                          │  ├─────────────────┤  │
                          │  │  ML/NLP Package  │  │  ← recommenders, sentiment,
                          │  ├─────────────────┤  │     embeddings (called by services)
                          │  │  Data Access     │  │  ← SQLAlchemy models/repositories
                          │  └─────────────────┘  │
                          └──────────┬────────────┘
                                     │
                     ┌───────────────┼────────────────┐
                     ▼               ▼                ▼
             ┌───────────────┐ ┌───────────┐  ┌────────────────┐
             │ PostgreSQL DB  │ │ TMDB API  │  │ pgvector ext.   │
             │ (+ pgvector)   │ │ (external)│  │ (semantic search│
             │                │ │           │  │  lives inside DB│
             └────────────────┘ └───────────┘  └─────────────────┘
```

Note: pgvector is an extension inside the same PostgreSQL instance, not a separate store — shown separately above purely to highlight its role.

## Component Responsibilities

### Frontend (React)
- Consumes the FastAPI REST API only; no direct DB or TMDB access.
- Handles auth token storage/refresh, movie browsing, review/rating UI, search UI.
- Later (V2/V3): displays recommendation explanations, sentiment breakdowns, Movie DNA visualization.

### FastAPI Backend
- **API Routers** — thin, handle HTTP concerns (validation via Pydantic schemas, auth checks, status codes) and delegate to services.
- **Service layer** — business logic (e.g., "rate a movie," "get recommendations for user X"), orchestrates data access + ML modules.
- **ML/NLP package** — isolated from API code (per Development Rules). Exposes clean function/class interfaces (e.g., `recommend_for_user(user_id) -> list[MovieScore]`) so routers/services never touch model internals directly.
- **Data access layer** — SQLAlchemy models + repository-style query functions.

### PostgreSQL
- System of record for users, movies (cached from TMDB), ratings, reviews, watchlists, recommendation results.
- `pgvector` extension added when semantic search (V2) begins — avoids introducing a separate vector DB (e.g., Pinecone/FAISS service) and keeps operational complexity low, which matters for a student-run deployment.

### TMDB Integration
- Backend-only outbound calls to TMDB (never from frontend, to protect the API key).
- Movie metadata is fetched and cached into our DB (not queried live on every request) — keeps the app usable if TMDB rate-limits or is down, and satisfies "don't permanently rely on external mock/live data" concerns.

### ML Services (content-based, collaborative, hybrid recommenders)
- MVP: content-based similarity using movie metadata (genres, keywords, TMDB vectors) — simple, explainable, no training pipeline needed yet.
- V2: collaborative filtering trained offline (batch job/script) on MovieLens + real ratings, results cached in `recommendation_results` table rather than computed synchronously per request.
- Hybrid layer blends both scores.

### NLP Services (V2+)
- Sentiment/aspect-based sentiment, summarization, spoiler/credibility detection operate on review text, run as background/batch jobs (not blocking the request that creates a review), writing results back to the `reviews` table or a related analysis table.

### Embeddings / Vector Search (V2+)
- Sentence-transformer embeddings generated for movie overviews (and later, review content) for semantic search — stored as `vector` columns via pgvector, queried with cosine-distance similarity search from the `search` router.

### Authentication
- JWT access tokens issued by the `auth` router; passwords hashed (e.g., bcrypt/argon2 via passlib).
- Stateless auth; no server-side session store needed at this scale.

### Deployment Architecture (indicative, finalized later)
- Docker Compose for local dev: `api`, `db` (Postgres+pgvector), later `frontend` container or static hosting.
- Production target to be decided (Development Rule: don't decide silently) — candidates to evaluate later: a single VM (Railway/Render/EC2) running docker-compose, or split static frontend hosting + managed Postgres + containerized API. This decision is deferred until MVP is functionally complete.

## Communication Summary

- Frontend ↔ Backend: REST/JSON over HTTPS, JWT bearer auth.
- Backend ↔ PostgreSQL: SQLAlchemy (sync or async — to confirm at backend-setup milestone).
- Backend ↔ TMDB: server-side HTTPS calls, API key from environment variable, results cached in DB.
- ML/NLP modules: called in-process by the service layer (no separate microservice/API boundary at this stage) — kept in a clearly separated Python package so this boundary could become a real service boundary later without major rework.
