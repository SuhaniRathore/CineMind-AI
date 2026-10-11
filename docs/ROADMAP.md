# ROADMAP.md — CineMind AI

Each milestone should end in something runnable/testable — not just code that "should work." We do not start milestone N+1 until N is functionally verified.

## Current Verification Snapshot — October 11, 2026

- Compose API, PostgreSQL, and web containers are running; health reports the database connected.
- Backend, frontend, and core CRUD/recommendation code are implemented. Verification results are reported in the root README and latest project handoff, not inferred from milestone labels.
- NCF inference is conditionally wired to recommendations. Existing model artifacts and prepared shards are present, but the configured MovieLens 25M ratings CSV is header-only; fresh full-dataset training and evaluation remain unverified.
- Semantic search/pgvector and the remaining V2/V3 roadmap features remain planned or incomplete unless independently verified.

## Milestone 1 — Project Foundation
- Repo structure, `.gitignore`, `README.md` skeleton
- Backend package skeleton (routers/services/models/schemas/ml folders, empty but structured)
- `.env.example`, environment variable loading
- Docker Compose skeleton (api + db, no real logic yet)
- **Verify:** containers build and start; FastAPI returns a health-check `200 OK`

## Milestone 2 — Backend Setup
- FastAPI app instance, base routing, error handling, CORS config
- Pydantic settings/config module
- Basic logging setup
- **Verify:** `/health` and a placeholder `/movies` route respond correctly

## Milestone 3 — Database Setup
- SQLAlchemy engine/session setup, Alembic migrations
- Core models: `users`, `movies`, `genres`, `movie_genres`
- **Verify:** migrations run cleanly against the Dockerized Postgres; tables exist as designed

## Milestone 4 — Movie Data Integration
- TMDB client module (server-side, API key via env var)
- Sync/cache job: pull a movie subset from TMDB into our DB
- `GET /movies`, `GET /movies/{id}`, basic search/filter endpoints
- **Verify:** real TMDB-sourced movies queryable through our own API

## Milestone 5 — Authentication
- `users` table finalized, password hashing, JWT issue/verify
- `POST /auth/register`, `POST /auth/login`, protected route example
- **Verify:** tests cover register/login/invalid-token cases

## Milestone 6 — Ratings
- `ratings` table, `POST/GET/PUT/DELETE /ratings`
- **Verify:** a user can rate a movie and see it reflected; tests pass

## Milestone 7 — Reviews & Watchlist
- `reviews` and `watchlist` tables + endpoints
- **Verify:** CRUD flows tested end-to-end

## Milestone 8 — MVP Recommendations (Content-Based)
- Simple genre/metadata similarity recommender in the isolated ML package
- `GET /recommendations/{user_id}` (or similar) surfaces results
- **→ MVP considered feature-complete after this milestone. Pause for review before V2.**

## Milestone 9 — Frontend (MVP)
- React app: auth flow, movie browsing/search, rating/review UI, watchlist
- Consumes MVP API only
- **Verify:** full user flow works end-to-end locally

## Milestone 10 — Testing & Hardening (MVP)
- Expand test coverage (auth, ratings, reviews, recommendations)
- Basic input validation/security review (rate limiting, auth edge cases)

## Milestone 11 — MVP Dockerization & Deploy
- Full docker-compose (api + db + frontend/static)
- Deploy MVP to a chosen low-cost target (decision deferred per ARCHITECTURE.md)
- **→ MVP milestone; natural checkpoint to demo/pause.**

---

### V2 Milestones (after MVP approval)

12. NLP pipeline setup (sentiment + aspect-based sentiment on reviews, batch job)
13. Review summarization
14. Collaborative filtering recommender (MovieLens-trained) + hybrid blending
15. Embeddings + pgvector setup, semantic search endpoint
16. Explainable recommendations (surface reasoning in API + UI)
17. Movie DNA user preference profile
18. Analytics dashboard
19. Frontend polish for V2 features
20. V2 testing & hardening

### V3 Milestones (optional, time-permitting)

21. Mood-based recommendations
22. Movie comparison tool
23. Spoiler detection
24. Review credibility analysis
25. Voice-based search
26. Group recommendations
27. Production-grade deployment (CI/CD, monitoring)

---

## Notes
- Milestones 1–11 (MVP) are the priority and should each be small enough to complete and test within a short work session.
- V2/V3 ordering can be adjusted based on what proves most valuable to demo, but should be revisited explicitly (per Development Rules), not silently reordered mid-build.
