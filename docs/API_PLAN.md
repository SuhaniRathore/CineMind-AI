# API_PLAN.md — CineMind AI

Planning only — endpoints below are **not implemented yet**. Tier tags (MVP/V2/V3) indicate when each group becomes active, matching `REQUIREMENTS.md`.

## Authentication `/auth` — MVP
- `POST /auth/register` — create account
- `POST /auth/login` — issue JWT access token
- `POST /auth/refresh` — refresh token (if refresh-token flow is adopted)
- `GET /auth/me` — current user info (protected)

## Users `/users` — MVP (basic), V2 (profile depth)
- `GET /users/{id}` — public profile info
- `PATCH /users/me` — update own profile
- `GET /users/me/movie-dna` — V2: return preference profile / Movie DNA

## Movies `/movies` — MVP
- `GET /movies` — list/browse with filters (genre, year, sort)
- `GET /movies/{id}` — movie detail
- `GET /movies/{id}/similar` — MVP: content-based similar movies

## Ratings `/ratings` — MVP
- `POST /ratings` — create/update a rating for a movie
- `GET /ratings/me` — list current user's ratings
- `DELETE /ratings/{id}` — remove a rating

## Reviews `/reviews` — MVP (CRUD), V2 (analysis fields populated)
- `POST /reviews` — create a review
- `GET /movies/{id}/reviews` — list reviews for a movie (includes sentiment/spoiler/credibility fields once V2 lands, null until then)
- `PATCH /reviews/{id}` — edit own review
- `DELETE /reviews/{id}` — delete own review
- `GET /movies/{id}/reviews/summary` — V2: AI-generated summary of reviews for a movie

## Watchlist & History `/watchlist`, `/history` — MVP
- `POST /watchlist` / `DELETE /watchlist/{movie_id}` — add/remove
- `GET /watchlist/me`
- `POST /history` — log a watched movie
- `GET /history/me`

## Recommendations `/recommendations` — MVP (content-based) → V2 (hybrid + explainable) → V3 (mood/group)
- `GET /recommendations/me` — personalized recommendations (algorithm used depends on current tier)
- `GET /recommendations/me?explain=true` — V2: include reasoning payload
- `GET /recommendations/mood?mood=...` — V3
- `POST /recommendations/group` — V3: submit multiple user IDs, get group suggestion

## Search `/search` — MVP (basic) → V2 (semantic)
- `GET /search/movies?q=...` — MVP: title/keyword search
- `GET /search/semantic?q=...` — V2: natural-language semantic search via embeddings
- `POST /search/voice` — V3: accept audio, transcribe, run semantic search

## Analytics `/analytics` — V2
- `GET /analytics/movies/{id}` — rating distribution, sentiment breakdown for a movie
- `GET /analytics/trending` — trending movies based on recent activity

## Comparison `/compare` — V3
- `GET /compare?movie_ids=1,2` — side-by-side attribute + sentiment comparison

## Cross-Cutting Notes
- All non-auth endpoints that touch user-specific data require a valid JWT.
- Pydantic response models defined per endpoint (no raw ORM objects returned directly).
- Pagination (`limit`/`offset` or cursor-based — to decide at implementation) applied to all list endpoints from the start, even in MVP, to avoid a breaking change later.
- Endpoint list above is not final — additions should be reflected here before/while implementing (Development Rule 20).
