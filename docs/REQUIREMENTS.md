# REQUIREMENTS.md — CineMind AI

Requirements are split into three tiers. **We build MVP first, completely, before touching V2.**
Each tier assumes the previous tier is done and stable.

---

## MVP — Minimum Viable Product

Goal: a working, deployable movie platform with real data, auth, ratings/reviews, and *one* working recommendation approach. No advanced NLP yet.

### Functional
- User registration/login (JWT-based auth, hashed passwords)
- Browse/search movies (basic text search by title, filter by genre/year)
- Movie detail page: poster, synopsis, cast, genres, release info (from TMDB)
- Users can rate movies (1–5) and write a text review
- Users can view their own ratings/reviews and edit/delete them
- Users can maintain a watchlist
- Basic content-based recommendations ("similar movies" using genre/metadata similarity)
- Movie data sourced from TMDB API (cached in our own PostgreSQL DB, not mocked long-term)

### Non-Functional
- FastAPI backend with modular structure (routes / services / models / schemas separated)
- PostgreSQL as system of record
- Dockerized for local dev (docker-compose: api + db)
- Basic automated tests for auth, ratings, and movie endpoints
- Environment-variable based secrets management
- Deployable (even if just to a single VM/container host at this stage)

### Explicitly Out of Scope for MVP
- Any ML/NLP beyond simple content-based similarity
- Semantic/embedding search
- Sentiment analysis, summarization, spoiler/credibility detection
- Voice search, group recommendations

---

## V2 — Advanced Features

Builds on a stable MVP. Introduces the "AI" in CineMind AI.

- Collaborative filtering recommendation (matrix factorization / user-item based, trained on MovieLens + real user ratings)
- Hybrid recommender combining content-based + collaborative signals
- Sentiment analysis on reviews (overall positive/negative/neutral)
- Aspect-based sentiment analysis (acting, plot, pacing, visuals, ending, etc.)
- AI-generated review summarization (per-movie "what reviewers are saying")
- Semantic movie search using sentence embeddings + pgvector
- Explainable recommendations (surface *why* a movie was recommended)
- User "Movie DNA" preference profile (derived taste vector, shown to user)
- Basic movie analytics dashboard (rating distributions, trending movies)

---

## V3 — Experimental / Standout Features

Optional, time-permitting, used to differentiate the project further.

- Mood-based recommendations ("I want something uplifting/dark/light")
- Movie comparison tool (side-by-side attribute + sentiment comparison)
- Spoiler detection in review text (auto-flag/hide spoilers)
- Review credibility analysis (detect low-effort/bot-like/spam reviews)
- Voice-based movie search (speech-to-text → NLP search pipeline)
- Group movie recommendations (aggregate preferences across multiple users)
- Full production deployment with CI/CD, monitoring, scaling considerations

---

## Prioritization Rule

If a feature from V2/V3 is tempting to build early because it's "interesting," it still waits until its tier is reached — per Development Rule: no premature scope expansion. This file is the source of truth for what's in scope at any given time; update it explicitly (not silently) if priorities change.
