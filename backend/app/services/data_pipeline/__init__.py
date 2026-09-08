"""
Data pipeline package for Milestone 12 — Data Expansion & ML Foundation.

This package handles ingestion of external datasets (primarily MovieLens)
and TMDB enrichment into the CineMind-AI database.

Modules:
    movielens_loader  — parse and map MovieLens CSV files to our schema
    tmdb_enricher     — bulk-enrich movie rows with TMDB detail data
    seed_runner       — orchestrate the full pipeline end-to-end

Architecture notes:
  - All modules are pure service logic — no FastAPI imports.
  - Database access via SQLAlchemy Session (dependency-injected by callers).
  - TMDB client is injected for testability (same pattern as movie_service.py).
  - MovieLens rating data is NEVER written to the app 'ratings' table —
    the scale (0.5–5.0 float) and constraint (1–10 int) are incompatible.
    MovieLens data is used exclusively for training the collaborative model.
"""
