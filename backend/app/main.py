"""
CineMind AI — FastAPI application entrypoint.

Milestone 1 scope: app instance, CORS, and a single health-check route.
No auth, database models, TMDB integration, or ML/NLP logic is wired in yet
(see ROADMAP.md for the milestone plan).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Application factory so tests can create isolated app instances."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="AI-powered movie intelligence and discovery platform.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)

    return app


app = create_app()
