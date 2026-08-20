"""
CineMind AI — FastAPI application entrypoint.

Milestone 5 scope: auth and users routers registered under /api/v1/auth
and /api/v1/users.  ML/NLP and frontend features are not wired in yet
(see ROADMAP.md for the milestone plan).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth
from app.api.routes import health
from app.api.routes import movies
from app.api.routes import users
from app.api.routes import ratings
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
    app.include_router(movies.router, prefix=settings.api_v1_prefix)
    app.include_router(auth.router, prefix=settings.api_v1_prefix)
    app.include_router(users.router, prefix=settings.api_v1_prefix)
    app.include_router(ratings.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()

