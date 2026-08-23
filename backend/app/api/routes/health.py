"""Health-check route.

Milestone 1: confirms the API process is up and configuration loads correctly.
Milestone 2: adds a non-fatal database connectivity probe (db_status field).
             No auth, TMDB, or ML logic here.
"""

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.db.database import check_db_connection
from app.schemas.health import HealthResponse
from app.core.limiter import limiter

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
@limiter.exempt
def get_health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Return a simple confirmation that the API is running.

    - **status**: always ``"ok"`` — never reflects DB state so that load-balancer
      health checks don't trip during a database outage.
    - **db_status**: ``"connected"`` when PostgreSQL is reachable,
      ``"unavailable"`` otherwise.
    """
    db_reachable = check_db_connection()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        db_status="connected" if db_reachable else "unavailable",
    )
