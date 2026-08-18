"""Pydantic schemas for the health-check endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response body confirming the API is running."""

    status: str
    app_name: str
    environment: str
    db_status: str  # "connected" | "unavailable"
