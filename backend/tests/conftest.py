import pytest
from app.core.limiter import limiter

@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limiting for all tests by default to prevent false failures."""
    limiter.enabled = False
    yield
    limiter.enabled = True
