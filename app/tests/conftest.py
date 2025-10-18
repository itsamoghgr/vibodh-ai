"""
Pytest Configuration and Fixtures
"""

import pytest
from fastapi.testclient import TestClient
from app.core.config import settings


@pytest.fixture
def test_client():
    """Create test client"""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_org_id():
    """Mock organization ID"""
    return "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def mock_user_id():
    """Mock user ID"""
    return "11111111-1111-1111-1111-111111111111"
