import pytest

from app import create_app


@pytest.fixture
def app():
    """Create and configure a testing Flask app instance."""
    app = create_app("testing")
    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_home_page(client):
    """Verify that the home page renders successfully with 200 OK and expected title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI-Powered Smart Study Assistant" in response.data


def test_health_check_endpoint(client):
    """Verify that the /health endpoint returns healthy JSON status."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["service"] == "AI-Powered Smart Study Assistant"
    assert json_data["upload_folder_ready"] is True
