import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Fixture qui charge FastAPI et le modèle YOLO uniquement pour les tests."""
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_predict_invalid_file_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"Ceci n'est pas une image", "text/plain")}
    )
    assert response.status_code == 400
