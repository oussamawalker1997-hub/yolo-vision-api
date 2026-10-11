import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

@pytest.fixture
def client():
    """Fixture qui intercepte le chargement du modèle et injecte un Mock (simulateur)."""
    # On bloque l'initialisation du vrai modèle YOLO dans le fichier main
    with patch("app.main.ObjectDetector") as MockDetector:
        # On configure le faux modèle pour qu'il renvoie un résultat factice
        mock_instance = MockDetector.return_value
        mock_instance.predict.return_value = [
            {
                "class_name": "person",
                "confidence": 0.99,
                "box": {"x_min": 10.0, "y_min": 20.0, "x_max": 100.0, "y_max": 200.0}
            }
        ]
        
        # Le TestClient démarre maintenant instantanément sans charger PyTorch
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
